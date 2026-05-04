use std::{
    fmt,
    io::{Read, Seek, SeekFrom, Write},
};

use anyhow::{anyhow, Result};
use serde::{
    de::{DeserializeSeed, SeqAccess, Visitor},
    Deserialize, Deserializer,
};

use crate::{
    index::{encode_record, write_header, write_ivf_header, IVF_LABEL_OFFSET, IVF_RECORD_LEN},
    vector::{quantize_i16, QuantizedI16Vector, Vector, DIMS},
};

const DEFAULT_IVF_CLUSTERS: usize = 256;
const DEFAULT_KMEANS_SAMPLE: usize = 32_768;
const DEFAULT_KMEANS_ITERS: usize = 6;

#[derive(Deserialize)]
struct ReferenceRecord {
    vector: Vector,
    label: String,
}

#[derive(Clone, Copy)]
struct QuantizedReference {
    vector: QuantizedI16Vector,
    label: u8,
}

pub fn build_index_from_json_reader(
    reader: impl Read,
    mut writer: impl Write + Seek,
) -> Result<u64> {
    build_ivf_index_from_json_reader(reader, &mut writer)
}

pub fn build_exact_index_from_json_reader(
    reader: impl Read,
    mut writer: impl Write + Seek,
) -> Result<u64> {
    write_header(&mut writer, 0)?;
    let mut deserializer = serde_json::Deserializer::from_reader(reader);
    let count = WriteIndexSeed {
        writer: &mut writer,
    }
    .deserialize(&mut deserializer)?;

    writer.flush()?;
    writer.seek(SeekFrom::Start(0))?;
    write_header(&mut writer, count)?;
    writer.flush()?;
    Ok(count)
}

pub fn build_ivf_index_from_json_reader(
    reader: impl Read,
    mut writer: impl Write + Seek,
) -> Result<u64> {
    let records = read_quantized_references(reader)?;
    if records.is_empty() {
        write_ivf_header(&mut writer, 0, 1)?;
        write_centroids(&mut writer, &[[0.0; DIMS]])?;
        write_offsets(&mut writer, &[0, 0])?;
        write_bboxes(&mut writer, &[[0; DIMS]])?;
        write_bboxes(&mut writer, &[[0; DIMS]])?;
        writer.flush()?;
        return Ok(0);
    }

    let cluster_count = env_usize("IVF_CLUSTERS", DEFAULT_IVF_CLUSTERS)
        .clamp(1, records.len())
        .min(u32::MAX as usize);
    let sample_size = env_usize("IVF_SAMPLE", DEFAULT_KMEANS_SAMPLE).clamp(1, records.len());
    let iterations = env_usize("IVF_KMEANS_ITERS", DEFAULT_KMEANS_ITERS).max(1);
    let centroids = train_centroids(&records, cluster_count, sample_size, iterations);
    let assignments = assign_records(&records, &centroids);
    let layout = build_cluster_layout(&records, assignments, cluster_count);

    write_ivf_header(&mut writer, records.len() as u64, cluster_count as u32)?;
    write_centroids(&mut writer, &centroids)?;
    write_offsets(&mut writer, &layout.offsets)?;
    write_bboxes(&mut writer, &layout.bbox_min)?;
    write_bboxes(&mut writer, &layout.bbox_max)?;
    write_records(&mut writer, &layout.records)?;
    writer.flush()?;
    Ok(records.len() as u64)
}

struct WriteIndexSeed<'a, W> {
    writer: &'a mut W,
}

impl<'de, W> DeserializeSeed<'de> for WriteIndexSeed<'_, W>
where
    W: Write,
{
    type Value = u64;

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(WriteIndexVisitor {
            writer: self.writer,
        })
    }
}

struct WriteIndexVisitor<'a, W> {
    writer: &'a mut W,
}

impl<'de, W> Visitor<'de> for WriteIndexVisitor<'_, W>
where
    W: Write,
{
    type Value = u64;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an array of reference vectors")
    }

    fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut count = 0u64;
        while let Some(record) = seq.next_element::<ReferenceRecord>()? {
            let encoded =
                encode_record(&record.vector, &record.label).map_err(serde::de::Error::custom)?;
            self.writer
                .write_all(&encoded)
                .map_err(serde::de::Error::custom)?;
            count += 1;
        }
        Ok(count)
    }
}

struct ClusterLayout {
    offsets: Vec<u64>,
    bbox_min: Vec<QuantizedI16Vector>,
    bbox_max: Vec<QuantizedI16Vector>,
    records: Vec<QuantizedReference>,
}

fn read_quantized_references(reader: impl Read) -> Result<Vec<QuantizedReference>> {
    let mut deserializer = serde_json::Deserializer::from_reader(reader);
    CollectIndexSeed
        .deserialize(&mut deserializer)
        .map_err(Into::into)
}

struct CollectIndexSeed;

impl<'de> DeserializeSeed<'de> for CollectIndexSeed {
    type Value = Vec<QuantizedReference>;

    fn deserialize<D>(self, deserializer: D) -> std::result::Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_seq(CollectIndexVisitor)
    }
}

struct CollectIndexVisitor;

impl<'de> Visitor<'de> for CollectIndexVisitor {
    type Value = Vec<QuantizedReference>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an array of reference vectors")
    }

    fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let hint = seq.size_hint().unwrap_or(0);
        let mut records = Vec::with_capacity(hint);
        while let Some(record) = seq.next_element::<ReferenceRecord>()? {
            records.push(QuantizedReference {
                vector: quantize_i16(&record.vector),
                label: label_to_u8(&record.label).map_err(serde::de::Error::custom)?,
            });
        }
        Ok(records)
    }
}

fn train_centroids(
    records: &[QuantizedReference],
    cluster_count: usize,
    sample_size: usize,
    iterations: usize,
) -> Vec<Vector> {
    let sample = sample_indices(records.len(), sample_size);
    let mut centroids = initial_centroids(records, &sample, cluster_count);

    for _ in 0..iterations {
        let mut sums = vec![[0.0f64; DIMS]; cluster_count];
        let mut counts = vec![0u32; cluster_count];

        for &record_idx in &sample {
            let cluster_id = nearest_centroid(&records[record_idx].vector, &centroids);
            counts[cluster_id] += 1;
            for (dim, sum) in sums[cluster_id].iter_mut().enumerate() {
                *sum += quantized_to_f32(records[record_idx].vector[dim]) as f64;
            }
        }

        for cluster_id in 0..cluster_count {
            if counts[cluster_id] == 0 {
                continue;
            }

            let count = counts[cluster_id] as f64;
            for dim in 0..DIMS {
                centroids[cluster_id][dim] = (sums[cluster_id][dim] / count) as f32;
            }
        }
    }

    centroids
}

fn sample_indices(len: usize, sample_size: usize) -> Vec<usize> {
    if sample_size >= len {
        return (0..len).collect();
    }

    let mut indices = Vec::with_capacity(sample_size);
    for sample_idx in 0..sample_size {
        indices.push(sample_idx * len / sample_size);
    }
    indices
}

fn initial_centroids(
    records: &[QuantizedReference],
    sample: &[usize],
    cluster_count: usize,
) -> Vec<Vector> {
    let mut centroids = Vec::with_capacity(cluster_count);
    for cluster_id in 0..cluster_count {
        let sample_idx = sample[cluster_id * sample.len() / cluster_count];
        let mut centroid = [0.0; DIMS];
        for (dim, value) in records[sample_idx].vector.iter().enumerate() {
            centroid[dim] = quantized_to_f32(*value);
        }
        centroids.push(centroid);
    }
    centroids
}

fn assign_records(records: &[QuantizedReference], centroids: &[Vector]) -> Vec<usize> {
    records
        .iter()
        .map(|record| nearest_centroid(&record.vector, centroids))
        .collect()
}

fn nearest_centroid(vector: &QuantizedI16Vector, centroids: &[Vector]) -> usize {
    let mut best_id = 0usize;
    let mut best_dist = f32::INFINITY;

    for (cluster_id, centroid) in centroids.iter().enumerate() {
        let mut dist = 0.0;
        for dim in 0..DIMS {
            let delta = quantized_to_f32(vector[dim]) - centroid[dim];
            dist += delta * delta;
        }

        if dist < best_dist {
            best_dist = dist;
            best_id = cluster_id;
        }
    }

    best_id
}

fn build_cluster_layout(
    records: &[QuantizedReference],
    assignments: Vec<usize>,
    cluster_count: usize,
) -> ClusterLayout {
    let mut counts = vec![0usize; cluster_count];
    for &cluster_id in &assignments {
        counts[cluster_id] += 1;
    }

    let mut offsets = vec![0u64; cluster_count + 1];
    for cluster_id in 0..cluster_count {
        offsets[cluster_id + 1] = offsets[cluster_id] + counts[cluster_id] as u64;
    }

    let mut positions: Vec<_> = offsets[..cluster_count]
        .iter()
        .map(|offset| *offset as usize)
        .collect();
    let mut ordered_records = vec![
        QuantizedReference {
            vector: [0; DIMS],
            label: 0,
        };
        records.len()
    ];

    let mut bbox_min = vec![[i16::MAX; DIMS]; cluster_count];
    let mut bbox_max = vec![[i16::MIN; DIMS]; cluster_count];
    for (record, cluster_id) in records.iter().zip(assignments) {
        let position = positions[cluster_id];
        ordered_records[position] = *record;
        positions[cluster_id] += 1;

        for dim in 0..DIMS {
            bbox_min[cluster_id][dim] = bbox_min[cluster_id][dim].min(record.vector[dim]);
            bbox_max[cluster_id][dim] = bbox_max[cluster_id][dim].max(record.vector[dim]);
        }
    }

    for cluster_id in 0..cluster_count {
        if counts[cluster_id] == 0 {
            bbox_min[cluster_id] = [0; DIMS];
            bbox_max[cluster_id] = [0; DIMS];
        }
    }

    ClusterLayout {
        offsets,
        bbox_min,
        bbox_max,
        records: ordered_records,
    }
}

fn write_centroids(mut writer: impl Write, centroids: &[Vector]) -> Result<()> {
    for centroid in centroids {
        for value in centroid {
            writer.write_all(&value.to_le_bytes())?;
        }
    }
    Ok(())
}

fn write_offsets(mut writer: impl Write, offsets: &[u64]) -> Result<()> {
    for offset in offsets {
        writer.write_all(&offset.to_le_bytes())?;
    }
    Ok(())
}

fn write_bboxes(mut writer: impl Write, bboxes: &[QuantizedI16Vector]) -> Result<()> {
    for bbox in bboxes {
        for value in bbox {
            writer.write_all(&value.to_le_bytes())?;
        }
    }
    Ok(())
}

fn write_records(mut writer: impl Write, records: &[QuantizedReference]) -> Result<()> {
    for record in records {
        let mut encoded = [0u8; IVF_RECORD_LEN];
        for (dim, value) in record.vector.iter().enumerate() {
            let offset = dim * std::mem::size_of::<i16>();
            encoded[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
        }
        encoded[IVF_LABEL_OFFSET] = record.label;
        writer.write_all(&encoded)?;
    }
    Ok(())
}

#[inline]
fn quantized_to_f32(value: i16) -> f32 {
    if value < 0 {
        -1.0
    } else {
        value as f32 / 10_000.0
    }
}

fn label_to_u8(label: &str) -> Result<u8> {
    match label {
        "fraud" => Ok(1),
        "legit" => Ok(0),
        other => Err(anyhow!("unknown label {other}")),
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Read, Seek};

    use super::*;
    use crate::index::{Index, SearchResult};

    #[test]
    fn builds_ivf_index_from_json_array() {
        let json = br#"[
          {"vector":[0,0,0,0,0,-1,-1,0,0,0,1,0,0.15,0],"label":"legit"},
          {"vector":[1,1,1,1,1,-1,-1,1,1,1,0,1,0.85,1],"label":"fraud"}
        ]"#;
        let mut output = Cursor::new(Vec::new());

        let count = build_index_from_json_reader(&json[..], &mut output).unwrap();

        assert_eq!(count, 2);
        output.seek(SeekFrom::Start(16)).unwrap();
        let mut count_bytes = [0u8; 8];
        output.read_exact(&mut count_bytes).unwrap();
        assert_eq!(u64::from_le_bytes(count_bytes), 2);
    }

    #[test]
    fn generated_exact_index_can_be_opened_and_scored() {
        let json = br#"[
          {"vector":[0,0,0,0,0,0,0,0,0,0,0,0,0,0],"label":"legit"},
          {"vector":[0.02,0,0,0,0,0,0,0,0,0,0,0,0,0],"label":"fraud"},
          {"vector":[0.04,0,0,0,0,0,0,0,0,0,0,0,0,0],"label":"fraud"},
          {"vector":[0.06,0,0,0,0,0,0,0,0,0,0,0,0,0],"label":"fraud"},
          {"vector":[0.08,0,0,0,0,0,0,0,0,0,0,0,0,0],"label":"legit"},
          {"vector":[1,1,1,1,1,1,1,1,1,1,1,1,1,1],"label":"fraud"}
        ]"#;
        let path = temp_index_path();
        let file = std::fs::File::create(&path).unwrap();
        build_exact_index_from_json_reader(&json[..], file).unwrap();

        let index = Index::open(&path).unwrap();
        let score = match index.fraud_score(&[0.0; 14], None) {
            SearchResult::Score(score) => score,
            SearchResult::TimedOut => unreachable!("test runs without a deadline"),
        };

        std::fs::remove_file(path).unwrap();
        assert_eq!(index.len(), 6);
        assert_eq!(score, 0.6);
    }

    #[test]
    fn generated_ivf_index_can_be_opened_and_scored() {
        let json = br#"[
          {"vector":[0,0,0,0,0,0,0,0,0,0,0,0,0,0],"label":"legit"},
          {"vector":[1,1,1,1,1,1,1,1,1,1,1,1,1,1],"label":"fraud"}
        ]"#;
        let path = temp_index_path();
        let file = std::fs::File::create(&path).unwrap();
        build_index_from_json_reader(&json[..], file).unwrap();

        let index = Index::open(&path).unwrap();
        let score = match index.fraud_score(&[0.0; 14], None) {
            SearchResult::Score(score) => score,
            SearchResult::TimedOut => unreachable!("test runs without a deadline"),
        };

        std::fs::remove_file(path).unwrap();
        assert_eq!(index.len(), 2);
        assert!((0.0..=1.0).contains(&score));
    }

    fn temp_index_path() -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "rinha-2026-index-{}-{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        path
    }
}
