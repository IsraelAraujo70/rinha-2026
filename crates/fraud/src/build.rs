use std::{
    fmt,
    io::{Read, Seek, SeekFrom, Write},
};

use anyhow::Result;
use serde::{
    de::{DeserializeSeed, SeqAccess, Visitor},
    Deserialize, Deserializer,
};

use crate::{
    index::{encode_record, write_header},
    vector::Vector,
};

#[derive(Deserialize)]
struct ReferenceRecord {
    vector: Vector,
    label: String,
}

pub fn build_index_from_json_reader(
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

#[cfg(test)]
mod tests {
    use std::io::{Cursor, Read, Seek};

    use super::*;
    use crate::index::{Index, SearchResult};

    #[test]
    fn builds_index_from_json_array_without_buffering_records() {
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
    fn generated_index_can_be_opened_and_scored() {
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
        build_index_from_json_reader(&json[..], file).unwrap();

        let index = Index::open(&path).unwrap();
        let score = match index.fraud_score(&[0.0; 14], None) {
            SearchResult::Score(score) => score,
            SearchResult::TimedOut => unreachable!("test runs without a deadline"),
        };

        std::fs::remove_file(path).unwrap();
        assert_eq!(index.len(), 6);
        assert_eq!(score, 0.6);
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
