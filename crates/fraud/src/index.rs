use std::{
    fs::File,
    path::Path,
    time::{Duration, Instant},
};

use anyhow::{anyhow, bail, Context, Result};
use memmap2::{Mmap, MmapOptions};

use crate::vector::{quantize, QuantizedVector, Vector, DIMS};

const MAGIC: &[u8; 8] = b"RINHA26\0";
const VERSION: u32 = 1;
const HEADER_LEN: usize = 24;
const RECORD_LEN: usize = DIMS + 1;
const K: usize = 5;

pub struct Index {
    mmap: Mmap,
    count: usize,
}

impl Index {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("failed to open index {}", path.as_ref().display()))?;
        let mmap = unsafe { MmapOptions::new().map(&file) }.context("failed to mmap index")?;
        if mmap.len() < HEADER_LEN {
            bail!("index file too short");
        }
        if &mmap[0..8] != MAGIC {
            bail!("invalid index magic");
        }
        let version = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
        let dims = u32::from_le_bytes(mmap[12..16].try_into().unwrap()) as usize;
        let count = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
        if version != VERSION {
            bail!("unsupported index version {version}");
        }
        if dims != DIMS {
            bail!("unsupported dimension count {dims}");
        }
        let expected_len = HEADER_LEN + count * RECORD_LEN;
        if mmap.len() != expected_len {
            bail!(
                "invalid index length: got {}, expected {expected_len}",
                mmap.len()
            );
        }
        Ok(Self { mmap, count })
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn fraud_score(&self, vector: &Vector, deadline: Option<Duration>) -> Option<f32> {
        if self.count == 0 {
            return Some(0.0);
        }

        let query = quantize(vector);
        let started_at = Instant::now();
        let mut best_dist = [u32::MAX; K];
        let mut best_label = [0u8; K];
        let records = &self.mmap[HEADER_LEN..];

        for idx in 0..self.count {
            if idx & 0x3fff == 0 {
                if let Some(max_duration) = deadline {
                    if started_at.elapsed() > max_duration {
                        return None;
                    }
                }
            }

            let offset = idx * RECORD_LEN;
            let record = &records[offset..offset + RECORD_LEN];
            let dist = squared_distance(&query, record[..DIMS].try_into().ok()?);
            insert_best(dist, record[DIMS], &mut best_dist, &mut best_label);
        }

        let neighbors = self.count.min(K);
        let frauds = best_label[..neighbors]
            .iter()
            .filter(|label| **label == 1)
            .count();
        Some(frauds as f32 / K as f32)
    }
}

#[inline]
fn squared_distance(query: &QuantizedVector, candidate: &QuantizedVector) -> u32 {
    let mut total = 0u32;
    for idx in 0..DIMS {
        let delta = query[idx] as i32 - candidate[idx] as i32;
        total += (delta * delta) as u32;
    }
    total
}

#[inline]
fn insert_best(dist: u32, label: u8, best_dist: &mut [u32; K], best_label: &mut [u8; K]) {
    if dist >= best_dist[K - 1] {
        return;
    }

    let mut pos = K - 1;
    while pos > 0 && dist < best_dist[pos - 1] {
        best_dist[pos] = best_dist[pos - 1];
        best_label[pos] = best_label[pos - 1];
        pos -= 1;
    }
    best_dist[pos] = dist;
    best_label[pos] = label;
}

pub fn write_header(mut writer: impl std::io::Write, count: u64) -> Result<()> {
    writer.write_all(MAGIC)?;
    writer.write_all(&VERSION.to_le_bytes())?;
    writer.write_all(&(DIMS as u32).to_le_bytes())?;
    writer.write_all(&count.to_le_bytes())?;
    Ok(())
}

pub fn encode_record(vector: &Vector, label: &str) -> Result<[u8; RECORD_LEN]> {
    let quantized = quantize(vector);
    let mut record = [0u8; RECORD_LEN];
    record[..DIMS].copy_from_slice(&quantized);
    record[DIMS] = match label {
        "fraud" => 1,
        "legit" => 0,
        other => return Err(anyhow!("unknown label {other}")),
    };
    Ok(record)
}
