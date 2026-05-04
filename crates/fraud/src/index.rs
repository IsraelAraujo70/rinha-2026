use std::{
    fs::File,
    path::Path,
    time::{Duration, Instant},
};

use anyhow::{anyhow, bail, Context, Result};
use memmap2::{Mmap, MmapOptions};

use crate::vector::{quantize, Vector, DIMS};

const MAGIC: &[u8; 8] = b"RINHA26\0";
const VERSION: u32 = 2;
const HEADER_LEN: usize = 24;
const RECORD_LEN: usize = 16;
const LABEL_OFFSET: usize = DIMS;
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
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                return unsafe { self.fraud_score_avx2(vector, deadline) };
            }
        }

        self.fraud_score_one_at_a_time(vector, deadline)
    }

    fn fraud_score_one_at_a_time(
        &self,
        vector: &Vector,
        deadline: Option<Duration>,
    ) -> Option<f32> {
        if self.count == 0 {
            return Some(0.0);
        }

        let query = padded_quantize(vector);
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
            let dist = squared_distance(&query, record);
            insert_best(dist, record[LABEL_OFFSET], &mut best_dist, &mut best_label);
        }

        Some(score_from_labels(self.count, &best_label))
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn fraud_score_avx2(&self, vector: &Vector, deadline: Option<Duration>) -> Option<f32> {
        if self.count == 0 {
            return Some(0.0);
        }

        let query = padded_quantize(vector);
        let started_at = Instant::now();
        let mut best_dist = [u32::MAX; K];
        let mut best_label = [0u8; K];
        let records = &self.mmap[HEADER_LEN..];
        let avx2_query = broadcast_query_avx2(&query);
        let avx2_mask = distance_mask_avx2();
        let avx2_zero = std::arch::x86_64::_mm256_setzero_si256();

        let mut idx = 0usize;
        while idx + 3 < self.count {
            if idx & 0x3fff == 0 {
                if let Some(max_duration) = deadline {
                    if started_at.elapsed() > max_duration {
                        return None;
                    }
                }
            }

            let offset = idx * RECORD_LEN;
            let first_pair = &records[offset..offset + RECORD_LEN * 2];
            let second_pair = &records[offset + RECORD_LEN * 2..offset + RECORD_LEN * 4];
            let [first_dist, second_dist] =
                squared_distance_pair_avx2(avx2_query, avx2_mask, avx2_zero, first_pair);
            let [third_dist, fourth_dist] =
                squared_distance_pair_avx2(avx2_query, avx2_mask, avx2_zero, second_pair);
            let first_label = first_pair[LABEL_OFFSET];
            let second_label = first_pair[RECORD_LEN + LABEL_OFFSET];
            let third_label = second_pair[LABEL_OFFSET];
            let fourth_label = second_pair[RECORD_LEN + LABEL_OFFSET];
            insert_best(first_dist, first_label, &mut best_dist, &mut best_label);
            insert_best(second_dist, second_label, &mut best_dist, &mut best_label);
            insert_best(third_dist, third_label, &mut best_dist, &mut best_label);
            insert_best(fourth_dist, fourth_label, &mut best_dist, &mut best_label);
            idx += 4;
        }

        while idx + 1 < self.count {
            let offset = idx * RECORD_LEN;
            let record_pair = &records[offset..offset + RECORD_LEN * 2];
            let [first_dist, second_dist] =
                squared_distance_pair_avx2(avx2_query, avx2_mask, avx2_zero, record_pair);
            insert_best(
                first_dist,
                record_pair[LABEL_OFFSET],
                &mut best_dist,
                &mut best_label,
            );
            insert_best(
                second_dist,
                record_pair[RECORD_LEN + LABEL_OFFSET],
                &mut best_dist,
                &mut best_label,
            );
            idx += 2;
        }

        if idx < self.count {
            let offset = idx * RECORD_LEN;
            let record = &records[offset..offset + RECORD_LEN];
            let dist = squared_distance(&query, record);
            insert_best(dist, record[LABEL_OFFSET], &mut best_dist, &mut best_label);
        }

        Some(score_from_labels(self.count, &best_label))
    }
}

fn score_from_labels(count: usize, best_label: &[u8; K]) -> f32 {
    let neighbors = count.min(K);
    let frauds = best_label[..neighbors]
        .iter()
        .filter(|label| **label == 1)
        .count();
    frauds as f32 / K as f32
}

#[inline]
fn padded_quantize(vector: &Vector) -> [u8; RECORD_LEN] {
    let quantized = quantize(vector);
    let mut padded = [0u8; RECORD_LEN];
    padded[..DIMS].copy_from_slice(&quantized);
    padded
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn squared_distance(query: &[u8; RECORD_LEN], candidate: &[u8]) -> u32 {
    unsafe { squared_distance_sse2(query, candidate) }
}

#[cfg(not(target_arch = "x86_64"))]
#[inline]
fn squared_distance(query: &[u8; RECORD_LEN], candidate: &[u8]) -> u32 {
    squared_distance_portable(query, candidate)
}

#[inline]
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
fn squared_distance_portable(query: &[u8; RECORD_LEN], candidate: &[u8]) -> u32 {
    let mut total = 0u32;
    for idx in 0..DIMS {
        let delta = query[idx] as i32 - candidate[idx] as i32;
        total += (delta * delta) as u32;
    }
    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn squared_distance_sse2(query: &[u8; RECORD_LEN], candidate: &[u8]) -> u32 {
    use std::arch::x86_64::{
        __m128i, _mm_add_epi32, _mm_and_si128, _mm_loadu_si128, _mm_madd_epi16, _mm_set_epi8,
        _mm_setzero_si128, _mm_sub_epi16, _mm_unpackhi_epi8, _mm_unpacklo_epi8,
    };

    let mask = _mm_set_epi8(0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let zero = _mm_setzero_si128();
    let query_bytes = _mm_loadu_si128(query.as_ptr().cast::<__m128i>());
    let candidate_bytes =
        _mm_and_si128(_mm_loadu_si128(candidate.as_ptr().cast::<__m128i>()), mask);

    let q_lo = _mm_unpacklo_epi8(query_bytes, zero);
    let q_hi = _mm_unpackhi_epi8(query_bytes, zero);
    let c_lo = _mm_unpacklo_epi8(candidate_bytes, zero);
    let c_hi = _mm_unpackhi_epi8(candidate_bytes, zero);

    let d_lo = _mm_sub_epi16(q_lo, c_lo);
    let d_hi = _mm_sub_epi16(q_hi, c_hi);
    let sums = _mm_add_epi32(_mm_madd_epi16(d_lo, d_lo), _mm_madd_epi16(d_hi, d_hi));

    horizontal_sum_i32x4(sums)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn squared_distance_pair_avx2(
    query_bytes: std::arch::x86_64::__m256i,
    mask: std::arch::x86_64::__m256i,
    zero: std::arch::x86_64::__m256i,
    candidates: &[u8],
) -> [u32; 2] {
    use std::arch::x86_64::{
        __m256i, _mm256_add_epi32, _mm256_and_si256, _mm256_castsi256_si128,
        _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_madd_epi16, _mm256_sub_epi16,
        _mm256_unpackhi_epi8, _mm256_unpacklo_epi8,
    };

    let candidate_bytes = _mm256_and_si256(
        _mm256_loadu_si256(candidates.as_ptr().cast::<__m256i>()),
        mask,
    );

    let q_lo = _mm256_unpacklo_epi8(query_bytes, zero);
    let q_hi = _mm256_unpackhi_epi8(query_bytes, zero);
    let c_lo = _mm256_unpacklo_epi8(candidate_bytes, zero);
    let c_hi = _mm256_unpackhi_epi8(candidate_bytes, zero);

    let d_lo = _mm256_sub_epi16(q_lo, c_lo);
    let d_hi = _mm256_sub_epi16(q_hi, c_hi);
    let sums = _mm256_add_epi32(_mm256_madd_epi16(d_lo, d_lo), _mm256_madd_epi16(d_hi, d_hi));

    [
        horizontal_sum_i32x4(_mm256_castsi256_si128(sums)),
        horizontal_sum_i32x4(_mm256_extracti128_si256::<1>(sums)),
    ]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn horizontal_sum_i32x4(value: std::arch::x86_64::__m128i) -> u32 {
    use std::arch::x86_64::{_mm_add_epi32, _mm_cvtsi128_si32, _mm_srli_si128};

    let sum = _mm_add_epi32(value, _mm_srli_si128::<8>(value));
    let sum = _mm_add_epi32(sum, _mm_srli_si128::<4>(sum));
    _mm_cvtsi128_si32(sum) as u32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn broadcast_query_avx2(query: &[u8; RECORD_LEN]) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::{__m128i, _mm256_broadcastsi128_si256, _mm_loadu_si128};

    _mm256_broadcastsi128_si256(_mm_loadu_si128(query.as_ptr().cast::<__m128i>()))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn distance_mask_avx2() -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::_mm256_set_epi8;

    _mm256_set_epi8(
        0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1,
    )
}

#[inline(always)]
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
    record[LABEL_OFFSET] = match label {
        "fraud" => 1,
        "legit" => 0,
        other => return Err(anyhow!("unknown label {other}")),
    };
    Ok(record)
}
