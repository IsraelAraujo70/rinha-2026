#[cfg(not(target_arch = "x86_64"))]
compile_error!("this implementation targets linux/amd64 only");

use std::{
    arch::x86_64::{
        __m128i, __m256i, _mm256_add_epi32, _mm256_and_si256, _mm256_broadcastsi128_si256,
        _mm256_castsi256_si128, _mm256_extracti128_si256, _mm256_loadu_si256, _mm256_madd_epi16,
        _mm256_set_epi16, _mm256_set_epi8, _mm256_setzero_si256, _mm256_storeu_si256,
        _mm256_sub_epi16, _mm256_unpackhi_epi8, _mm256_unpacklo_epi8, _mm_add_epi32, _mm_and_si128,
        _mm_cvtsi128_si32, _mm_loadu_si128, _mm_madd_epi16, _mm_set_epi8, _mm_setzero_si128,
        _mm_srli_si128, _mm_sub_epi16, _mm_unpackhi_epi8, _mm_unpacklo_epi8,
    },
    env,
    fs::File,
    mem::size_of,
    path::Path,
    time::{Duration, Instant},
};

use anyhow::{anyhow, bail, Context, Result};
use memmap2::{Mmap, MmapOptions};

use crate::vector::{quantize, quantize_i16, Vector, DIMS};

const MAGIC: &[u8; 8] = b"RINHA26\0";
const EXACT_VERSION: u32 = 2;
const IVF_VERSION: u32 = 3;
const EXACT_HEADER_LEN: usize = 24;
const EXACT_RECORD_LEN: usize = 16;
const EXACT_LABEL_OFFSET: usize = DIMS;
pub const IVF_HEADER_LEN: usize = 32;
pub const IVF_RECORD_LEN: usize = 32;
pub const IVF_LABEL_OFFSET: usize = DIMS * 2;
const MAX_IVF_CLUSTERS: usize = 4096;
const MAX_IVF_NPROBE: usize = 64;
const K: usize = 5;

pub struct Index {
    inner: IndexInner,
}

enum IndexInner {
    Exact(ExactIndex),
    Ivf(IvfIndex),
}

struct ExactIndex {
    mmap: Mmap,
    count: usize,
}

struct IvfIndex {
    mmap: Mmap,
    count: usize,
    cluster_count: usize,
    nprobe: usize,
    full_nprobe: usize,
    repair: bool,
    centroids_offset: usize,
    offsets_offset: usize,
    bbox_min_offset: usize,
    bbox_max_offset: usize,
    records_offset: usize,
}

#[derive(Clone, Copy)]
struct Avx2Distance {
    query: __m256i,
    mask: __m256i,
}

pub enum SearchResult {
    Score(f32),
    TimedOut,
}

impl Index {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("failed to open index {}", path.as_ref().display()))?;
        let mmap = unsafe { MmapOptions::new().map(&file) }.context("failed to mmap index")?;
        if mmap.len() < EXACT_HEADER_LEN {
            bail!("index file too short");
        }
        if &mmap[0..8] != MAGIC {
            bail!("invalid index magic");
        }
        let version = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
        let dims = u32::from_le_bytes(mmap[12..16].try_into().unwrap()) as usize;
        if dims != DIMS {
            bail!("unsupported dimension count {dims}");
        }

        prefault_pages(&mmap);

        let index = match version {
            EXACT_VERSION => Self::open_exact(mmap)?,
            IVF_VERSION => Self::open_ivf(mmap)?,
            other => bail!("unsupported index version {other}"),
        };
        index.warmup();
        Ok(index)
    }

    fn warmup(&self) {
        let mut state: u32 = 0x12345678;
        let mut sink: f32 = 0.0;
        for _ in 0..512 {
            let mut query = [0.0f32; DIMS];
            for value in query.iter_mut() {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                *value = (state >> 8) as f32 / (1u32 << 24) as f32;
            }
            if let SearchResult::Score(score) = self.fraud_score(&query, None) {
                sink += score;
            }
        }
        std::hint::black_box(sink);
    }

    pub fn len(&self) -> usize {
        match &self.inner {
            IndexInner::Exact(index) => index.count,
            IndexInner::Ivf(index) => index.count,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn fraud_score(&self, vector: &Vector, deadline: Option<Duration>) -> SearchResult {
        match &self.inner {
            IndexInner::Exact(index) => index.fraud_score(vector, deadline),
            IndexInner::Ivf(index) => index.fraud_score(vector, deadline),
        }
    }

    fn open_exact(mmap: Mmap) -> Result<Self> {
        let count = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
        let expected_len = EXACT_HEADER_LEN + count * EXACT_RECORD_LEN;
        if mmap.len() != expected_len {
            bail!(
                "invalid exact index length: got {}, expected {expected_len}",
                mmap.len()
            );
        }
        Ok(Self {
            inner: IndexInner::Exact(ExactIndex { mmap, count }),
        })
    }

    fn open_ivf(mmap: Mmap) -> Result<Self> {
        if mmap.len() < IVF_HEADER_LEN {
            bail!("ivf index file too short");
        }

        let count = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
        let cluster_count = u32::from_le_bytes(mmap[24..28].try_into().unwrap()) as usize;
        if cluster_count == 0 || cluster_count > MAX_IVF_CLUSTERS {
            bail!("unsupported ivf cluster count {cluster_count}");
        }

        let centroids_len = cluster_count * DIMS * size_of::<f32>();
        let offsets_len = (cluster_count + 1) * size_of::<u64>();
        let bbox_len = cluster_count * DIMS * size_of::<i16>();
        let centroids_offset = IVF_HEADER_LEN;
        let offsets_offset = centroids_offset + centroids_len;
        let bbox_min_offset = offsets_offset + offsets_len;
        let bbox_max_offset = bbox_min_offset + bbox_len;
        let records_offset = bbox_max_offset + bbox_len;
        let expected_len = records_offset + count * IVF_RECORD_LEN;
        if mmap.len() != expected_len {
            bail!(
                "invalid ivf index length: got {}, expected {expected_len}",
                mmap.len()
            );
        }

        let nprobe = env_usize("IVF_NPROBE", 1).clamp(1, cluster_count.min(MAX_IVF_NPROBE));
        let full_nprobe =
            env_usize("IVF_FULL_NPROBE", nprobe).clamp(nprobe, cluster_count.min(MAX_IVF_NPROBE));
        let repair = env_bool("IVF_REPAIR", false);
        Ok(Self {
            inner: IndexInner::Ivf(IvfIndex {
                mmap,
                count,
                cluster_count,
                nprobe,
                full_nprobe,
                repair,
                centroids_offset,
                offsets_offset,
                bbox_min_offset,
                bbox_max_offset,
                records_offset,
            }),
        })
    }
}

impl ExactIndex {
    fn fraud_score(&self, vector: &Vector, deadline: Option<Duration>) -> SearchResult {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { self.fraud_score_avx2(vector, deadline) };
        }

        self.fraud_score_one_at_a_time(vector, deadline)
    }

    fn fraud_score_one_at_a_time(
        &self,
        vector: &Vector,
        deadline: Option<Duration>,
    ) -> SearchResult {
        if self.count == 0 {
            return SearchResult::Score(0.0);
        }

        let query = padded_quantize(vector);
        let started_at = Instant::now();
        let mut best_dist = [u32::MAX; K];
        let mut best_label = [0u8; K];
        let records = &self.mmap[EXACT_HEADER_LEN..];

        for idx in 0..self.count {
            if idx & 0x3fff == 0 {
                if let Some(max_duration) = deadline {
                    if started_at.elapsed() > max_duration {
                        return SearchResult::TimedOut;
                    }
                }
            }

            let offset = idx * EXACT_RECORD_LEN;
            let record = &records[offset..offset + EXACT_RECORD_LEN];
            let dist = squared_distance(&query, record);
            insert_best(
                dist,
                record[EXACT_LABEL_OFFSET],
                &mut best_dist,
                &mut best_label,
            );
        }

        SearchResult::Score(score_from_labels(self.count, &best_label))
    }

    #[target_feature(enable = "avx2")]
    unsafe fn fraud_score_avx2(&self, vector: &Vector, deadline: Option<Duration>) -> SearchResult {
        if self.count == 0 {
            return SearchResult::Score(0.0);
        }

        let query = padded_quantize(vector);
        let started_at = Instant::now();
        let mut best_dist = [u32::MAX; K];
        let mut best_label = [0u8; K];
        let records = &self.mmap[EXACT_HEADER_LEN..];
        let avx2_query = broadcast_query_avx2(&query);
        let avx2_mask = distance_mask_avx2();
        let avx2_zero = _mm256_setzero_si256();

        let mut idx = 0usize;
        while idx + 3 < self.count {
            if idx & 0x3fff == 0 {
                if let Some(max_duration) = deadline {
                    if started_at.elapsed() > max_duration {
                        return SearchResult::TimedOut;
                    }
                }
            }

            let offset = idx * EXACT_RECORD_LEN;
            let first_pair = &records[offset..offset + EXACT_RECORD_LEN * 2];
            let second_pair =
                &records[offset + EXACT_RECORD_LEN * 2..offset + EXACT_RECORD_LEN * 4];
            let [first_dist, second_dist] =
                squared_distance_pair_avx2(avx2_query, avx2_mask, avx2_zero, first_pair);
            let [third_dist, fourth_dist] =
                squared_distance_pair_avx2(avx2_query, avx2_mask, avx2_zero, second_pair);
            let first_label = first_pair[EXACT_LABEL_OFFSET];
            let second_label = first_pair[EXACT_RECORD_LEN + EXACT_LABEL_OFFSET];
            let third_label = second_pair[EXACT_LABEL_OFFSET];
            let fourth_label = second_pair[EXACT_RECORD_LEN + EXACT_LABEL_OFFSET];
            insert_best(first_dist, first_label, &mut best_dist, &mut best_label);
            insert_best(second_dist, second_label, &mut best_dist, &mut best_label);
            insert_best(third_dist, third_label, &mut best_dist, &mut best_label);
            insert_best(fourth_dist, fourth_label, &mut best_dist, &mut best_label);
            idx += 4;
        }

        while idx + 1 < self.count {
            let offset = idx * EXACT_RECORD_LEN;
            let record_pair = &records[offset..offset + EXACT_RECORD_LEN * 2];
            let [first_dist, second_dist] =
                squared_distance_pair_avx2(avx2_query, avx2_mask, avx2_zero, record_pair);
            insert_best(
                first_dist,
                record_pair[EXACT_LABEL_OFFSET],
                &mut best_dist,
                &mut best_label,
            );
            insert_best(
                second_dist,
                record_pair[EXACT_RECORD_LEN + EXACT_LABEL_OFFSET],
                &mut best_dist,
                &mut best_label,
            );
            idx += 2;
        }

        if idx < self.count {
            let offset = idx * EXACT_RECORD_LEN;
            let record = &records[offset..offset + EXACT_RECORD_LEN];
            let dist = squared_distance(&query, record);
            insert_best(
                dist,
                record[EXACT_LABEL_OFFSET],
                &mut best_dist,
                &mut best_label,
            );
        }

        SearchResult::Score(score_from_labels(self.count, &best_label))
    }
}

impl IvfIndex {
    fn fraud_score(&self, vector: &Vector, deadline: Option<Duration>) -> SearchResult {
        if self.count == 0 {
            return SearchResult::Score(0.0);
        }

        let query = quantize_i16(vector);
        let distance = Avx2Distance {
            query: unsafe { load_query_i16_avx2(&query) },
            mask: unsafe { distance_mask_i16_avx2() },
        };
        let started_at = Instant::now();
        let mut best_dist = [u64::MAX; K];
        let mut best_label = [0u8; K];
        let mut cluster_dist = [f32::INFINITY; MAX_IVF_NPROBE];
        let mut cluster_ids = [0usize; MAX_IVF_NPROBE];

        let max_probes = self.full_nprobe;
        for cluster_id in 0..self.cluster_count {
            let dist = self.centroid_distance(vector, cluster_id);
            insert_best_cluster(
                dist,
                cluster_id,
                max_probes,
                &mut cluster_dist,
                &mut cluster_ids,
            );
        }

        let mut visited = [false; MAX_IVF_CLUSTERS];
        for &cluster_id in &cluster_ids[..self.nprobe] {
            visited[cluster_id] = true;
            if self.scan_cluster(
                cluster_id,
                distance,
                &mut best_dist,
                &mut best_label,
                started_at,
                deadline,
            ) {
                return SearchResult::TimedOut;
            }
        }

        if max_probes > self.nprobe {
            let frauds = fraud_count(&best_label);
            if frauds == 2 || frauds == 3 {
                for &cluster_id in &cluster_ids[self.nprobe..max_probes] {
                    visited[cluster_id] = true;
                    if self.scan_cluster(
                        cluster_id,
                        distance,
                        &mut best_dist,
                        &mut best_label,
                        started_at,
                        deadline,
                    ) {
                        return SearchResult::Score(score_from_labels(self.count, &best_label));
                    }
                }
            }
        }

        if self.repair {
            for (cluster_id, was_visited) in visited.iter().enumerate().take(self.cluster_count) {
                if *was_visited || self.bbox_lower_bound(&query, cluster_id) > best_dist[K - 1] {
                    continue;
                }

                if self.scan_cluster(
                    cluster_id,
                    distance,
                    &mut best_dist,
                    &mut best_label,
                    started_at,
                    deadline,
                ) {
                    return SearchResult::TimedOut;
                }
            }
        }

        SearchResult::Score(score_from_labels(self.count, &best_label))
    }

    #[inline]
    fn centroid_distance(&self, vector: &Vector, cluster_id: usize) -> f32 {
        let base = self.centroids_offset + cluster_id * DIMS * size_of::<f32>();
        let mut total = 0.0;
        for (dim, value) in vector.iter().enumerate() {
            let offset = base + dim * size_of::<f32>();
            let centroid = f32::from_le_bytes(self.mmap[offset..offset + 4].try_into().unwrap());
            let delta = *value - centroid;
            total += delta * delta;
        }
        total
    }

    fn scan_cluster(
        &self,
        cluster_id: usize,
        distance: Avx2Distance,
        best_dist: &mut [u64; K],
        best_label: &mut [u8; K],
        started_at: Instant,
        deadline: Option<Duration>,
    ) -> bool {
        let start = self.cluster_offset(cluster_id);
        let end = self.cluster_offset(cluster_id + 1);
        for record_idx in start..end {
            if record_idx & 0x3fff == 0 {
                if let Some(max_duration) = deadline {
                    if started_at.elapsed() > max_duration {
                        return true;
                    }
                }
            }

            let record_offset = self.records_offset + record_idx * IVF_RECORD_LEN;
            let record = &self.mmap[record_offset..record_offset + IVF_RECORD_LEN];
            let dist = unsafe { squared_distance_i16_avx2(distance.query, distance.mask, record) };
            insert_best_u64(dist, record[IVF_LABEL_OFFSET], best_dist, best_label);
        }
        false
    }

    #[inline]
    fn cluster_offset(&self, cluster_id: usize) -> usize {
        let offset = self.offsets_offset + cluster_id * size_of::<u64>();
        u64::from_le_bytes(self.mmap[offset..offset + 8].try_into().unwrap()) as usize
    }

    #[inline]
    fn bbox_lower_bound(&self, query: &[i16; DIMS], cluster_id: usize) -> u64 {
        let min_base = self.bbox_min_offset + cluster_id * DIMS * size_of::<i16>();
        let max_base = self.bbox_max_offset + cluster_id * DIMS * size_of::<i16>();
        let mut total = 0u64;
        for (dim, value) in query.iter().enumerate() {
            let min_offset = min_base + dim * size_of::<i16>();
            let max_offset = max_base + dim * size_of::<i16>();
            let min = i16::from_le_bytes(self.mmap[min_offset..min_offset + 2].try_into().unwrap());
            let max = i16::from_le_bytes(self.mmap[max_offset..max_offset + 2].try_into().unwrap());
            let delta = if *value < min {
                min as i32 - *value as i32
            } else if *value > max {
                *value as i32 - max as i32
            } else {
                0
            };
            total += (delta * delta) as u64;
        }
        total
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
fn fraud_count(best_label: &[u8; K]) -> usize {
    best_label.iter().filter(|label| **label == 1).count()
}

#[inline]
fn padded_quantize(vector: &Vector) -> [u8; EXACT_RECORD_LEN] {
    let quantized = quantize(vector);
    let mut padded = [0u8; EXACT_RECORD_LEN];
    padded[..DIMS].copy_from_slice(&quantized);
    padded
}

#[inline]
fn squared_distance(query: &[u8; EXACT_RECORD_LEN], candidate: &[u8]) -> u32 {
    unsafe { squared_distance_sse2(query, candidate) }
}

#[target_feature(enable = "sse2")]
unsafe fn squared_distance_sse2(query: &[u8; EXACT_RECORD_LEN], candidate: &[u8]) -> u32 {
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

#[target_feature(enable = "avx2")]
unsafe fn squared_distance_pair_avx2(
    query_bytes: __m256i,
    mask: __m256i,
    zero: __m256i,
    candidates: &[u8],
) -> [u32; 2] {
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

#[target_feature(enable = "sse2")]
unsafe fn horizontal_sum_i32x4(value: __m128i) -> u32 {
    let sum = _mm_add_epi32(value, _mm_srli_si128::<8>(value));
    let sum = _mm_add_epi32(sum, _mm_srli_si128::<4>(sum));
    _mm_cvtsi128_si32(sum) as u32
}

#[target_feature(enable = "avx2")]
unsafe fn broadcast_query_avx2(query: &[u8; EXACT_RECORD_LEN]) -> __m256i {
    _mm256_broadcastsi128_si256(_mm_loadu_si128(query.as_ptr().cast::<__m128i>()))
}

#[target_feature(enable = "avx2")]
unsafe fn distance_mask_avx2() -> __m256i {
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

#[inline(always)]
fn insert_best_u64(dist: u64, label: u8, best_dist: &mut [u64; K], best_label: &mut [u8; K]) {
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

#[inline(always)]
fn insert_best_cluster(
    dist: f32,
    cluster_id: usize,
    limit: usize,
    best_dist: &mut [f32; MAX_IVF_NPROBE],
    best_cluster: &mut [usize; MAX_IVF_NPROBE],
) {
    if dist >= best_dist[limit - 1] {
        return;
    }

    let mut pos = limit - 1;
    while pos > 0 && dist < best_dist[pos - 1] {
        best_dist[pos] = best_dist[pos - 1];
        best_cluster[pos] = best_cluster[pos - 1];
        pos -= 1;
    }
    best_dist[pos] = dist;
    best_cluster[pos] = cluster_id;
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_query_i16_avx2(query: &[i16; DIMS]) -> __m256i {
    let mut padded = [0i16; 16];
    padded[..DIMS].copy_from_slice(query);
    _mm256_loadu_si256(padded.as_ptr().cast::<__m256i>())
}

#[target_feature(enable = "avx2")]
fn distance_mask_i16_avx2() -> __m256i {
    _mm256_set_epi16(0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)
}

#[target_feature(enable = "avx2")]
unsafe fn squared_distance_i16_avx2(query: __m256i, mask: __m256i, candidate: &[u8]) -> u64 {
    let candidate = _mm256_and_si256(
        _mm256_loadu_si256(candidate.as_ptr().cast::<__m256i>()),
        mask,
    );
    let diff = _mm256_sub_epi16(query, candidate);
    let sums = _mm256_madd_epi16(diff, diff);
    let mut lanes = [0i32; 8];
    _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), sums);
    lanes.iter().map(|lane| *lane as u32 as u64).sum()
}

fn env_usize(name: &str, default: usize) -> usize {
    env::var(name)
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_bool(name: &str, default: bool) -> bool {
    env::var(name)
        .ok()
        .and_then(|raw| match raw.as_str() {
            "1" | "true" | "TRUE" | "yes" | "YES" => Some(true),
            "0" | "false" | "FALSE" | "no" | "NO" => Some(false),
            _ => None,
        })
        .unwrap_or(default)
}

pub fn write_header(mut writer: impl std::io::Write, count: u64) -> Result<()> {
    writer.write_all(MAGIC)?;
    writer.write_all(&EXACT_VERSION.to_le_bytes())?;
    writer.write_all(&(DIMS as u32).to_le_bytes())?;
    writer.write_all(&count.to_le_bytes())?;
    Ok(())
}

pub fn write_ivf_header(
    mut writer: impl std::io::Write,
    count: u64,
    cluster_count: u32,
) -> Result<()> {
    writer.write_all(MAGIC)?;
    writer.write_all(&IVF_VERSION.to_le_bytes())?;
    writer.write_all(&(DIMS as u32).to_le_bytes())?;
    writer.write_all(&count.to_le_bytes())?;
    writer.write_all(&cluster_count.to_le_bytes())?;
    writer.write_all(&0u32.to_le_bytes())?;
    Ok(())
}

pub fn encode_record(vector: &Vector, label: &str) -> Result<[u8; EXACT_RECORD_LEN]> {
    let quantized = quantize(vector);
    let mut record = [0u8; EXACT_RECORD_LEN];
    record[..DIMS].copy_from_slice(&quantized);
    record[EXACT_LABEL_OFFSET] = match label {
        "fraud" => 1,
        "legit" => 0,
        other => return Err(anyhow!("unknown label {other}")),
    };
    Ok(record)
}

fn prefault_pages(mmap: &Mmap) {
    let bytes = &mmap[..];
    let page = 4096usize;
    let mut acc: u8 = 0;
    let mut offset = 0usize;
    while offset < bytes.len() {
        acc ^= bytes[offset];
        offset += page;
    }
    if !bytes.is_empty() {
        acc ^= bytes[bytes.len() - 1];
    }
    std::hint::black_box(acc);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avx2_i16_distance_ignores_label_lane() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }

        let query = [10_000, 0, -10_000, 500, 900, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut candidate = [0u8; IVF_RECORD_LEN];
        let values: [i16; DIMS] = [9_000, 1, -9_000, 400, 800, 9, 8, 7, 6, 5, 4, 3, 2, 1];
        for (dim, value) in values.iter().enumerate() {
            let offset = dim * size_of::<i16>();
            candidate[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
        }
        candidate[IVF_LABEL_OFFSET] = 255;

        let distance = unsafe {
            squared_distance_i16_avx2(
                load_query_i16_avx2(&query),
                distance_mask_i16_avx2(),
                &candidate,
            )
        };
        let expected = query
            .iter()
            .zip(values)
            .map(|(a, b)| {
                let delta = *a as i32 - b as i32;
                (delta * delta) as u64
            })
            .sum::<u64>();

        assert_eq!(distance, expected);
    }
}
