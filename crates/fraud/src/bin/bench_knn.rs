use std::{env, fs::File, path::PathBuf, time::Instant};

use anyhow::{Context, Result};
use fraud::{index::Index, payload::FraudRequest, vector::vectorize};

fn main() -> Result<()> {
    let mut args = env::args_os().skip(1);
    let index_path = args
        .next()
        .map(PathBuf::from)
        .context("usage: bench_knn <data.bin> <payloads.json> [iterations]")?;
    let payloads_path = args
        .next()
        .map(PathBuf::from)
        .context("usage: bench_knn <data.bin> <payloads.json> [iterations]")?;
    let iterations = args
        .next()
        .and_then(|raw| raw.to_string_lossy().parse::<usize>().ok())
        .unwrap_or(1_000);

    let index = Index::open(&index_path)?;
    let payloads_file =
        File::open(&payloads_path).with_context(|| format!("open {}", payloads_path.display()))?;
    let payloads: Vec<FraudRequest> = serde_json::from_reader(payloads_file)
        .with_context(|| format!("parse {}", payloads_path.display()))?;
    let vectors: Vec<_> = payloads.iter().map(vectorize).collect();
    if vectors.is_empty() {
        anyhow::bail!("payloads file has no entries");
    }

    let mut durations = Vec::with_capacity(iterations);
    let mut checksum = 0.0f32;
    for idx in 0..iterations {
        let vector = &vectors[idx % vectors.len()];
        let started_at = Instant::now();
        let score = index.fraud_score(vector, None).unwrap_or(0.0);
        durations.push(started_at.elapsed().as_nanos() as u64);
        checksum += score;
    }

    durations.sort_unstable();
    let p50 = percentile(&durations, 50);
    let p95 = percentile(&durations, 95);
    let p99 = percentile(&durations, 99);
    let avg = durations.iter().sum::<u64>() / durations.len() as u64;
    println!(
        "records={} payloads={} iterations={} avg={}us p50={}us p95={}us p99={}us checksum={:.3}",
        index.len(),
        vectors.len(),
        iterations,
        avg / 1_000,
        p50 / 1_000,
        p95 / 1_000,
        p99 / 1_000,
        checksum
    );
    Ok(())
}

fn percentile(values: &[u64], percentile: usize) -> u64 {
    let idx = ((values.len() - 1) * percentile) / 100;
    values[idx]
}
