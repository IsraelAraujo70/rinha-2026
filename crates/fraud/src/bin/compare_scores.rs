use std::{env, fs::File, path::PathBuf};

use anyhow::{Context, Result};
use fraud::{
    index::{Index, SearchResult},
    payload::FraudRequest,
    vector::{vectorize, Vector},
};

fn main() -> Result<()> {
    let mut args = env::args_os().skip(1);
    let expected_path = args
        .next()
        .map(PathBuf::from)
        .context("usage: compare_scores <expected.bin> <candidate.bin> <payloads.json>")?;
    let candidate_path = args
        .next()
        .map(PathBuf::from)
        .context("usage: compare_scores <expected.bin> <candidate.bin> <payloads.json>")?;
    let payloads_path = args
        .next()
        .map(PathBuf::from)
        .context("usage: compare_scores <expected.bin> <candidate.bin> <payloads.json>")?;

    let expected = Index::open(&expected_path)?;
    let candidate = Index::open(&candidate_path)?;
    let payloads_file =
        File::open(&payloads_path).with_context(|| format!("open {}", payloads_path.display()))?;
    let payloads: Vec<FraudRequest> = serde_json::from_reader(payloads_file)
        .with_context(|| format!("parse {}", payloads_path.display()))?;

    let mut score_mismatches = 0usize;
    let mut decision_mismatches = 0usize;
    let mut max_delta = 0.0f32;

    for payload in &payloads {
        let vector = vectorize(payload);
        let expected_score = score(&expected, &vector);
        let candidate_score = score(&candidate, &vector);
        let delta = (expected_score - candidate_score).abs();
        max_delta = max_delta.max(delta);

        if delta > f32::EPSILON {
            score_mismatches += 1;
        }
        if is_approved(expected_score) != is_approved(candidate_score) {
            decision_mismatches += 1;
        }
    }

    println!(
        "payloads={} score_mismatches={} decision_mismatches={} max_delta={:.3}",
        payloads.len(),
        score_mismatches,
        decision_mismatches,
        max_delta
    );
    Ok(())
}

fn score(index: &Index, vector: &Vector) -> f32 {
    match index.fraud_score(vector, None) {
        SearchResult::Score(score) => score,
        SearchResult::TimedOut => unreachable!("comparison runs without a deadline"),
    }
}

fn is_approved(score: f32) -> bool {
    score < 0.6
}
