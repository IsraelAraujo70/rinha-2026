use std::time::Instant;

use fraud::{fast_parse::parse_and_vectorize, payload::FraudRequest, vector::vectorize};

const PAYLOAD: &[u8] = br#"{
  "id": "tx-1329056812",
  "transaction": { "amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z" },
  "customer": { "avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016", "MERC-021", "MERC-042", "MERC-077"] },
  "merchant": { "id": "MERC-016", "mcc": "5411", "avg_amount": 60.25 },
  "terminal": { "is_online": false, "card_present": true, "km_from_home": 29.23 },
  "last_transaction": { "timestamp": "2026-03-11T17:30:21Z", "km_from_current": 12.5 }
}"#;

fn main() {
    let iters: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200_000);

    let mut sink = 0.0f32;
    let started = Instant::now();
    for _ in 0..iters {
        let v = parse_and_vectorize(PAYLOAD).expect("fast");
        sink += v[0] + v[12];
    }
    let fast_ns = started.elapsed().as_nanos() as u64 / iters as u64;

    let started = Instant::now();
    for _ in 0..iters {
        let req: FraudRequest = serde_json::from_slice(PAYLOAD).expect("serde");
        let v = vectorize(&req);
        sink += v[0] + v[12];
    }
    let serde_ns = started.elapsed().as_nanos() as u64 / iters as u64;

    println!(
        "iters={iters} fast={fast_ns}ns/op serde={serde_ns}ns/op speedup={:.2}x sink={:.4}",
        serde_ns as f64 / fast_ns as f64,
        sink
    );
}
