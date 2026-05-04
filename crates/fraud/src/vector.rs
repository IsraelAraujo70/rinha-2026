use chrono::{Datelike, Timelike};

use crate::payload::FraudRequest;

pub const DIMS: usize = 14;

pub type Vector = [f32; DIMS];
pub type QuantizedVector = [u8; DIMS];
pub type QuantizedI16Vector = [i16; DIMS];

const MAX_AMOUNT: f32 = 10_000.0;
const MAX_INSTALLMENTS: f32 = 12.0;
const AMOUNT_VS_AVG_RATIO: f32 = 10.0;
const MAX_MINUTES: f32 = 1_440.0;
const MAX_KM: f32 = 1_000.0;
const MAX_TX_COUNT_24H: f32 = 20.0;
const MAX_MERCHANT_AVG_AMOUNT: f32 = 10_000.0;

#[inline]
fn clamp01(value: f32) -> f32 {
    if value.is_nan() {
        0.0
    } else {
        value.clamp(0.0, 1.0)
    }
}

#[inline]
pub fn mcc_risk(mcc: &str) -> f32 {
    match mcc {
        "5411" => 0.15,
        "5812" => 0.30,
        "5912" => 0.20,
        "5944" => 0.45,
        "7801" => 0.80,
        "7802" => 0.75,
        "7995" => 0.85,
        "4511" => 0.35,
        "5311" => 0.25,
        "5999" => 0.50,
        _ => 0.50,
    }
}

pub fn vectorize(request: &FraudRequest) -> Vector {
    let requested_at = request.transaction.requested_at;
    let avg_amount = request.customer.avg_amount.max(0.01);
    let unknown_merchant = !request
        .customer
        .known_merchants
        .iter()
        .any(|known| known == &request.merchant.id);

    let mut vector = [0.0; DIMS];
    vector[0] = clamp01(request.transaction.amount / MAX_AMOUNT);
    vector[1] = clamp01(request.transaction.installments as f32 / MAX_INSTALLMENTS);
    vector[2] = clamp01((request.transaction.amount / avg_amount) / AMOUNT_VS_AVG_RATIO);
    vector[3] = requested_at.hour() as f32 / 23.0;
    vector[4] = requested_at.weekday().num_days_from_monday() as f32 / 6.0;

    if let Some(last) = &request.last_transaction {
        let minutes = (requested_at - last.timestamp).num_seconds().max(0) as f32 / 60.0;
        vector[5] = clamp01(minutes / MAX_MINUTES);
        vector[6] = clamp01(last.km_from_current / MAX_KM);
    } else {
        vector[5] = -1.0;
        vector[6] = -1.0;
    }

    vector[7] = clamp01(request.terminal.km_from_home / MAX_KM);
    vector[8] = clamp01(request.customer.tx_count_24h as f32 / MAX_TX_COUNT_24H);
    vector[9] = if request.terminal.is_online { 1.0 } else { 0.0 };
    vector[10] = if request.terminal.card_present {
        1.0
    } else {
        0.0
    };
    vector[11] = if unknown_merchant { 1.0 } else { 0.0 };
    vector[12] = mcc_risk(&request.merchant.mcc);
    vector[13] = clamp01(request.merchant.avg_amount / MAX_MERCHANT_AVG_AMOUNT);
    vector
}

pub fn quantize(vector: &Vector) -> QuantizedVector {
    let mut output = [0; DIMS];
    for (idx, value) in vector.iter().enumerate() {
        output[idx] = quantize_dim(*value);
    }
    output
}

#[inline]
pub fn quantize_dim(value: f32) -> u8 {
    if value < 0.0 {
        0
    } else {
        let scaled = 128.0 + clamp01(value) * 127.0;
        scaled.round() as u8
    }
}

pub fn quantize_i16(vector: &Vector) -> QuantizedI16Vector {
    let mut output = [0; DIMS];
    for (idx, value) in vector.iter().enumerate() {
        output[idx] = quantize_i16_dim(*value);
    }
    output
}

#[inline]
pub fn quantize_i16_dim(value: f32) -> i16 {
    if value < 0.0 {
        -10_000
    } else {
        (clamp01(value) * 10_000.0).round() as i16
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::payload::FraudRequest;

    #[test]
    fn vectorizes_official_legit_example() {
        let payload = r#"{
          "id": "tx-1329056812",
          "transaction": { "amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z" },
          "customer": { "avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016"] },
          "merchant": { "id": "MERC-016", "mcc": "5411", "avg_amount": 60.25 },
          "terminal": { "is_online": false, "card_present": true, "km_from_home": 29.23 },
          "last_transaction": null
        }"#;
        let request: FraudRequest = serde_json::from_str(payload).unwrap();
        let vector = vectorize(&request);

        assert!((vector[0] - 0.0041).abs() < 0.0001);
        assert!((vector[1] - 0.1667).abs() < 0.0001);
        assert!((vector[2] - 0.05).abs() < 0.0001);
        assert!((vector[3] - 0.7826).abs() < 0.0001);
        assert!((vector[4] - 0.3333).abs() < 0.0001);
        assert_eq!(vector[5], -1.0);
        assert_eq!(vector[6], -1.0);
        assert!((vector[7] - 0.0292).abs() < 0.0001);
        assert!((vector[8] - 0.15).abs() < 0.0001);
        assert_eq!(vector[9], 0.0);
        assert_eq!(vector[10], 1.0);
        assert_eq!(vector[11], 0.0);
        assert_eq!(vector[12], 0.15);
        assert!((vector[13] - 0.006).abs() < 0.0001);
    }
}
