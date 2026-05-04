use chrono::{DateTime, Utc};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct FraudRequest {
    pub id: String,
    pub transaction: Transaction,
    pub customer: Customer,
    pub merchant: Merchant,
    pub terminal: Terminal,
    pub last_transaction: Option<LastTransaction>,
}

#[derive(Debug, Deserialize)]
pub struct Transaction {
    pub amount: f32,
    pub installments: u8,
    pub requested_at: DateTime<Utc>,
}

#[derive(Debug, Deserialize)]
pub struct Customer {
    pub avg_amount: f32,
    pub tx_count_24h: u16,
    pub known_merchants: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct Merchant {
    pub id: String,
    pub mcc: String,
    pub avg_amount: f32,
}

#[derive(Debug, Deserialize)]
pub struct Terminal {
    pub is_online: bool,
    pub card_present: bool,
    pub km_from_home: f32,
}

#[derive(Debug, Deserialize)]
pub struct LastTransaction {
    pub timestamp: DateTime<Utc>,
    pub km_from_current: f32,
}
