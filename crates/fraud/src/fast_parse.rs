use crate::vector::{Vector, DIMS};

const MAX_AMOUNT: f32 = 10_000.0;
const MAX_INSTALLMENTS: f32 = 12.0;
const AMOUNT_VS_AVG_RATIO: f32 = 10.0;
const MAX_MINUTES: f32 = 1_440.0;
const MAX_KM: f32 = 1_000.0;
const MAX_TX_COUNT_24H: f32 = 20.0;
const MAX_MERCHANT_AVG_AMOUNT: f32 = 10_000.0;

pub fn parse_and_vectorize(body: &[u8]) -> Option<Vector> {
    let mut p = Parser::new(body);
    parse_root(&mut p)
}

fn parse_root(p: &mut Parser) -> Option<Vector> {
    p.skip_ws();
    p.expect(b'{')?;

    let mut amount = 0.0f32;
    let mut installments = 0.0f32;
    let mut req_hour = 0u8;
    let mut req_weekday = 0u8;
    let mut req_epoch = 0i64;

    let mut avg_amount = 0.0f32;
    let mut tx_count_24h = 0.0f32;
    let mut known_slice: &[u8] = &[];

    let mut merchant_id: &[u8] = &[];
    let mut mcc: &[u8] = &[];
    let mut merchant_avg = 0.0f32;

    let mut is_online = false;
    let mut card_present = false;
    let mut km_from_home = 0.0f32;

    let mut last_present = false;
    let mut last_epoch = 0i64;
    let mut last_km = 0.0f32;

    let mut first = true;
    loop {
        p.skip_ws();
        if p.peek()? == b'}' {
            p.advance();
            break;
        }
        if !first {
            p.expect(b',')?;
            p.skip_ws();
        }
        first = false;

        let key = p.parse_key()?;
        p.skip_ws();
        p.expect(b':')?;
        p.skip_ws();

        if key == b"transaction" {
            parse_transaction(p, &mut amount, &mut installments, &mut req_hour, &mut req_weekday, &mut req_epoch)?;
        } else if key == b"customer" {
            parse_customer(p, &mut avg_amount, &mut tx_count_24h, &mut known_slice)?;
        } else if key == b"merchant" {
            parse_merchant(p, &mut merchant_id, &mut mcc, &mut merchant_avg)?;
        } else if key == b"terminal" {
            parse_terminal(p, &mut is_online, &mut card_present, &mut km_from_home)?;
        } else if key == b"last_transaction" {
            if p.peek()? == b'n' {
                p.parse_null()?;
            } else {
                last_present = true;
                parse_last_tx(p, &mut last_epoch, &mut last_km)?;
            }
        } else if key == b"id" {
            p.skip_string()?;
        } else {
            p.skip_value()?;
        }
    }

    let unknown_merchant = scan_unknown(known_slice, merchant_id);
    let avg_for_ratio = if avg_amount > 0.01 { avg_amount } else { 0.01 };

    let mut v = [0.0f32; DIMS];
    v[0] = clamp01(amount / MAX_AMOUNT);
    v[1] = clamp01(installments / MAX_INSTALLMENTS);
    v[2] = clamp01((amount / avg_for_ratio) / AMOUNT_VS_AVG_RATIO);
    v[3] = req_hour as f32 / 23.0;
    v[4] = req_weekday as f32 / 6.0;
    if last_present {
        let minutes = (req_epoch - last_epoch).max(0) as f32 / 60.0;
        v[5] = clamp01(minutes / MAX_MINUTES);
        v[6] = clamp01(last_km / MAX_KM);
    } else {
        v[5] = -1.0;
        v[6] = -1.0;
    }
    v[7] = clamp01(km_from_home / MAX_KM);
    v[8] = clamp01(tx_count_24h / MAX_TX_COUNT_24H);
    v[9] = if is_online { 1.0 } else { 0.0 };
    v[10] = if card_present { 1.0 } else { 0.0 };
    v[11] = if unknown_merchant { 1.0 } else { 0.0 };
    v[12] = mcc_risk_bytes(mcc);
    v[13] = clamp01(merchant_avg / MAX_MERCHANT_AVG_AMOUNT);
    Some(v)
}

fn parse_transaction(
    p: &mut Parser,
    amount: &mut f32,
    installments: &mut f32,
    req_hour: &mut u8,
    req_weekday: &mut u8,
    req_epoch: &mut i64,
) -> Option<()> {
    p.expect(b'{')?;
    let mut first = true;
    loop {
        p.skip_ws();
        if p.peek()? == b'}' { p.advance(); return Some(()); }
        if !first { p.expect(b',')?; p.skip_ws(); }
        first = false;
        let key = p.parse_key()?;
        p.skip_ws(); p.expect(b':')?; p.skip_ws();
        if key == b"amount" {
            *amount = p.parse_f32()?;
        } else if key == b"installments" {
            *installments = p.parse_uint()? as f32;
        } else if key == b"requested_at" {
            let s = p.parse_string()?;
            let (h, w, e) = parse_iso8601(s)?;
            *req_hour = h;
            *req_weekday = w;
            *req_epoch = e;
        } else {
            p.skip_value()?;
        }
    }
}

fn parse_customer<'a>(
    p: &mut Parser<'a>,
    avg_amount: &mut f32,
    tx_count_24h: &mut f32,
    known_slice: &mut &'a [u8],
) -> Option<()> {
    p.expect(b'{')?;
    let mut first = true;
    loop {
        p.skip_ws();
        if p.peek()? == b'}' { p.advance(); return Some(()); }
        if !first { p.expect(b',')?; p.skip_ws(); }
        first = false;
        let key = p.parse_key()?;
        p.skip_ws(); p.expect(b':')?; p.skip_ws();
        if key == b"avg_amount" {
            *avg_amount = p.parse_f32()?;
        } else if key == b"tx_count_24h" {
            *tx_count_24h = p.parse_uint()? as f32;
        } else if key == b"known_merchants" {
            *known_slice = p.parse_array_slice()?;
        } else {
            p.skip_value()?;
        }
    }
}

fn parse_merchant<'a>(
    p: &mut Parser<'a>,
    id: &mut &'a [u8],
    mcc: &mut &'a [u8],
    avg: &mut f32,
) -> Option<()> {
    p.expect(b'{')?;
    let mut first = true;
    loop {
        p.skip_ws();
        if p.peek()? == b'}' { p.advance(); return Some(()); }
        if !first { p.expect(b',')?; p.skip_ws(); }
        first = false;
        let key = p.parse_key()?;
        p.skip_ws(); p.expect(b':')?; p.skip_ws();
        if key == b"id" {
            *id = p.parse_string()?;
        } else if key == b"mcc" {
            *mcc = p.parse_string()?;
        } else if key == b"avg_amount" {
            *avg = p.parse_f32()?;
        } else {
            p.skip_value()?;
        }
    }
}

fn parse_terminal(
    p: &mut Parser,
    is_online: &mut bool,
    card_present: &mut bool,
    km_from_home: &mut f32,
) -> Option<()> {
    p.expect(b'{')?;
    let mut first = true;
    loop {
        p.skip_ws();
        if p.peek()? == b'}' { p.advance(); return Some(()); }
        if !first { p.expect(b',')?; p.skip_ws(); }
        first = false;
        let key = p.parse_key()?;
        p.skip_ws(); p.expect(b':')?; p.skip_ws();
        if key == b"is_online" {
            *is_online = p.parse_bool()?;
        } else if key == b"card_present" {
            *card_present = p.parse_bool()?;
        } else if key == b"km_from_home" {
            *km_from_home = p.parse_f32()?;
        } else {
            p.skip_value()?;
        }
    }
}

fn parse_last_tx(p: &mut Parser, last_epoch: &mut i64, last_km: &mut f32) -> Option<()> {
    p.expect(b'{')?;
    let mut first = true;
    loop {
        p.skip_ws();
        if p.peek()? == b'}' { p.advance(); return Some(()); }
        if !first { p.expect(b',')?; p.skip_ws(); }
        first = false;
        let key = p.parse_key()?;
        p.skip_ws(); p.expect(b':')?; p.skip_ws();
        if key == b"timestamp" {
            let s = p.parse_string()?;
            let (_h, _w, e) = parse_iso8601(s)?;
            *last_epoch = e;
        } else if key == b"km_from_current" {
            *last_km = p.parse_f32()?;
        } else {
            p.skip_value()?;
        }
    }
}

#[inline]
fn clamp01(v: f32) -> f32 {
    if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) }
}

fn mcc_risk_bytes(mcc: &[u8]) -> f32 {
    if mcc.len() != 4 { return 0.50; }
    match mcc {
        b"5411" => 0.15,
        b"5812" => 0.30,
        b"5912" => 0.20,
        b"5944" => 0.45,
        b"7801" => 0.80,
        b"7802" => 0.75,
        b"7995" => 0.85,
        b"4511" => 0.35,
        b"5311" => 0.25,
        b"5999" => 0.50,
        _ => 0.50,
    }
}

fn scan_unknown(known: &[u8], id: &[u8]) -> bool {
    if id.is_empty() { return true; }
    let mut i = 0;
    while i < known.len() {
        while i < known.len() && known[i] != b'"' { i += 1; }
        if i >= known.len() { break; }
        i += 1;
        let start = i;
        while i < known.len() && known[i] != b'"' { i += 1; }
        if i > known.len() { break; }
        if &known[start..i] == id {
            return false;
        }
        if i < known.len() { i += 1; }
    }
    true
}

struct Parser<'a> {
    body: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(body: &'a [u8]) -> Self {
        Self { body, pos: 0 }
    }

    #[inline]
    fn peek(&self) -> Option<u8> {
        self.body.get(self.pos).copied()
    }

    #[inline]
    fn advance(&mut self) {
        self.pos += 1;
    }

    #[inline]
    fn skip_ws(&mut self) {
        while self.pos < self.body.len() {
            let b = self.body[self.pos];
            if b == b' ' || b == b'\t' || b == b'\r' || b == b'\n' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    #[inline]
    fn expect(&mut self, want: u8) -> Option<()> {
        if self.peek()? == want {
            self.pos += 1;
            Some(())
        } else {
            None
        }
    }

    fn parse_key(&mut self) -> Option<&'a [u8]> {
        self.expect(b'"')?;
        let start = self.pos;
        while self.pos < self.body.len() && self.body[self.pos] != b'"' {
            self.pos += 1;
        }
        if self.pos >= self.body.len() { return None; }
        let end = self.pos;
        self.pos += 1;
        Some(&self.body[start..end])
    }

    fn parse_string(&mut self) -> Option<&'a [u8]> {
        self.parse_key()
    }

    fn skip_string(&mut self) -> Option<()> {
        self.expect(b'"')?;
        while self.pos < self.body.len() && self.body[self.pos] != b'"' {
            if self.body[self.pos] == b'\\' { self.pos += 2; } else { self.pos += 1; }
        }
        if self.pos >= self.body.len() { return None; }
        self.pos += 1;
        Some(())
    }

    fn parse_uint(&mut self) -> Option<u64> {
        let start = self.pos;
        while self.pos < self.body.len() && self.body[self.pos].is_ascii_digit() {
            self.pos += 1;
        }
        if self.pos == start { return None; }
        let mut v = 0u64;
        for &b in &self.body[start..self.pos] {
            v = v * 10 + (b - b'0') as u64;
        }
        Some(v)
    }

    fn parse_f32(&mut self) -> Option<f32> {
        let start = self.pos;
        if self.peek()? == b'-' { self.pos += 1; }
        while self.pos < self.body.len() {
            let b = self.body[self.pos];
            if b.is_ascii_digit() || b == b'.' || b == b'e' || b == b'E' || b == b'+' || b == b'-' {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start { return None; }
        let s = unsafe { std::str::from_utf8_unchecked(&self.body[start..self.pos]) };
        s.parse::<f32>().ok()
    }

    fn parse_bool(&mut self) -> Option<bool> {
        if self.body[self.pos..].starts_with(b"true") {
            self.pos += 4;
            Some(true)
        } else if self.body[self.pos..].starts_with(b"false") {
            self.pos += 5;
            Some(false)
        } else {
            None
        }
    }

    fn parse_null(&mut self) -> Option<()> {
        if self.body[self.pos..].starts_with(b"null") {
            self.pos += 4;
            Some(())
        } else {
            None
        }
    }

    fn parse_array_slice(&mut self) -> Option<&'a [u8]> {
        self.expect(b'[')?;
        let start = self.pos;
        let mut depth = 1i32;
        while self.pos < self.body.len() && depth > 0 {
            match self.body[self.pos] {
                b'"' => {
                    self.pos += 1;
                    while self.pos < self.body.len() && self.body[self.pos] != b'"' {
                        if self.body[self.pos] == b'\\' { self.pos += 1; }
                        self.pos += 1;
                    }
                    if self.pos < self.body.len() { self.pos += 1; }
                }
                b'[' => { depth += 1; self.pos += 1; }
                b']' => { depth -= 1; if depth == 0 { break; } self.pos += 1; }
                _ => self.pos += 1,
            }
        }
        if depth != 0 { return None; }
        let end = self.pos;
        self.pos += 1;
        Some(&self.body[start..end])
    }

    fn skip_value(&mut self) -> Option<()> {
        self.skip_ws();
        let b = self.peek()?;
        match b {
            b'"' => self.skip_string(),
            b'{' => self.skip_object(),
            b'[' => self.skip_array_value(),
            b't' | b'f' => self.parse_bool().map(|_| ()),
            b'n' => self.parse_null(),
            c if c.is_ascii_digit() || c == b'-' => self.skip_number(),
            _ => None,
        }
    }

    fn skip_number(&mut self) -> Option<()> {
        if self.peek()? == b'-' { self.pos += 1; }
        while self.pos < self.body.len() {
            let b = self.body[self.pos];
            if b.is_ascii_digit() || b == b'.' || b == b'e' || b == b'E' || b == b'+' || b == b'-' {
                self.pos += 1;
            } else { break; }
        }
        Some(())
    }

    fn skip_object(&mut self) -> Option<()> {
        self.expect(b'{')?;
        let mut depth = 1i32;
        while self.pos < self.body.len() && depth > 0 {
            match self.body[self.pos] {
                b'"' => { self.skip_string()?; }
                b'{' => { depth += 1; self.pos += 1; }
                b'}' => { depth -= 1; self.pos += 1; }
                _ => self.pos += 1,
            }
        }
        if depth != 0 { None } else { Some(()) }
    }

    fn skip_array_value(&mut self) -> Option<()> {
        self.expect(b'[')?;
        let mut depth = 1i32;
        while self.pos < self.body.len() && depth > 0 {
            match self.body[self.pos] {
                b'"' => { self.skip_string()?; }
                b'[' => { depth += 1; self.pos += 1; }
                b']' => { depth -= 1; self.pos += 1; }
                b'{' => { self.skip_object()?; }
                _ => self.pos += 1,
            }
        }
        if depth != 0 { None } else { Some(()) }
    }
}

fn ascii_to_u32(b: &[u8]) -> Option<u32> {
    let mut v = 0u32;
    for &c in b {
        if !c.is_ascii_digit() { return None; }
        v = v * 10 + (c - b'0') as u32;
    }
    Some(v)
}

fn parse_iso8601(s: &[u8]) -> Option<(u8, u8, i64)> {
    if s.len() < 19 { return None; }
    let year = ascii_to_u32(&s[0..4])? as i32;
    let month = ascii_to_u32(&s[5..7])?;
    let day = ascii_to_u32(&s[8..10])?;
    let hour = ascii_to_u32(&s[11..13])? as u8;
    let minute = ascii_to_u32(&s[14..16])?;
    let second = ascii_to_u32(&s[17..19])?;

    let epoch_days = days_from_civil(year, month, day);
    let epoch_seconds =
        epoch_days * 86_400 + hour as i64 * 3_600 + minute as i64 * 60 + second as i64;
    let weekday = (((epoch_days + 3) % 7 + 7) % 7) as u8;
    Some((hour, weekday, epoch_seconds))
}

// Howard Hinnant's date algorithm; days from 1970-01-01 (Gregorian, proleptic).
fn days_from_civil(y: i32, m: u32, d: u32) -> i64 {
    let mi = m as i32;
    let di = d as i32;
    let y = y - if mi <= 2 { 1 } else { 0 };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as i64;
    let doy_m = if mi > 2 { mi - 3 } else { mi + 9 };
    let doy = ((153 * doy_m + 2) / 5 + di - 1) as i64;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era as i64 * 146_097 + doe - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::payload::FraudRequest;
    use crate::vector::vectorize;

    fn assert_close(a: [f32; DIMS], b: [f32; DIMS], eps: f32) {
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= eps, "dim {i}: fast={x} ref={y}");
        }
    }

    #[test]
    fn matches_vectorize_for_official_legit() {
        let payload = br#"{
          "id": "tx-1329056812",
          "transaction": { "amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z" },
          "customer": { "avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016"] },
          "merchant": { "id": "MERC-016", "mcc": "5411", "avg_amount": 60.25 },
          "terminal": { "is_online": false, "card_present": true, "km_from_home": 29.23 },
          "last_transaction": null
        }"#;
        let fast = parse_and_vectorize(payload).expect("fast parser");
        let req: FraudRequest = serde_json::from_slice(payload).unwrap();
        let reference = vectorize(&req);
        assert_close(fast, reference, 1e-5);
    }

    #[test]
    fn matches_vectorize_with_last_transaction() {
        let payload = br#"{
          "id": "tx-99",
          "transaction": { "amount": 1500.50, "installments": 6, "requested_at": "2026-04-20T03:15:00Z" },
          "customer": { "avg_amount": 200.00, "tx_count_24h": 12, "known_merchants": ["MERC-A"] },
          "merchant": { "id": "MERC-Z", "mcc": "7995", "avg_amount": 5000.0 },
          "terminal": { "is_online": true, "card_present": false, "km_from_home": 250.5 },
          "last_transaction": { "timestamp": "2026-04-20T02:45:30Z", "km_from_current": 12.7 }
        }"#;
        let fast = parse_and_vectorize(payload).expect("fast parser");
        let req: FraudRequest = serde_json::from_slice(payload).unwrap();
        let reference = vectorize(&req);
        assert_close(fast, reference, 1e-5);
    }

    #[test]
    fn matches_vectorize_for_known_merchant() {
        let payload = br#"{
          "id": "tx-1",
          "transaction": { "amount": 50.0, "installments": 1, "requested_at": "2025-01-01T00:00:00Z" },
          "customer": { "avg_amount": 75.0, "tx_count_24h": 1, "known_merchants": ["A","B","TARGET","C"] },
          "merchant": { "id": "TARGET", "mcc": "5411", "avg_amount": 50.0 },
          "terminal": { "is_online": true, "card_present": true, "km_from_home": 0.0 },
          "last_transaction": null
        }"#;
        let fast = parse_and_vectorize(payload).expect("fast parser");
        let req: FraudRequest = serde_json::from_slice(payload).unwrap();
        let reference = vectorize(&req);
        assert_close(fast, reference, 1e-5);
        assert_eq!(fast[11], 0.0);
    }

    #[test]
    fn iso8601_dow_thursday_1970() {
        let (_, w, e) = parse_iso8601(b"1970-01-01T00:00:00Z").unwrap();
        assert_eq!(e, 0);
        assert_eq!(w, 3); // Thursday = 3 (num_days_from_monday)
    }

    #[test]
    fn iso8601_dow_known_dates() {
        // 2026-03-11 is a Wednesday
        let (_, w, _) = parse_iso8601(b"2026-03-11T18:45:53Z").unwrap();
        assert_eq!(w, 2);
        // 2026-04-20 is a Monday
        let (_, w, _) = parse_iso8601(b"2026-04-20T00:00:00Z").unwrap();
        assert_eq!(w, 0);
    }

    #[test]
    fn rejects_truncated() {
        assert!(parse_and_vectorize(b"{").is_none());
        assert!(parse_and_vectorize(b"{\"id\":").is_none());
    }
}
