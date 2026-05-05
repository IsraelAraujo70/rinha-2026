use std::{env, path::PathBuf, time::Duration};

use fraud::{
    fast_parse::parse_and_vectorize,
    index::{Index, SearchResult},
};
use mimalloc::MiMalloc;
use monoio::{
    buf::SliceMut,
    io::{AsyncReadRent, AsyncWriteRentExt},
    net::{ListenerOpts, TcpListener, UnixListener},
};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const DEFAULT_KNN_TIMEOUT_US: u64 = 1_000;

const FRAUD_RESPONSES: [&[u8]; 6] = [
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 35\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}",
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 36\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}",
];
const FRAUD_FALLBACK: &[u8] = FRAUD_RESPONSES[0];
const READY_RESPONSE: &[u8] = b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n";

fn main() -> anyhow::Result<()> {
    let index_path = env::var_os("INDEX_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/index/data.bin"));
    let index: &'static Index = Box::leak(Box::new(Index::open(&index_path)?));
    let knn_timeout = configured_timeout();

    let uds_path = env::var_os("API_UDS_PATH").map(PathBuf::from).filter(|p| !p.as_os_str().is_empty());

    if let Some(path) = uds_path {
        // io_uring + AF_UNIX has had subtle issues with subsequent reads on a
        // kept-alive connection (the second recv blocks even when data has
        // already arrived). epoll is fine here; for the throughput levels in
        // this challenge it is not the bottleneck.
        let mut rt = monoio::RuntimeBuilder::<monoio::LegacyDriver>::new()
            .with_entries(256)
            .enable_timer()
            .build()?;
        let _ = std::fs::remove_file(&path);
        // SO_REUSEPORT is meaningless for AF_UNIX; setsockopt returns ENOTSUP.
        let opts = ListenerOpts::new().reuse_port(false).reuse_addr(false);
        rt.block_on(async move {
            let listener = UnixListener::bind_with_config(&path, &opts).expect("bind unix");
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o666));
            loop {
                match listener.accept().await {
                    Ok((stream, _)) => {
                        monoio::spawn(handle_connection(stream, index, knn_timeout));
                        // Without an explicit yield here, the spawned UDS task
                        // never gets polled — accept().await loops back to a
                        // fresh op submission and the runtime picks I/O over
                        // draining its task queue. A 1 ns timer is enough to
                        // force the inner task drain.
                        monoio::time::sleep(std::time::Duration::from_nanos(1)).await;
                    }
                    Err(_) => continue,
                }
            }
        });
    } else {
        let addr_str = env::var("API_ADDR").unwrap_or_else(|_| "0.0.0.0:8080".to_string());
        let addr: std::net::SocketAddr = addr_str.parse()?;
        let mut rt = monoio::RuntimeBuilder::<monoio::FusionDriver>::new()
            .with_entries(256)
            .build()?;
        rt.block_on(async move {
            let listener = TcpListener::bind(addr).expect("bind");
            loop {
                match listener.accept().await {
                    Ok((stream, _)) => {
                        let _ = stream.set_nodelay(true);
                        monoio::spawn(handle_connection(stream, index, knn_timeout));
                    }
                    Err(_) => continue,
                }
            }
        });
    }
    Ok(())
}

fn configured_timeout() -> Duration {
    Duration::from_micros(
        env::var("KNN_TIMEOUT_US")
            .ok()
            .and_then(|raw| raw.parse::<u64>().ok())
            .filter(|micros| *micros > 0)
            .unwrap_or(DEFAULT_KNN_TIMEOUT_US),
    )
}

async fn handle_connection<S>(mut stream: S, index: &'static Index, knn_timeout: Duration)
where
    S: AsyncReadRent + AsyncWriteRentExt,
{
    // 8 KB initial: covers any conformant Rinha payload (~600 B) and HTTP
    // headers nginx adds (~250 B). Box<[u8]> has fixed len; monoio's set_init
    // is a no-op for it, so the buffer state does not get corrupted across
    // consecutive reads on a kept-alive connection (Vec<u8> went down to
    // bytes-just-read and broke the next read submission).
    let mut buf: Box<[u8]> = vec![0u8; 8192].into_boxed_slice();
    let mut start: usize = 0;
    let mut filled: usize = 0;

    loop {
        let head_end = loop {
            if let Some(rel) = find_headers_end(&buf[start..filled]) {
                break start + rel + 4;
            }
            if filled == buf.len() {
                if start > 0 {
                    buf.copy_within(start..filled, 0);
                    filled -= start;
                    start = 0;
                } else {
                    // grow Box by reallocating
                    let mut grown = vec![0u8; buf.len() * 2].into_boxed_slice();
                    grown[..buf.len()].copy_from_slice(&buf);
                    buf = grown;
                }
            }
            let cap = buf.len();
            let slice = unsafe { SliceMut::new_unchecked(buf, filled, cap) };
            let (res, returned) = stream.read(slice).await;
            buf = returned.into_inner();
            match res {
                Ok(0) => return,
                Ok(n) => filled += n,
                Err(_) => return,
            }
        };

        let (method, content_len) = parse_request_head(&buf[start..head_end - 4]);

        let next_start = if method == Method::Post {
            let total = head_end + content_len;
            if total > buf.len() {
                let mut grown = vec![0u8; total].into_boxed_slice();
                grown[..buf.len()].copy_from_slice(&buf);
                buf = grown;
            }
            while filled < total {
                let cap = buf.len();
                let slice = unsafe { SliceMut::new_unchecked(buf, filled, cap) };
                let (res, returned) = stream.read(slice).await;
                buf = returned.into_inner();
                match res {
                    Ok(0) => return,
                    Ok(n) => filled += n,
                    Err(_) => return,
                }
            }
            let body = &buf[head_end..total];
            let response = score_body(index, knn_timeout, body);
            let (res, _) = stream.write_all(response).await;
            if res.is_err() {
                return;
            }
            total
        } else {
            let (res, _) = stream.write_all(READY_RESPONSE).await;
            if res.is_err() {
                return;
            }
            head_end
        };

        if next_start == filled {
            start = 0;
            filled = 0;
        } else {
            start = next_start;
        }
    }
}

#[inline]
fn find_headers_end(buf: &[u8]) -> Option<usize> {
    if buf.len() < 4 {
        return None;
    }
    memchr_end(buf)
}

#[inline]
fn memchr_end(buf: &[u8]) -> Option<usize> {
    let mut i = 0;
    let end = buf.len().saturating_sub(3);
    while i < end {
        if buf[i] == b'\r'
            && buf[i + 1] == b'\n'
            && buf[i + 2] == b'\r'
            && buf[i + 3] == b'\n'
        {
            return Some(i);
        }
        i += 1;
    }
    None
}

#[derive(PartialEq, Eq)]
enum Method {
    Post,
    Other,
}

fn parse_request_head(headers: &[u8]) -> (Method, usize) {
    let method = if headers.starts_with(b"POST ") {
        Method::Post
    } else {
        Method::Other
    };
    let content_len = find_content_length(headers).unwrap_or(0);
    (method, content_len)
}

fn find_content_length(headers: &[u8]) -> Option<usize> {
    const NEEDLE: &[u8] = b"Content-Length:";
    let mut i = 0;
    while i + NEEDLE.len() <= headers.len() {
        let slice = &headers[i..i + NEEDLE.len()];
        let matches = slice == NEEDLE
            || (slice.len() == NEEDLE.len()
                && slice
                    .iter()
                    .zip(NEEDLE.iter())
                    .all(|(a, b)| a.eq_ignore_ascii_case(b)));
        if matches {
            let mut j = i + NEEDLE.len();
            while j < headers.len() && (headers[j] == b' ' || headers[j] == b'\t') {
                j += 1;
            }
            let mut val: usize = 0;
            let mut any_digit = false;
            while j < headers.len() && headers[j].is_ascii_digit() {
                val = val * 10 + (headers[j] - b'0') as usize;
                any_digit = true;
                j += 1;
            }
            return any_digit.then_some(val);
        }
        i += 1;
    }
    None
}

#[inline]
fn score_body(index: &Index, knn_timeout: Duration, body: &[u8]) -> &'static [u8] {
    let vector = match parse_and_vectorize(body) {
        Some(v) => v,
        None => return FRAUD_FALLBACK,
    };
    let fraud_score = match index.fraud_score(&vector, Some(knn_timeout)) {
        SearchResult::Score(s) => s,
        SearchResult::TimedOut => return FRAUD_FALLBACK,
    };
    let count = (fraud_score * 5.0).round() as usize;
    FRAUD_RESPONSES[count.min(5)]
}
