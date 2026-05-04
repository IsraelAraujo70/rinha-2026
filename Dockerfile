FROM --platform=$BUILDPLATFORM rust:1-bookworm AS builder

WORKDIR /app
COPY Cargo.toml ./
COPY Cargo.lock ./
COPY .cargo ./.cargo
COPY crates ./crates
RUN cargo build --release --bin api --bin build_index

FROM debian:bookworm-slim AS data

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/build_index /usr/local/bin/build_index
RUN mkdir -p /index \
    && curl -fsSL https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources/references.json.gz -o /tmp/references.json.gz \
    && build_index /tmp/references.json.gz /index/data.bin \
    && rm /tmp/references.json.gz

FROM debian:bookworm-slim

ENV API_ADDR=0.0.0.0:8080 \
    INDEX_PATH=/index/data.bin \
    RUST_LOG=warn

COPY --from=builder /app/target/release/api /usr/local/bin/api
COPY --from=data /index/data.bin /index/data.bin

EXPOSE 8080
CMD ["/usr/local/bin/api"]
