# rinha-2026

Tentativa de resolver a [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026).

## Estrutura

- `crates/fraud/src/bin/api.rs`: servidor HTTP com `GET /ready` e `POST /fraud-score`.
- `crates/fraud/src/bin/build_index.rs`: converte `references.json.gz` para `/index/data.bin`.
- `crates/fraud/src/vector.rs`: vetorização das 14 dimensões da regra oficial.
- `crates/fraud/src/index.rs`: leitura mmap do índice e KNN brute-force quantizado.
- `nginx.conf` e `docker-compose.yml`: load balancer na porta `9999` e duas réplicas da API.

## Desenvolvimento

```bash
cargo test
cargo run --release --bin api
```

Sem `/index/data.bin`, a API sobe em modo fallback e aprova com `fraud_score: 0.0`.
Para gerar o índice local:

```bash
curl -fsSL https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources/references.json.gz -o /tmp/references.json.gz
cargo run --release --bin build_index -- /tmp/references.json.gz /tmp/data.bin
INDEX_PATH=/tmp/data.bin cargo run --release --bin api
```

Para subir a topologia completa:

```bash
docker compose up --build
curl -i http://localhost:9999/ready
```

## Baseline atual

O indexador aceita `references.json.gz` e também amostras `.json` sem gzip:

```bash
cargo run --release --bin build_index -- /tmp/references.json.gz /tmp/data.bin
cargo run --release --bin bench_knn -- /tmp/data.bin /tmp/example-payloads.json 1000
```

Medições locais com o dataset oficial completo:

- `references.json.gz`: 48 MB.
- `data.bin` v1, 15 bytes/registro: 43 MB para 3.000.000 registros.
- `data.bin` v2, 16 bytes/registro: 46 MB para 3.000.000 registros.
- `build_index` v2: 9,18s.
- KNN brute-force escalar v1: `avg=6608us p50=6546us p95=6930us p99=7701us`.
- KNN v2 alinhado + AVX2: `avg=2979us p50=2931us p95=3249us p99=3805us`.

Essa versão ainda é exata, com checksum preservado no benchmark, mas segue longe do alvo de p99 sub-ms. A próxima etapa é avaliar índice aproximado ou particionamento que reduza candidatos.
