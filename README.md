# rinha-2026

Tentativa de resolver a [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026).

## Estrutura

- `crates/fraud/src/bin/api.rs`: servidor HTTP com `GET /ready` e `POST /fraud-score`.
- `crates/fraud/src/bin/build_index.rs`: converte `references.json.gz` para `/index/data.bin`.
- `crates/fraud/src/bin/compare_scores.rs`: compara scores de dois índices para medir divergência.
- `crates/fraud/src/vector.rs`: vetorização das 14 dimensões da regra oficial.
- `crates/fraud/src/index.rs`: leitura mmap do índice e KNN exato v2 ou IVF v3.
- `nginx.conf` e `docker-compose.yml`: load balancer na porta `9999` e duas réplicas da API.

## Desenvolvimento

```bash
cargo test
```

Para gerar o índice local e subir a API:

```bash
curl -fsSL https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources/references.json.gz -o /tmp/references.json.gz
IVF_CLUSTERS=256 cargo run --release --bin build_index -- /tmp/references.json.gz /tmp/data.bin
IVF_NPROBE=1 INDEX_PATH=/tmp/data.bin cargo run --release --bin api
```

A API falha no startup se o índice não existir ou estiver inválido. Durante uma requisição, payload inválido, pânico no scoring ou timeout interno retornam fallback aprovado com HTTP 200 para evitar penalidade de erro HTTP.

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
- `data.bin` v3 IVF, `IVF_CLUSTERS=256`: 92 MB para 3.000.000 registros.
- `build_index` v3 IVF, `IVF_SAMPLE=32768 IVF_KMEANS_ITERS=6`: `elapsed=0:10.92 maxrss=201308KB`.
- KNN v3 IVF, `IVF_NPROBE=1`: `avg=82us p50=74us p95=179us p99=249us checksum=1940.000`.
- KNN v3 IVF, `IVF_NPROBE=4`: `avg=300us p50=285us p95=658us p99=753us checksum=1940.000`.
- Comparação v2 exato vs v3 IVF nos 50 payloads locais: `score_mismatches=1 decision_mismatches=0 max_delta=0.200`.

O `build_index` padrão gera IVF v3. Para gerar o brute-force exato v2 para comparação:

```bash
INDEX_KIND=exact cargo run --release --bin build_index -- /tmp/references.json.gz /tmp/data-v2.bin
cargo run --release --bin compare_scores -- /tmp/data-v2.bin /tmp/data.bin /tmp/example-payloads.json
```

O IVF reduz candidatos antes do KNN e coloca o caminho crítico abaixo de 1ms no benchmark local. O trade-off passa a ser `IVF_NPROBE`: valores menores são mais rápidos e mais aproximados; valores maiores escaneiam mais clusters e reduzem o risco de divergência.
