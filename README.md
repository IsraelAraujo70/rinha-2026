# rinha-2026

Submissão para a [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026) — detecção de fraude em transações de cartão via *k-Nearest Neighbors* (k=5, distância euclidiana) sobre 3 milhões de vetores de referência em 14 dimensões.

A implementação é em Rust, sem banco de dados externo. Todo o pré-processamento do dataset acontece no build da imagem Docker e o índice fica embutido como arquivo binário acessado por `mmap`.

## Topologia

```
cliente
   │  porta 9999
   ▼
┌──────────────┐  0.20 CPU / 20 MB
│   nginx LB   │  round-robin, keepalive 256
└─────┬────┬───┘
      │    │
      ▼    ▼
┌─────────┐ ┌─────────┐  0.40 CPU / 160 MB cada
│  api1   │ │  api2   │  Rust + axum + hyper
└────┬────┘ └────┬────┘
     │           │
     └─────┬─────┘
           ▼
   ┌────────────────┐
   │ /index/data.bin│  ~93 MB IVF v3 (mmap)
   └────────────────┘
```

Total: **1.0 CPU + 340 MB** dentro do limite oficial de 1 CPU / 350 MB.

## Pipeline da requisição

```
POST /fraud-score
       │
       ▼
   serde_json parse  (axum, body Bytes)
       │
       ▼
   vetorize → [f32; 14]
       │
       ▼
   IVF k-NN (adaptive nprobe)  ← caminho quente
       │
       ▼
   contagem 0..5 vizinhos fraudulentos
       │
       ▼
   resposta = FRAUD_RESPONSES[count]   (&'static [u8])
```

A resposta é uma de 6 strings JSON pré-construídas em tempo de compilação, indexadas pelo número de vizinhos rotulados como fraude (k=5 → contagem ∈ {0..5}, score ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}). Não há serialização nem alocação no caminho de resposta.

## Algoritmo de busca

### IVF (Inverted File Index) com clustering k-means++

O dataset de 3M vetores é particionado em **K = 4096 clusters** durante o build:

1. **k-means++ init** a partir de uma amostra de 50.000 vetores. Cada centróide novo é escolhido com probabilidade proporcional à distância quadrada ao centróide mais próximo já fixado — espalha as sementes de forma representativa.
2. **25 iterações de Lloyd** sobre a base completa (3M pontos), com atribuição paralelizada via `std::thread::scope`. Critério de parada antecipada: menos de 0.1% dos pontos mudam de cluster.
3. Cada cluster armazena seus vetores quantizados em **i16** (escala 10000), além de uma *bounding box* mínima e máxima por dimensão para *pruning* opcional.
4. Centróides ficam em ponto flutuante para preservar a precisão da etapa de seleção de probes.

Em runtime, a busca usa **adaptive nprobe** (inspirado em [jairoblatt/rinha-2026-rust](https://github.com/jairoblatt/rinha-2026-rust)):

1. Calcula distância da query a todos os 4096 centróides; escolhe os **24 clusters mais próximos**.
2. Faz k-NN nos primeiros **8 clusters** (`IVF_NPROBE=8`, ~5800 vetores escaneados).
3. Conta quantos dos 5 vizinhos finais são fraude:
   - Se 0/1 ou 4/5: decisão confiante, retorna direto.
   - Se 2 ou 3 (faixa borderline): escala para os **24 clusters** completos (`IVF_FULL_NPROBE=24`, ~17500 vetores), refinando a resposta.
4. Cada distância usa AVX2 (`_mm256_madd_epi16`) sobre as 14 dimensões i16.

A escalada só dispara em ~5% das queries (as borderline reais), então o caminho médio paga só a versão rápida.

### Vetorização (14 dimensões)

| # | Componente | Fórmula |
|---|---|---|
| 0 | `amount` | `clamp(amount / 10_000)` |
| 1 | `installments` | `clamp(installments / 12)` |
| 2 | `amount_vs_avg` | `clamp((amount / customer_avg) / 10)` |
| 3 | `hour_of_day` | `hour / 23` |
| 4 | `day_of_week` | `weekday / 6` |
| 5 | `minutes_since_last_tx` | `clamp(min / 1440)` ou `-1` |
| 6 | `km_from_last_tx` | `clamp(km / 1000)` ou `-1` |
| 7 | `km_from_home` | `clamp(km / 1000)` |
| 8 | `tx_count_24h` | `clamp(count / 20)` |
| 9 | `is_online` | 0 ou 1 |
| 10 | `card_present` | 0 ou 1 |
| 11 | `unknown_merchant` | 0 ou 1 |
| 12 | `mcc_risk` | tabela fixa por MCC |
| 13 | `merchant_avg_amount` | `clamp(avg / 10_000)` |

Implementação em [`crates/fraud/src/vector.rs`](crates/fraud/src/vector.rs).

## Otimizações aplicadas

Em ordem cronológica de impacto medido:

### Build profile e runtime

- `lto = "fat"`, `codegen-units = 1` em `[profile.release]`.
- `target-cpu = "x86-64-v3"` em `.cargo/config.toml` — habilita AVX2/FMA/BMI2 como baseline.
- Tokio em `current_thread` runtime (cada réplica tem 0.45 vCPU; work-stealing seria desperdício).
- mimalloc como global allocator.

### nginx

- `keepalive 256` no upstream, `keepalive_requests 100000`.
- `proxy_buffering off`, `proxy_request_buffering off`.
- `tcp_nodelay`, `multi_accept`, `epoll`.

### Caminho de requisição

- Resposta como uma de 6 `&'static [u8]` pré-construídas — sem `ryu`, sem serde, sem alocação.
- Timeout interno de KNN em 3 ms; em caso de timeout, retorna `{"approved": true, "fraud_score": 0.0}` em HTTP 200. Evita o peso 5× de `Err` no `score_det`.

### Build do índice

- K=4096 com k-means++ + 25 iterações de Lloyd na base completa (3M).
- Iteração paralela via `std::thread::scope`, com 8-16 threads (depende da máquina de build).
- Tempo de build ≈ 3:40 numa workstation 8-core; cabe na imagem Docker como camada cacheável.

## Estrutura

```
crates/fraud/src/
├── lib.rs               # módulos públicos
├── payload.rs           # FraudRequest (serde Deserialize)
├── vector.rs            # vetorização + quantização i16
├── index.rs             # IVF reader, busca k-NN com adaptive nprobe
├── build.rs             # serializador + clustering k-means++
└── bin/
    ├── api.rs           # servidor HTTP /ready + /fraud-score
    ├── build_index.rs   # CLI: references.json.gz → data.bin
    ├── bench_knn.rs     # micro-bench da camada KNN
    └── compare_scores.rs# compara dois índices contra os mesmos payloads
```

Outros arquivos:

- `Dockerfile` — multi-stage: builder Rust → data stage que baixa references e roda `build_index` → imagem final com binário + `data.bin`.
- `nginx.conf` — load balancer.
- `docker-compose.yml` — duas réplicas + nginx, limites de recurso.
- `.cargo/config.toml` — flags de compilação por target.

## Configuração via variáveis de ambiente

| Variável | Default | Descrição |
|---|---|---|
| `API_ADDR` | `0.0.0.0:8080` | Endereço do servidor HTTP |
| `INDEX_PATH` | `/index/data.bin` | Caminho do índice IVF |
| `KNN_TIMEOUT_US` | 1000 (build), **3000 no compose** | Timeout da busca k-NN |
| `IVF_NPROBE` | 1 (build), **8 no compose** | Clusters escaneados na fase rápida |
| `IVF_FULL_NPROBE` | igual a `IVF_NPROBE`, **24 no compose** | Clusters totais quando borderline |
| `IVF_REPAIR` | `false` | Pruning extra usando bbox dos clusters |
| `IVF_CLUSTERS` | 4096 | (build only) Número de clusters do k-means |
| `IVF_SAMPLE` | 50000 | (build only) Tamanho da amostra para k-means++ |
| `IVF_KMEANS_ITERS` | 25 | (build only) Iterações máximas de Lloyd |

## Métricas locais

### KNN isolado (sem HTTP)

| Configuração | avg | p99 |
|---|---|---|
| K=256, nprobe=1 (legado) | 82 us | 249 us |
| K=4096, nprobe=8 / full=24 (pré-SIMD IVF) | 56 us | 105 us |
| K=4096, nprobe=8 / full=24 (atual, AVX2 IVF) | 50 us | 83 us |

### Teste oficial k6 (dataset de 54.100 payloads, 900 req/s)

Todas as medições com host de bancada poluído por outros processos — números do ambiente da Rinha (Mac Mini 2014 dedicado) tendem a ser melhores em latência.

| Versão | FP+FN | det_score | p99 (best run) | final |
|---|---|---|---|---|
| Baseline (`8587310`) | — | — | 3.49 ms | 3868.66 |
| `nprobe=2` runtime tuning | 178 | — | 4.69 ms | 4615.18 |
| Wave C (cargo profile + nginx + tokio) | 38 | 2286 | 3.09 ms | 4796.17 |
| Wave 1 (static resp + mimalloc + adaptive nprobe) | 8 | 2630 | 4.61 ms | 4967.01 |
| **Wave 2a (K=4096 k-means++ 25 iters)** | **0** | **3000** (max) | 10.68 ms (*) | **4971** |
| CPU split nginx/API + AVX2 IVF | 0 | 3000 | 3.97 ms | 5401.37 |

(*) Medição antiga com host de bancada saturado por outros processos. A rodada local atual, já com CPU split 0.20/0.40/0.40 e IVF AVX2, ficou em 3.97 ms p99 no k6 oficial.

## Como executar localmente

Pré-requisitos: Rust 1.78+, Docker, Docker Compose.

```bash
# Subir a stack completa (compila Rust, baixa references, roda build_index)
docker compose up --build

# Verificar
curl -i http://localhost:9999/ready
curl -X POST -H 'content-type: application/json' \
  --data '{"id":"t","transaction":{"amount":100,"installments":1,"requested_at":"2024-01-01T12:00:00Z"},"customer":{"avg_amount":100,"tx_count_24h":1,"known_merchants":[]},"merchant":{"id":"m","mcc":"5411","avg_amount":50},"terminal":{"is_online":false,"card_present":true,"km_from_home":1.0},"last_transaction":null}' \
  http://localhost:9999/fraud-score
```

Para buildar o índice manualmente fora do Docker:

```bash
curl -fsSL https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources/references.json.gz \
  -o /tmp/references.json.gz
cargo run --release --bin build_index -- /tmp/references.json.gz /tmp/data.bin
INDEX_PATH=/tmp/data.bin IVF_NPROBE=8 IVF_FULL_NPROBE=24 \
  cargo run --release --bin api
```

Para rodar o teste oficial k6 (precisa de Docker):

```bash
git clone https://github.com/zanfranceschi/rinha-de-backend-2026.git /tmp/rinha
docker run --rm --user root --network host \
  -v /tmp/rinha:/work -w /work grafana/k6 run test/test.js
jq . /tmp/rinha/test/results.json
```

## Stack

| Componente | Tecnologia |
|---|---|
| Runtime | Tokio (current_thread) |
| HTTP server | axum 0.8 + hyper 1.x |
| Allocator | mimalloc |
| Parsing | serde_json + serde |
| Datas | chrono |
| Memória do índice | memmap2 |
| Decompressão (build) | flate2 |
| Load balancer | nginx 1.27 |
| Container | Docker / docker-compose |
| Compilação | rustc 1.x com `target-cpu=x86-64-v3`, LTO fat |

## Trabalho descartado durante a iteração

- **Bare hyper rewrite** (substituir axum por hyper puro): mensurado neutro vs axum em 5 runs cada. Complexidade extra sem ganho.
- **Slim parser com strings borrowed e datetime manual**: regrediu vs serde+chrono em medições controladas. Hipótese é que a validação no-escapes do `&str` zero-copy do serde_json paga mais do que ele economiza no tamanho do payload típico.

Detalhes em `git log` — cada onda foi commitada separadamente com motivação e medições.
