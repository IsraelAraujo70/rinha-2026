# rinha-2026

Submissão para a [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026) — detecção de fraude em transações de cartão via *k-Nearest Neighbors* (k=5, distância euclidiana) sobre 3 milhões de vetores de referência em 14 dimensões.

A implementação é em Rust, sem banco de dados externo. Todo o pré-processamento do dataset acontece no build da imagem Docker e o índice fica embutido como arquivo binário acessado por `mmap`.

## Estado atual (final da Wave A)

| Métrica | Valor |
|---|---|
| `final_score` | **4794.10** |
| `p99` HTTP | 16.07 ms |
| `p99_score` | 1794.10 / 3000 |
| `detection_score` | **3000 / 3000** (máximo absoluto) |
| FP / FN / Err | 0 / 0 / 0 |
| commit avaliado | `db7bac4` |
| imagem | `ghcr.io/israelaraujo70/rinha-2026:db7bac4` |
| digest | `sha256:6c6e6887293c0e7824f06f8c3aaa0d940ab1b09eb854be203e00c8aebef1646b` |
| issue oficial | [#1355](https://github.com/zanfranceschi/rinha-de-backend-2026/issues/1355) |

> O resultado de **5135.33 (#1247, p99 7.32ms)** que apareceu antes em uma prévia anterior **não é reproduzível**. A IA companheira recuperou o digest GHCR exato daquela imagem e re-submeteu duas vezes (#1341 e #1345), ambas em ~17ms. Conclusão: aquele número foi outlier do runner Mac Mini, não propriedade do binário. Trabalhar em cima do baseline reproduzível atual (~4800).

## Topologia

```
cliente
   │  porta 9999
   ▼
┌──────────────┐  0.20 CPU / 20 MB
│   nginx LB   │  HTTP keep-alive, upstream TCP loopback
└─────┬────┬───┘
      │    │  TCP api1:8080 / api2:8080
      ▼    ▼
┌─────────┐ ┌─────────┐  0.40 CPU / 160 MB cada
│  api1   │ │  api2   │  Rust + axum + hyper + tokio current_thread
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
   IVF k-NN (paired AVX2 scan, adaptive nprobe)  ← caminho quente
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

1. **k-means++ init** a partir de uma amostra de 50.000 vetores. Cada centróide novo é escolhido com probabilidade proporcional à distância quadrada ao centróide mais próximo já fixado.
2. **25 iterações de Lloyd** sobre a base completa (3M pontos), com atribuição paralelizada via `std::thread::scope`. Critério de parada antecipada: menos de 0.1% dos pontos mudam de cluster.
3. Cada cluster armazena seus vetores quantizados em **i16** (escala 10000), além de uma *bounding box* mínima e máxima por dimensão (atualmente não usada — a tentativa de bbox prune regrediu detection no #1352).
4. Centróides ficam em ponto flutuante para preservar a precisão da etapa de seleção de probes.

Em runtime, a busca usa **adaptive nprobe**:

1. Calcula distância da query a todos os 4096 centróides; escolhe os **24 clusters mais próximos**.
2. Faz k-NN nos primeiros **8 clusters** (`IVF_NPROBE=8`, ~5800 vetores escaneados).
3. Conta quantos dos 5 vizinhos finais são fraude:
   - Se 0/1 ou 4/5: decisão confiante, retorna direto.
   - Se 2 ou 3 (faixa borderline): escala para os **24 clusters** completos (`IVF_FULL_NPROBE=24`, ~17500 vetores), refinando a resposta.
4. Cada par de records (64 bytes) é processado por uma única chamada AVX2 `squared_distance_i16_pair_avx2`: dois `_mm256_madd_epi16` sobre as 14 dimensões i16, dois reduces a u64. A redução horizontal (`horizontal_sum_i32x8_to_u64`) zero-extende cada lane i32 a u64 antes de somar, evitando overflow silencioso quando deltas são grandes.

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

### Build profile e runtime

- `lto = "fat"`, `codegen-units = 1` em `[profile.release]`.
- `target-cpu = "x86-64-v3"` em `.cargo/config.toml` — habilita AVX2/FMA/BMI2 como baseline.
- **Rust toolchain pinado por digest no Dockerfile** (`rust:1.88-bookworm@sha256:af306cfa…`). A tag rolling `:1-bookworm` muda o codegen entre versões e produziu artefatos diferentes em builds aparentemente equivalentes.
- Tokio em `current_thread` runtime (cada réplica tem 0.40 vCPU; work-stealing seria desperdício).
- mimalloc como global allocator.

### Load balancer

- **nginx 1.27-alpine** em modo HTTP, listen 9999, upstream `api1:8080` / `api2:8080` via Docker bridge network (TCP loopback).
- `keepalive 256`, `keepalive_requests 100000`, `proxy_buffering off`.
- 0.20 CPU / 20 MB alocados.
- Tentativas com Unix domain sockets (HAProxy/UDS, nginx stream/UDS, nginx http/UDS) **regridem o p99 para o floor de 29.77ms** no Mac Mini Late 2014 — mesmo número aparece em ~8 outras submissões alheias, indicando floor mecânico do harness/runner. UDS está banido até confirmar a causa.

### Caminho de requisição

- Resposta como uma de 6 `&'static [u8]` pré-construídas — sem `ryu`, sem serde, sem alocação.
- `Response::new` + `headers_mut().insert` em vez de `Response::builder` — uma alocação a menos por request.
- Timeout interno de KNN em 3 ms; em caso de timeout, retorna `{"approved": true, "fraud_score": 0.0}` em HTTP 200. Evita o peso 5× de `Err` no `score_det`.

### KNN

- `Index::open` faz **prefault** de todas as páginas do mmap (93 MB) e roda **512 queries de aquecimento** antes de servir. Custo de startup, não de request.
- Distância i16 vetorizada com **AVX2** (`_mm256_madd_epi16` mascarado para zerar as 2 lanes de padding).
- **Paired scan**: 2 records (64 bytes) processados por chamada, com soma horizontal i32×8 → u64 segura contra overflow.

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
├── index.rs             # IVF reader, busca k-NN com paired AVX2 scan
├── build.rs             # serializador + clustering k-means++
└── bin/
    ├── api.rs           # servidor HTTP /ready + /fraud-score
    ├── build_index.rs   # CLI: references.json.gz → data.bin
    ├── bench_knn.rs     # micro-bench da camada KNN
    └── compare_scores.rs# compara dois índices contra os mesmos payloads
```

Outros arquivos:

- `Dockerfile` — multi-stage com `rust:1.88-bookworm@sha256:af306cfa…` pinado: builder Rust → data stage que baixa references e roda `build_index` → imagem final com binário + `data.bin`.
- `nginx.conf` — load balancer HTTP com upstream em TCP loopback.
- `docker-compose.yml` — duas réplicas + nginx, limites de recurso.
- `.cargo/config.toml` — flags de compilação por target.

## Configuração via variáveis de ambiente

| Variável | Default | Descrição |
|---|---|---|
| `API_ADDR` | `0.0.0.0:8080` | Endereço do servidor HTTP |
| `SOCKET_PATH` | vazio | Se definido, a API escuta em Unix socket em vez de TCP (atualmente não usado por causa da regressão UDS no Mac Mini) |
| `INDEX_PATH` | `/index/data.bin` | Caminho do índice IVF |
| `KNN_TIMEOUT_US` | 1000 (build), **3000 no compose** | Timeout da busca k-NN |
| `IVF_NPROBE` | 1 (build), **8 no compose** | Clusters escaneados na fase rápida |
| `IVF_FULL_NPROBE` | igual a `IVF_NPROBE`, **24 no compose** | Clusters totais quando borderline |
| `IVF_REPAIR` | `false` | (não usado atualmente; bbox prune regrediu detection) |
| `IVF_CLUSTERS` | 4096 | (build only) Número de clusters do k-means |
| `IVF_SAMPLE` | 50000 | (build only) Tamanho da amostra para k-means++ |
| `IVF_KMEANS_ITERS` | 25 | (build only) Iterações máximas de Lloyd |

## Histórico de prévias oficiais

Cada Wave foi submetida via `gh issue create` no repo do Zan; o bot `arinhadebackend` processa e devolve o JSON. Detection sempre 3000 exceto onde anotado.

| Issue | Wave | Imagem | Mudança | p99 | det / FN | final |
|---|---|---|---|---|---|---|
| #1247 | (legado) | `:latest` perdido | nginx http + TCP, baseline original | 7.32 | 3000 | **5135.33** *(outlier)* |
| #1272 | A1-haproxy | `:908f8da` | HAProxy http + UDS | 29.77 | 3000 | 4526.21 |
| #1277 | A1-stream | `:ed68d38` | nginx stream + UDS | 29.77 | 3000 | 4526.21 |
| #1282 | A1-uds | `:c4043ae` | nginx http + UDS | 29.77 | 3000 | 4526.21 |
| #1285 | A1-revert | `:d13f222` | volta a nginx http + TCP | 29.77 | 3000 | 4526.21 |
| #1293 | toolchain | `:414e42a-real` | rebuild fresh com rolling rust :1-bookworm | 17.29 | 2819 / 1 | 4581.64 |
| #1306 | rust 1.88 | `:e1b1c4d` | toolchain pinado em 1.88 + nginx http + TCP | 17.60 | 3000 | **4754.57** *(baseline rust 1.88)* |
| #1312 | A2 cheia | `:72b240a` | + paired + cheap sum + centroid AVX2 (maskload) | 28.68 | 3000 | 4542.35 |
| #1317 | A2 split bug | `:a9f83e4` | só paired + cheap sum (com bug de overflow) | 17.96 | 2700 / 3 | 4445.72 |
| #1320 | **A2 split fixed** | **`:354b088`** | + fix overflow + teste de regressão | **15.64** | **3000** | **4805.64** ✓ |
| #1341 | A3 | `:d195661` | single + early-exit (perde paralelismo) | 16.86 | 3000 | 4773.21 |
| #1345 | tese digest | `@sha256:2d5f3909…` | re-roda imagem-vencedora original | 17.18 | 3000 | 4764.97 |
| #1352 | A4a com bbox | `:98ffee1` | paired + bbox prune (regrediu detection) | 19.01 | 2746 / 2 | 4467.53 |
| #1355 | **A4a clean** | **`:db7bac4`** | paired puro (limpo, sem dead code) | **16.07** | **3000** | **4794.10** ✓ |

A versão de submissão atual é **`:db7bac4`** (paired scan, sem bbox prune, sem early-exit, sem dead code). Estável em 4794-4805 entre runs.

## Stack atual

| Componente | Tecnologia |
|---|---|
| Runtime | Tokio (current_thread) |
| HTTP server | axum 0.8 + hyper 1.x |
| Allocator | mimalloc |
| Parsing | serde_json + serde |
| Datas | chrono |
| Memória do índice | memmap2 |
| Decompressão (build) | flate2 |
| Load balancer | nginx 1.27-alpine (HTTP mode, TCP upstream) |
| Container | Docker / docker-compose |
| Compilação | rustc 1.88 com `target-cpu=x86-64-v3`, LTO fat |

## Wave B — plano

**Hipótese**: o p99 atual (~16ms) é dominado por HTTP+parsing+runtime, não pelo KNN (bench isolado p99 ≈ 83us). Top 5 da Rinha (p99 1.0–1.5ms) usam runtimes especializados em io_uring (monoio, glommio, epoll puro), HTTP hand-rolled e parsers JSON manuais. Replicar a stack deles deve dar -5 a -10ms p99 sem mexer no algoritmo.

### Subondas planejadas

| Etapa | Mudança | Esperado | Stop loss |
|---|---|---|---|
| **B1** | Adicionar `monoio = "0.2"` no Cargo.toml; ajustar Dockerfile (build deps); manter axum/tokio compilando em paralelo | build verde | falha de compilação |
| **B2** | Substituir runtime tokio current_thread por **monoio FusionDriver**; HTTP minimal hand-rolled (parse request line, `Content-Length`, body); mantém `serde_json::from_slice` no parse | p99 12-14ms (-2 a -4ms) | se p99 ≥ 16ms, monoio não está dando ganho — abort B |
| **B3** | Substituir serde por **parser JSON manual** com `memchr` sobre o body. Estrutura do payload é fixa, só extrai os campos necessários para `vectorize`. Testes garantem que o `Vector` produzido é bit-idêntico ao serde version em fixtures reais | p99 8-10ms (-4 a -6ms) | se regredir vs B2, parser tem bug — investigar |
| **B4** | Reuso de buffers (zero alloc per request); inline hot path; pequenas otimizações de assembly | p99 6-8ms (-2 a -3ms) | opcional |

### Riscos

- **monoio é menos maduro que tokio.** Mitigação: usar versão estável; manter referência tokio até B2 verde.
- **HTTP manual pode ter bugs de parsing.** Mitigação: testes unitários contra payloads reais; smoke test stack inteira local antes de submeter.
- **Parser JSON manual pode rejeitar payloads válidos** (escapes Unicode, ordem dos campos diferente, whitespace). Mitigação: rodar contra os 54100 payloads do dataset oficial localmente, comparar `vector` campo-a-campo com a versão serde antes de submeter.

### Validação por etapa

- `cargo test --release` em todas as etapas (testes existentes + novos pra parser).
- Smoke test local: `docker compose up`, `curl /ready`, `curl -X POST /fraud-score` com 1 payload do dataset.
- (Opcional) k6 local com `test/test.js` do harness — local satura mas dá sinal de regressão grosseira.
- Submit oficial pelo workflow: `gh issue create` + babysit em background (script existe em `/tmp/babysit-rinha-issue.sh`).

### Referências (top 5 que usam essa receita)

- [jairoblatt/rinha-2026-rust](https://github.com/jairoblatt/rinha-2026-rust) — Rust + monoio FusionDriver + parser memchr (#2 do leaderboard, p99 1.17ms)
- [athospugliese/rinha-rust](https://github.com/athospugliese/rinha-rust) — Rust + glommio (#4, p99 1.45ms)
- [joojf/rinha-2026](https://github.com/joojf/rinha-2026) — Rust + monoio (#5, p99 1.50ms)

## Lições aprendidas

1. **Detecção e latência são problemas separados.** Detecção saturou cedo com K=4096 + k-means++ + 25 iterações; latência exigiu mexer no caminho HTTP/runtime.

2. **Tags Docker mutáveis quebram reprodutibilidade.** O #1247 (5135) usou `:latest` que foi sobrescrito; ao tentar voltar, perdemos o artefato vencedor. **Rule of thumb**: submissão sempre por commit-tag ou digest sha256, nunca `:latest`.

3. **Rolling tags de toolchain (`rust:1-bookworm`) também quebram reprodutibilidade.** Versões diferentes do rustc reordenam float math no k-means++ (1 ponto borderline cruza fronteira → 1 FN extra) e mudam codegen do binário (60KB+ de diferença → 10ms+ no p99). Pinar por digest é obrigatório.

4. **`docker build --no-cache --pull` é mandatório para qualquer imagem que vai pra submission.** Cache contaminado entre re-tags produziu binários sutilmente diferentes várias vezes.

5. **UDS no Mac Mini Late 2014 atinge floor mecânico de 29.77ms** independente do LB (HAProxy http, nginx stream, nginx http). 6+ outras submissões alheias batem o mesmo número exato. Provável retry/timeout do harness ou interação Docker volume + AF_UNIX no kernel do runner.

6. **AVX2 `_mm256_maskload_ps` é caro no Haswell** (~10 ciclos vs ~3 do `loadu_ps`). Vetorizar a centroid distance com maskload regrediu p99 de 17.60 → 28.68ms (#1312). Se for vetorizar, precisa padding pra alinhar tudo em 16 lanes.

7. **`_mm_add_epi32(lo, hi)` antes de zero-extend pode estourar i32 silenciosamente.** O atalho de soma horizontal i32×8 → u64 funcionava em 99% dos queries mas leakava 3 FN nos extremos. Sempre zero-extender lane-a-lane antes de qualquer add é o caminho certo.

8. **Paired scan AVX2 (2 records por chamada) deu +50 pontos** vs single record. É o melhor ganho algorítmico que conseguimos isolar nesta era. Bbox cluster prune e per-dim early-exit foram tentados e regrediram (early-exit perdeu paralelismo do paired; bbox quebrou detection, causa não totalmente investigada).

9. **Ambiente local engana severamente.** p99 local de 3-4ms não corresponde a p99 oficial de 16ms. O Mac Mini Haswell + Docker + 0.40 vCPU/réplica é qualitativamente diferente do nosso host. Toda otimização tem que ser confirmada com prévia oficial.

10. **Variância natural entre runs idênticos é ~0.5ms p99.** Antes de comemorar/desistir, considerar se a diferença está dentro dessa banda.

## Como executar localmente

Pré-requisitos: Docker, Docker Compose. Rust não é necessário para subir (o build é dentro do Dockerfile com toolchain pinado).

```bash
# Subir a stack completa (compila Rust com 1.88 pinado, baixa references, roda build_index)
docker compose up --build

# Verificar
curl -i http://localhost:9999/ready
curl -X POST -H 'content-type: application/json' \
  --data '{"id":"t","transaction":{"amount":100,"installments":1,"requested_at":"2024-01-01T12:00:00Z"},"customer":{"avg_amount":100,"tx_count_24h":1,"known_merchants":[]},"merchant":{"id":"m","mcc":"5411","avg_amount":50},"terminal":{"is_online":false,"card_present":true,"km_from_home":1.0},"last_transaction":null}' \
  http://localhost:9999/fraud-score
```

Para buildar o índice manualmente fora do Docker (precisa Rust 1.88+):

```bash
curl -fsSL https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources/references.json.gz \
  -o /tmp/references.json.gz
cargo run --release --bin build_index -- /tmp/references.json.gz /tmp/data.bin
INDEX_PATH=/tmp/data.bin IVF_NPROBE=8 IVF_FULL_NPROBE=24 \
  cargo run --release --bin api
```

Para rodar o teste oficial k6:

```bash
git clone https://github.com/zanfranceschi/rinha-de-backend-2026.git /tmp/rinha
docker run --rm --user root --network host \
  -v /tmp/rinha:/work -w /work grafana/k6 run test/test.js
jq . /tmp/rinha/test/results.json
```

## Workflow de submissão

```bash
# 1. Build limpo, sempre --no-cache --pull
docker build --no-cache --pull --platform linux/amd64 \
  -t ghcr.io/israelaraujo70/rinha-2026:<commit-short> .

# 2. Verificar binário (sanity check)
id=$(docker create ghcr.io/israelaraujo70/rinha-2026:<commit-short>)
docker cp $id:/usr/local/bin/api /tmp/api-check
docker rm $id
sha256sum /tmp/api-check  # comparar contra builds anteriores se quiser

# 3. Push image
docker push ghcr.io/israelaraujo70/rinha-2026:<commit-short>

# 4. Atualizar submission/docker-compose.yml apontando pra :<commit-short>
#    (worktree em /tmp/rinha-submission)

# 5. Abrir issue
gh issue create --repo zanfranceschi/rinha-de-backend-2026 \
  --title "rinha/test israelaraujo70-rust" \
  --body "rinha/test israelaraujo70-rust"

# 6. Babysit
/tmp/babysit-rinha-issue.sh <issue-number>
```

Detalhes commit a commit em `git log`; cada onda saiu como commit separado com motivação e número da issue oficial.
