# rinha-2026

Submissão para a [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026) — detecção de fraude em transações de cartão via *k-Nearest Neighbors* (k=5, distância euclidiana) sobre 3 milhões de vetores de referência em 14 dimensões.

A implementação é em Rust, sem banco de dados externo. Todo o pré-processamento do dataset acontece no build da imagem Docker e o índice fica embutido como arquivo binário acessado por `mmap`.

## Estado atual

| Métrica | Valor |
|---|---|
| `final_score` | **5064.35** |
| `p99` HTTP | 8.62 ms |
| `p99_score` | 2064.35 / 3000 |
| `detection_score` | **3000 / 3000** (máximo absoluto) |
| FP / FN / Err | 0 / 0 / 0 |
| commit | `a7d408b` |
| imagem | `ghcr.io/israelaraujo70/rinha-2026:a7d408b` |
| issue oficial | [#1543](https://github.com/zanfranceschi/rinha-de-backend-2026/issues/1543) |

> A primeira preview oficial reportou p99 de **105-112 ms**. Hoje, p99 de 8.62 ms. Corte de ~92% na latência de cauda.

## Topologia

```
cliente
   │  porta 9999
   ▼
┌──────────────┐  0.20 CPU / 20 MB
│   nginx LB   │  HTTP keep-alive, upstream UDS
└─────┬────┬───┘
      │    │  Unix socket (/sockets/api1.sock /sockets/api2.sock)
      ▼    ▼
┌─────────┐ ┌─────────┐  0.40 CPU / 160 MB cada
│  api1   │ │  api2   │  Rust + axum + hyper + tokio current_thread
└────┬────┘ └────┬────┘
     │           │
     └─────┬─────┘
           ▼
   ┌────────────────┐
   │ /index/data.bin│  ~93 MB IVF v3 (mmap, recodificado SoA na RAM)
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
   IVF k-NN (block-8 SoA AVX2 fmadd, adaptive nprobe) ← caminho quente
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
3. Cada cluster armazena seus vetores quantizados em **i16** (escala 10000), além de uma *bounding box* mínima e máxima por dimensão (usada apenas no path opcional de repair).
4. Centróides ficam em ponto flutuante para preservar a precisão da etapa de seleção de probes.

Em runtime, a busca usa **adaptive nprobe**:

1. Calcula distância da query a todos os 4096 centróides; escolhe os **64 clusters mais próximos**.
2. Faz k-NN nos primeiros **8 clusters** (`IVF_NPROBE=8`, ~5800 vetores escaneados).
3. Conta quantos dos 5 vizinhos finais são fraude:
   - Se 0/1 ou 4/5: decisão confiante, retorna direto.
   - Se 2 ou 3 (faixa borderline): escala para os **64 clusters** completos (`IVF_FULL_NPROBE=64`, ~46k vetores), refinando a resposta.

A escalação tier-2 de 24 → 64 clusters foi a mudança que zerou o último false negative que vinha sobrevivendo (+260 pts isolados, 1 linha de env).

### Scan kernel: Block-8 SoA AVX2 fmadd

Ao abrir o índice, o reader recodifica os records do mmap (AoS, 32 B/record) para uma estrutura **SoA in-memory** com blocos de 8 records:

```
block (224 bytes):
  dim 0  de records 0..7   (8× i16 = 16 B)
  dim 1  de records 0..7   (16 B)
  ...
  dim 13 de records 0..7   (16 B)
labels separados, 8 B por bloco
```

Custo de RAM extra: ~1.5 MB para o dataset de 50k records. Cabe folgado nos 160 MB por API.

O kernel de distância processa 8 records simultâneos por iteração:

```
for d in 0..14:
  load 8 i16 do bloco            (_mm_loadu_si128, 16 B)
  widen 8 i16 → 8 i32            (_mm256_cvtepi16_epi32)
  cvt 8 i32 → 8 f32              (_mm256_cvtepi32_ps)
  broadcast query[d] como f32    (_mm256_set1_ps)
  diff = query - record          (_mm256_sub_ps)
  accum += diff*diff             (_mm256_fmadd_ps)
```

Sem `hsum` por record — cada um dos 8 lanes do `__m256` carrega sua própria distância acumulada.

**Threshold pruning de bloco inteiro**: `_mm256_cmp_ps` + `_mm256_movemask_ps` checam se algum dos 8 records bate o `best_dist[K-1]` antes de extrair distâncias e atualizar o top-K. Quando todos os 8 estão piores, o bloco é descartado sem custo extra.

**Padding**: o último bloco de cada cluster (se o cluster não é múltiplo de 8) é preenchido com `i16::MAX` em todas as dimensões. A distância produzida (≥ 1.5e10) está sempre acima de qualquer top-K real, então o threshold pruning filtra naturalmente os slots inválidos sem precisar de length check explícito.

> **Cuidado com SIMD theatre.** O block-8 substituiu um kernel single-record com per-dim early-exit que já era bem perto do ótimo neste workload. O ganho mensurado contra a versão anterior foi de **+1 ponto** (5063 → 5064), dentro da variance do runner. p99 não estava sendo dominado pelo scan IVF — estava sendo dominado por overhead de transport, runtime e CPU throttle.

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
- `target-cpu = "x86-64-v3"` em `.cargo/config.toml` — habilita AVX2/FMA/BMI2 como baseline para todo o codegen, não só nos blocos `target_feature`.
- **Rust toolchain pinado por digest no Dockerfile** (`rust:1.88-bookworm@sha256:af306cfa…`). A tag rolling `:1-bookworm` muda o codegen entre versões e produziu artefatos diferentes em builds aparentemente equivalentes.
- Tokio em `current_thread` runtime (cada réplica tem 0.40 vCPU; work-stealing seria desperdício).
- mimalloc como global allocator.

### Load balancer

- **nginx 1.27-alpine** em modo HTTP, listen 9999, upstream em **Unix Domain Socket**: `unix:/sockets/api1.sock` e `unix:/sockets/api2.sock`.
- `keepalive 256`, `keepalive_requests 100000`, `proxy_buffering off`.
- Volume Docker compartilhado `sockets` mounta `/sockets` em nginx + ambas as APIs.
- 0.20 CPU / 20 MB alocados.
- A troca de TCP loopback por UDS foi **a maior fatia de ganho do projeto**: cortou ~7 ms do p99 no runner oficial. +260 pontos em uma alteração de configuração. axum + tokio aguentam UDS keep-alive sem stress (`axum::serve(UnixListener::bind(...), app)`).

### Caminho de requisição

- Resposta como uma de 6 `&'static [u8]` pré-construídas — sem `ryu`, sem serde, sem alocação.
- `Response::new` + `headers_mut().insert` em vez de `Response::builder` — uma alocação a menos por request.
- Timeout interno de KNN em 3 ms; em caso de timeout, retorna `{"approved": true, "fraud_score": 0.0}` em HTTP 200. Evita o peso 5× de `Err` no `score_det`.

### KNN

- `Index::open` faz **prefault** de todas as páginas do mmap (93 MB) e roda **512 queries de aquecimento** antes de servir. Custo de startup, não de request — esse warmup sozinho cortou o p99 inicial de ~110 ms para a vizinhança de 16 ms.
- Índice recodificado em **SoA block-8** na RAM (~1.5 MB extra), conforme descrito acima.
- Distância via AVX2 `fmadd` processando 8 records por iteração; pruning de bloco inteiro via `movemask_ps`.

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
├── index.rs             # IVF reader, busca k-NN com block-8 SoA AVX2
├── build.rs             # serializador + clustering k-means++
└── bin/
    ├── api.rs           # servidor HTTP /ready + /fraud-score (TCP ou UDS)
    ├── build_index.rs   # CLI: references.json.gz → data.bin
    ├── bench_knn.rs     # micro-bench da camada KNN
    └── compare_scores.rs# compara dois índices contra os mesmos payloads
```

Outros arquivos:

- `Dockerfile` — multi-stage com `rust:1.88-bookworm@sha256:af306cfa…` pinado: builder Rust → data stage que baixa references e roda `build_index` → imagem final com binário + `data.bin`.
- `nginx.conf` — load balancer HTTP com upstream em UDS.
- `docker-compose.yml` — duas réplicas + nginx, volume compartilhado para os sockets, limites de recurso.
- `.cargo/config.toml` — flags de compilação por target.

## Configuração via variáveis de ambiente

| Variável | Default | Descrição |
|---|---|---|
| `API_ADDR` | `0.0.0.0:8080` | Endereço do servidor HTTP TCP (fallback quando `SOCKET_PATH` não está definido). |
| `SOCKET_PATH` | vazio (binário); `/sockets/apiN.sock` no compose | Se definido, a API escuta em Unix socket no path indicado **e ignora `API_ADDR`**. Esta é a configuração ativa em produção. |
| `INDEX_PATH` | `/index/data.bin` | Caminho do índice IVF. |
| `KNN_TIMEOUT_US` | 1000 (binário); **3000 no compose** | Timeout da busca k-NN. |
| `IVF_NPROBE` | 1 (binário); **8 no compose** | Clusters escaneados na fase rápida. |
| `IVF_FULL_NPROBE` | igual a `IVF_NPROBE` (binário); **64 no compose** | Clusters totais quando a query é borderline (2 ou 3 fraudes em K=5). Subiu de 24 para 64 para eliminar 1 FN persistente, +260 pts. |
| `IVF_REPAIR` | `false` | Path opcional de repair pós-tier-2 com bbox prune (off por default). |
| `IVF_CLUSTERS` | 4096 | (build only) Número de clusters do k-means. |
| `IVF_SAMPLE` | 50000 | (build only) Tamanho da amostra para k-means++. |
| `IVF_KMEANS_ITERS` | 25 | (build only) Iterações máximas de Lloyd. |

## Histórico de prévias oficiais

Cada submissão foi enviada via `gh issue create` no repo do Zan; o bot `arinhadebackend` processa e devolve o JSON.

| Issue | Imagem | Mudança principal | p99 | det / FN | final |
|---|---|---|---|---|---|
| #1247 | `:latest` perdido | baseline original | 7.32 | 3000 | **5135.33** *(outlier não-reproduzível)* |
| #1320 | `:354b088` | paired AVX2 + fix overflow | 15.64 | 3000 | 4805.64 |
| #1355 | `:db7bac4` | paired puro (referência verificada) | 16.07 | 3000 | 4794.10 |
| #1453 | `:b3902dd` | monoio + UDS + fases 1-5 (regressão) | 59.53 | 3000 | 4225.29 |
| #1462 | `:b3902dd` | mesma imagem em TCP | 34.26 | 2700 / 3 | 4165.22 |
| #1482 | `:db7bac4` | rollback para baseline | 17.83 | 2819 / 1 | 4568.25 |
| #1496 | `:4a9fc7f` | + parser custom + prefetch + nprobe=64 | 17.49 | 3000 | 4757.10 |
| #1505 | `:d195661` | per-dim early-exit + nprobe=64 | 15.79 | 3000 | 4801.70 |
| #1525 | `:d195661` | + nginx **UDS** (mesma imagem) | **8.64** | 3000 | **5063.43** ✓ |
| #1531 | `:d195661` | UDS + nprobe=6 (regressão) | 9.46 | 3000 | 4843.36 |
| #1543 | `:a7d408b` | block-8 SoA AVX2 fmadd | **8.62** | 3000 | **5064.35** ✓ |

A submissão atual é **`:a7d408b`** com **UDS + `IVF_FULL_NPROBE=64`**. O ganho real entre #1525 e #1543 está dentro da variance do runner (~250 pts entre execuções idênticas), então o block-8 não é provavelmente um win mensurável — mas tampouco regrediu.

## Stack atual

| Componente | Tecnologia |
|---|---|
| Runtime | Tokio (current_thread) |
| HTTP server | axum 0.8 + hyper 1.x |
| Allocator | mimalloc |
| Parsing | serde_json + serde |
| Datas | chrono |
| Memória do índice | memmap2 + SoA block-8 in-RAM |
| Decompressão (build) | flate2 |
| Load balancer | nginx 1.27-alpine (HTTP mode, upstream UDS) |
| Container | Docker / docker-compose |
| Compilação | rustc 1.88 com `target-cpu=x86-64-v3`, LTO fat |

## Lições aprendidas

1. **Detecção e latência são problemas separados.** Detecção saturou cedo com K=4096 + k-means++ + 25 iterações + tier-2 amplo (`IVF_FULL_NPROBE=64`). Latência exigiu mexer no caminho HTTP/runtime e no transport entre proxy e API.

2. **Configuração ambiental frequentemente bate refactor de algoritmo no ROI por hora investida.** A maior fatia de ganho do projeto veio de uma única troca de TCP loopback por UDS no `nginx.conf` + 3 linhas no `docker-compose.yml`. Ganho mensurável: -7 ms no p99, +260 pts.

3. **SIMD só vale se o hot path é mesmo o gargalo.** Refatorei o scan kernel de single-record (per-dim early-exit) para block-8 SoA AVX2 fmadd. Custou 3 horas de código e validação. Ganho real: +1 ponto, dentro da variance. O p99 já não estava sendo dominado pelo scan; profile antes de refatorar.

4. **Tags Docker mutáveis quebram reprodutibilidade.** O #1247 (5135) usou `:latest` que foi sobrescrito; ao tentar voltar, perdemos o artefato vencedor. **Rule of thumb**: submissão sempre por commit-tag (`:<sha>`) ou digest sha256, nunca `:latest`.

5. **Rolling tags de toolchain (`rust:1-bookworm`) também quebram reprodutibilidade.** Versões diferentes do rustc reordenam float math no k-means++ (1 ponto borderline cruza fronteira → 1 FN extra) e mudam codegen do binário (60 KB+ de diferença → 10 ms+ no p99). Pinar por digest é obrigatório.

6. **`docker build --no-cache --pull` é mandatório para qualquer imagem que vai pra submission.** Cache contaminado entre re-tags produziu binários sutilmente diferentes várias vezes.

7. **Variance do runner é ~250 pts entre execuções idênticas.** Mudanças menores que isso são noise. Eu fiz vários submits achando que estava no caminho certo até cair a ficha que estava seguindo ruído. Pra ter sinal, ou o experimento é big swing ou precisa repetição estatística.

8. **monoio 0.2.4 + UDS tem armadilhas.** Três bugs distintos com AF_UNIX: `bind` reclamando de SO_REUSEPORT, task spawn não polled após accept, e segunda `recv` na mesma conn keep-alive nunca completando. axum + tokio aguentam UDS sem stress — escolhi tokio.

9. **`_mm256_maskload_ps` é caro no Haswell** (~10 ciclos vs ~3 do `loadu_ps`). Vetorizar a centroid distance com maskload regrediu p99 de 17.60 → 28.68 ms num teste anterior. Pra vetorizar é mais saudável padding pra alinhar tudo.

10. **`_mm_add_epi32(lo, hi)` antes de zero-extend pode estourar i32 silenciosamente.** A soma horizontal i32×8 → u64 funcionava em 99% dos queries mas leakava 3 FN nos extremos. Sempre zero-extender lane-a-lane antes de qualquer add é o caminho certo.

11. **Parser JSON manual é uma armadilha.** Escrevi um parser zero-alloc que era 4× mais rápido em micro-bench. Em produção, regrediu 296 pontos por bug sutil em timestamps com timezone offset não-Z. Os micro-benchmarks não cobriam o perfil real do dataset.

12. **Ambiente local engana severamente.** p99 local de 3-4 ms não corresponde a p99 oficial de 8-16 ms. O Mac Mini Late 2014 + Docker + 0.40 vCPU/réplica é qualitativamente diferente do host de desenvolvimento. Toda otimização tem que ser confirmada com prévia oficial.

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
INDEX_PATH=/tmp/data.bin IVF_NPROBE=8 IVF_FULL_NPROBE=64 \
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

# 2. Push image
docker push ghcr.io/israelaraujo70/rinha-2026:<commit-short>

# 3. Atualizar submission/docker-compose.yml apontando pra :<commit-short>
#    (na branch `submission`, manter compose com SOCKET_PATH e volume `sockets`)

# 4. Abrir issue
gh issue create --repo zanfranceschi/rinha-de-backend-2026 \
  --title "rinha/test israelaraujo70-rust" \
  --body "rinha/test israelaraujo70-rust"

# 5. Acompanhar
gh issue view <issue-number> --repo zanfranceschi/rinha-de-backend-2026 \
  --json comments --jq '.comments[].body'
```

> O bot do Zan avalia o estado **atual** do branch `submission` no momento da execução, não o snapshot do momento em que o issue foi criado. Por isso é preciso garantir que `submission/docker-compose.yml` aponta pro digest certo *antes* de criar o issue.

Detalhes commit a commit em `git log`; cada experimento saiu como commit separado com motivação e número da issue oficial.
