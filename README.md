# rinha-2026

Submissão para a [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026) — detecção de fraude em transações de cartão via *k-Nearest Neighbors* (k=5, distância euclidiana) sobre 3 milhões de vetores de referência em 14 dimensões.

A implementação é em Rust, sem banco de dados externo. Todo o pré-processamento do dataset acontece no build da imagem Docker e o índice fica embutido como arquivo binário acessado por `mmap`.

## Resultado oficial (prévia, Mac Mini Late 2014)

| Métrica | Valor |
|---|---|
| `final_score` | **5135.33** |
| `p99` HTTP | 7.32 ms |
| `p99_score` | 2135.33 / 3000 |
| `detection_score` | **3000 / 3000** (máximo absoluto) |
| FP / FN / Err | 0 / 0 / 0 |
| commit avaliado | `55fceed` |

## Topologia

```
cliente
   │  porta 9999
   ▼
┌──────────────┐  0.20 CPU / 30 MB
│  HAProxy LB  │  HTTP keep-alive + http-reuse
└─────┬────┬───┘
      │    │  Unix domain sockets
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

Total: **1.0 CPU + 350 MB** dentro do limite oficial de 1 CPU / 350 MB.

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
- `target-cpu = "x86-64-v3"` em `.cargo/config.toml` — habilita AVX2/FMA/BMI2 como baseline (note: o `.cargo/` precisa ser copiado no Dockerfile, fácil de esquecer).
- Tokio em `current_thread` runtime (cada réplica tem 0.40 vCPU; work-stealing seria desperdício).
- mimalloc como global allocator.

### HAProxy + Unix domain sockets

- HAProxy 3.3 como load balancer em `0.20` CPU / `30MB`.
- APIs escutam em Unix domain sockets (`/sockets/api1.sock`, `/sockets/api2.sock`) compartilhados por volume Docker.
- `mode http`, `option http-keep-alive` e `http-reuse always` para reaproveitar conexões upstream.
- `ulimit nofile=65535` para LB e APIs.

### Caminho de requisição

- Resposta como uma de 6 `&'static [u8]` pré-construídas — sem `ryu`, sem serde, sem alocação.
- `Response::new` + `headers_mut().insert` em vez de `Response::builder` — uma alocação a menos por request.
- Timeout interno de KNN em 3 ms; em caso de timeout, retorna `{"approved": true, "fraud_score": 0.0}` em HTTP 200. Evita o peso 5× de `Err` no `score_det`.

### KNN

- `Index::open` faz **prefault** de todas as páginas do mmap (93 MB) e roda **512 queries de aquecimento** antes de servir. Custo de startup, não de request.
- Distância i16 vetorizada com **AVX2** (`_mm256_madd_epi16` mascarado para zerar as 2 lanes de padding) — bench p99 de 105 us para 83 us.

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
- `haproxy.cfg` — load balancer HTTP com upstream em Unix sockets.
- `docker-compose.yml` — duas réplicas + HAProxy, limites de recurso.
- `.cargo/config.toml` — flags de compilação por target.

## Configuração via variáveis de ambiente

| Variável | Default | Descrição |
|---|---|---|
| `API_ADDR` | `0.0.0.0:8080` | Endereço do servidor HTTP |
| `SOCKET_PATH` | vazio | Se definido, a API escuta em Unix socket em vez de TCP |
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

Local em host de bancada com outros processos competindo por CPU; oficial no Mac Mini Late 2014 dedicado da Rinha.

| Versão | Ambiente | FP+FN | det_score | p99 | final |
|---|---|---|---|---|---|
| Baseline (`8587310`) | local | — | — | 3.49 ms | 3868.66 |
| `nprobe=2` runtime tuning | local | 178 | — | 4.69 ms | 4615.18 |
| Wave C (cargo profile + nginx + tokio) | local | 38 | 2286 | 3.09 ms | 4796.17 |
| Wave 1 (static resp + mimalloc + adaptive nprobe) | local | 8 | 2630 | 4.61 ms | 4967.01 |
| Wave 2a (K=4096 k-means++ 25 iters) | local | 0 | 3000 | 10.68 ms (*) | 4971 |
| Wave 2a (mesmo commit, primeira prévia) | **Mac Mini** | 1 | 2819 | 105.50 ms | 3796.15 |
| `.cargo/config.toml` no Dockerfile | **Mac Mini** | 1 | 2819 | 112.86 ms | 3766.85 |
| Warmup do mmap + KNN no startup | **Mac Mini** | 0 | 3000 | 113.83 ms | 3943.74 |
| CPU split nginx/API + AVX2 IVF | **Mac Mini** | 0 | 3000 | 7.32 ms | 5135.33 |
| nginx stream + Unix sockets | local | 0 | 3000 | 3.46 ms | 5460.78 |
| HAProxy TCP + Unix sockets | local | 0 | 3000 | 3.41 ms | 5466.84 |
| **HAProxy HTTP reuse + Unix sockets** | local | **0** | **3000** | **3.14 ms** | **5503.33** |

(*) Medição antiga com host de bancada saturado.

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
| Load balancer | HAProxy 3.3 |
| Container | Docker / docker-compose |
| Compilação | rustc 1.x com `target-cpu=x86-64-v3`, LTO fat |

## Descobertas

Anotações honestas do que mexeu o ponteiro e o que não mexeu, em ordem de aprendizado:

1. **Detecção e latência são problemas separados.** Detecção saturou cedo com K=4096 + k-means++ + 25 iterações; latência exigiu mexer no caminho HTTP/LB.

2. **O ambiente local engana.** Bench local de KNN em dezenas de microssegundos não prevê sozinho o p99 HTTP no Mac Mini. A primeira prévia oficial mostrou que fila no load balancer podia dominar tudo.

3. **CPU no load balancer importou, mas só até certo ponto.** Subir nginx de `0.10` para `0.20` CPU derrubou o p99 oficial de ~113 ms para 7.32 ms. Depois disso, tirar CPU das APIs para dar `0.25` ao LB regrediu no teste local.

4. **Unix domain sockets foram a próxima alavanca real.** Trocar upstream TCP por UDS reduziu o p99 local para a faixa de 3 ms mantendo detecção perfeita.

5. **HAProxy HTTP com `http-reuse always` ganhou do nginx stream e do HAProxy TCP.** Melhor rodada local: 3.14 ms p99, `det_score=3000`, score 5503.33.

6. **AVX2 explícito na distância IVF i16 valeu a pena.** Trocar o loop escalar de 14 dims por `_mm256_madd_epi16` reduziu o p99 do KNN isolado de ~105 us para ~83 us.

7. **k-means++ + 25 iterações + K=4096 zerou os erros no preview.** Só aumentar K sem bons centróides não basta.

8. **Adaptive nprobe (8 -> 24 nos borderline)** deixa pagar pouco no caso médio sem perder recall nos difíceis.

9. **Trabalho que pareceu boa ideia e regrediu / ficou neutro:**
   - Substituir axum por bare hyper: neutro em 5 runs locais. O overhead do axum não estava custando o suficiente para justificar a complexidade extra.
   - Parser JSON com `&str` borrowed + datetime manual: regrediu localmente. Hipótese é que o serde_json valida ausência de escapes para fazer o borrow zero-copy, e essa validação custa mais do que economiza num payload pequeno típico. Pode ser que valha em parser totalmente manual com `memchr` (jairoblatt faz isso), mas a versão híbrida não compensou.
   - Servidor HTTP manual em Tokio sobre UDS: piorou no k6 local (`5.03 ms` p99) contra Axum/Hyper sobre UDS, então foi descartado.
   - Mais CPU para o load balancer (`0.25/0.375/0.375`): piorou bastante no host local; as APIs ainda precisam dos `0.40` por réplica.

10. **A diferença até 6000.** O teto absoluto da Rinha é p99 <= 1 ms para zerar `p99_score` em 3000. Com detecção no máximo, os pontos restantes dependem só de cortar caminho HTTP/runtime.

Detalhes commit a commit em `git log`; cada onda saiu como commit separado com motivação e número.
