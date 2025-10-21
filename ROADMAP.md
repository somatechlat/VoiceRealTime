# Voice Engine Roadmap & Sprint Plan

**Project Goal** – Deliver a production‑ready, OpenAI Realtime‑compatible voice agent that uses the Kokoro TTS model (hexgrad/Kokoro‑82M) with full streaming, interactive cancel, and UI controls. All components must be open‑source, observable, and capable of handling millions of transactions per day.

---

## High‑Level Milestones
| Milestone | Description | Target Sprint |
|-----------|-------------|----------------|
| **M1 – Core Engine Wiring** | Kokoro TTS backend, voice selection, speed control, fallback to Piper/espeak, streaming support. | Sprint 1 |
| **M2 – UI & API Enhancements** | `/v1/tts/voices` endpoint, advanced‑voice.html UI with voice picker, speed slider, barge‑in, interrupt button. | Sprint 2 |
| **M3 – Session Config & Cancel** | Read `tts.voice`/`tts.speed` from `session.session_config`; implement WS `response.cancel` that instantly stops streaming. | Sprint 3 |
| **M4 – Performance & Reliability** | Load‑testing, Prometheus/Grafana observability, Redis‑based rate limiting, CI pipelines, automated regression tests. | Sprint 4 |
| **M5 – Production‑Ready Release** | Multi‑region Docker‑Compose/K8s deployment, documentation, SDK generation, hand‑off checklist. | Sprint 5 |

---

## Sprint Structure (2‑3 weeks each)
We will run **parallel workstreams** so that up to three developers can be productive at the same time. Each sprint contains a **To‑Do list** (owner‑agnostic) and a **Definition of Done (DoD)**.

### Sprint 1 – Core Engine Wiring
**Goal:** Get Kokoro TTS streaming end‑to‑end and ensure fallback paths work.

#### To‑Do
1. **Dockerfile pre‑fetch fix** – ensure `Kokoro‑82M.onnx` and voice files are copied into `/app/cache/kokoro` at build time.
2. **TTSProvider abstraction** – create `tts/provider.py` with `class TTSProvider(ABC): async def synthesize(text, voice, speed) -> AsyncIterator[bytes]`.
3. **KokoroProvider implementation** – stream PCM chunks, expose `KOKORO_VOICE` and `KOKORO_SPEED` env vars.
4. **PiperFallbackProvider** – simple wrapper that calls the existing Piper CLI and yields chunks.
5. **Gateway integration** – modify the `/v1/realtime` WS handler to call `TTSProvider.synthesize` and forward `response.audio.delta` events.
6. **Unit tests** – verify that a short prompt returns at least three audio chunks and that the voice/ speed env vars affect output length.
7. **Health endpoint** – update `/health` to report `kokoro_model_present: true/false`.

#### Definition of Done
- Docker image builds without errors and contains the ONNX model.
- A WebSocket client receives streaming audio for a test prompt.
- Fallback Piper is used when `TTS_ENGINE=python`.
- 100 % unit‑test coverage for the new provider classes.

---
### Sprint 2 – UI & API Enhancements
**Goal:** Expose voice list and give end‑users UI controls.

#### To‑Do
1. **`/v1/tts/voices` endpoint** – read directories under `${KOKORO_MODEL_DIR}/voices` and return JSON matching OpenAI’s voice schema (`id`, `name`, `preview_url`).
2. **Advanced UI (`advanced-voice.html`)** –
   - Add a **voice dropdown** populated from the new endpoint.
   - Add a **speed slider** (0.5‑2.0) that sends `session.update` with `tts.speed`.
   - Implement a **barge‑in button** that sends `session.update` with `audio.interrupt=true`.
   - Implement an **interrupt/cancel button** that sends `response.cancel`.
3. **Frontend‑to‑backend mapping** – in `realtime_ws.py` read `session.session_config` for `tts.voice` and `tts.speed` and pass them to the `TTSProvider`.
4. **Styling** – use Tailwind (already in repo) for a clean, responsive layout.
5. **Integration tests** – headless Selenium or Playwright script that loads the UI, selects a voice, changes speed, and verifies audio plays.

#### Definition of Done
- `/v1/tts/voices` returns a non‑empty list with correct JSON fields.
- UI loads, populates the dropdown, and can change voice & speed.
- Pressing *Cancel* stops audio playback instantly (no more than 50 ms after click).
- End‑to‑end test passes on CI.

---
### Sprint 3 – Session Config & Interactive Cancel
**Goal:** Make the system respect per‑session TTS settings and support real‑time interruption.

#### To‑Do
1. **Session model** – extend `session.session_config` schema to include `tts.voice` and `tts.speed` (default from env). Persist in Redis hash.
2. **`session.update` handler** – validate incoming voice/speed, store in Redis, and apply to the next TTS request.
3. **`response.cancel` implementation** –
   - Add a cancellation flag in the streaming coroutine.
   - When the WS receives `response.cancel`, set the flag and break the async generator.
   - Send `response.cancelled` event to the client.
4. **Barge‑in handling** – when a new `input_audio_buffer.append` arrives while TTS is streaming, automatically trigger `response.cancel`.
5. **Performance test** – simulate 5 k concurrent sessions with random cancels; ensure cancel latency ≤ 50 ms.
6. **Documentation** – update API spec in `openapi.yaml` to include the new fields and cancel semantics.

#### Definition of Done
- Session config values survive across multiple requests within the same session.
- Cancel button stops streaming audio in ≤ 50 ms (measured in automated test).
- Barge‑in automatically cancels TTS when user starts speaking.
- API docs reflect new fields.

---
### Sprint 4 – Performance, Observability & CI
**Goal:** Harden the service for production traffic.

#### To‑Do
1. **Prometheus metrics** – expose latency histograms (`voice_asr_first_partial_ms`, `voice_tts_first_chunk_ms`, `voice_bargein_cancel_ms`) and queue‑depth gauges.
2. **Grafana dashboards** – import the community “OpenAI Realtime” dashboard and add panels for Kokoro model cache size.
3. **Redis rate‑limiting** – implement token‑bucket middleware; add config `RATE_LIMIT=120` per user.
4. **Load testing** – write a Locust script that opens 2 k WS clients, streams prompts, and randomly cancels. Capture SLOs.
5. **GitHub Actions CI** – lint (ruff), type‑check (mypy), unit tests, integration tests, and a nightly load‑test run.
6. **Failure injection** – chaos‑monkey script that kills a TTS worker to verify graceful fallback.

#### Definition of Done
- All metrics appear in Prometheus and Grafana.
- Rate‑limit headers are returned on overload.
- Locust run shows 99th‑percentile latency < 250 ms for first TTS chunk.
- CI pipeline passes on every PR.

---
### Sprint 5 – Production Release & Documentation
**Goal:** Ship the final product and enable external developers.

#### To‑Do
1. **Multi‑region Docker‑Compose / Helm chart** – include Caddy, Redis Cluster, and GPU workers.
2. **SDK generation** – use `openapi-generator` to produce Python, JavaScript, and TypeScript client libraries.
3. **User guide** – markdown docs covering deployment, env‑vars, UI usage, and troubleshooting.
4. **Security audit** – run `bandit` and address any findings; add JWT auth middleware.
5. **Version bump** – tag `v1.0.0` and publish Docker images to Docker Hub under `openvoiceos/voice‑engine`.
6. **Post‑mortem checklist** – run a simulated traffic spike, verify auto‑scaling, and document lessons learned.

#### Definition of Done
- Deployable Helm chart tested on a Kubernetes cluster.
- SDKs published to PyPI and npm.
- Documentation site built with MkDocs and hosted on GitHub Pages.
- Release notes and changelog updated.

---

## Parallel Workstreams (Team Allocation)
| Workstream | Typical Owner(s) | Sprint(s) |
|------------|------------------|-----------|
| **Docker & Model Prefetch** | Infra / DevOps | 1‑2 |
| **TTS Provider & Fallback** | Backend Engineer | 1‑3 |
| **WebSocket Protocol & Session Logic** | Backend Engineer | 1‑3 |
| **Advanced UI (HTML/JS/Tailwind)** | Front‑end Engineer | 2‑3 |
| **Observability (Prometheus/Grafana)** | SRE / Ops | 4 |
| **Load & Chaos Testing** | QA Engineer | 4‑5 |
| **Documentation & SDKs** | Technical Writer / API Engineer | 5 |

Each workstream has a clear deliverable per sprint, so the team can work concurrently without stepping on each other’s toes.

---

## Quick Start for Contributors
1. **Clone the repo** and run `docker compose up -d --build`.
2. **Open VS Code** – the workspace already contains Tailwind, FastAPI, and Docker configs.
3. **Pick a sprint** from the table above, create a branch `sprintX‑<feature>` and start coding.
4. **Run tests** locally with `pytest` and `locust --headless -u 10 -r 2`.
5. **Push** and open a PR; CI will run the full pipeline.

---

*This roadmap is a living document – feel free to adjust sprint dates or task granularity as the team discovers new constraints.*
