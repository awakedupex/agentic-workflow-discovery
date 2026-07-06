<div align="center">
  <h1>🔄 Agentic Workflow Discovery Engine</h1>
  <p><em>Discover macro-tasks from enterprise UI clickstreams and predict the next user action — with zero data leakage.</em></p>

  <p>
    <img src="https://img.shields.io/badge/python-3.11%2B-blue?logo=python" alt="Python 3.11+">
    <img src="https://img.shields.io/badge/tests-59%20passing-brightgreen" alt="59 tests passing">
    <img src="https://img.shields.io/badge/latency-%3C15ms-brightgreen" alt="Inference <15ms">
    <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License">
  </p>
</div>

---

## 🎯 What This Does

A modular, production-grade pipeline that ingests messy UI event logs and:

1. **Cleans** the noise — collapses consecutive duplicate states (25% reduction)
2. **Discovers macro-tasks** — partitions UI states into task clusters via spectral graph analysis (NMI = 1.0 for disjoint tasks)
3. **Predicts next actions** — a GRU neural network forecasts the next UI interaction with **82% top-3 accuracy** in **<15ms**
4. **Tunes decisions** — finds the optimal automation threshold under asymmetric costs (missing a task costs 10× more than a false alarm)

---

## 🏗️ Architecture

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│  Ingestion   │ →  │    Graph     │ →  │    Model     │ →  │ Optimization│ →  │  Operating   │
│  & Cleaning  │    │  Clustering  │    │  (GRU/LSTM)  │    │ (Threshold) │    │    Point     │
├─────────────┤    ├──────────────┤    ├──────────────┤    ├─────────────┤    ├──────────────┤
│ • Pydantic  │    │ • NetworkX   │    │ • PyTorch    │    │ • Expected  │    │ • Decision   │
│   validation│    │ • sklearn    │    │   GRU        │    │   cost min  │    │   gating     │
│ • UI loop   │    │   Spectral   │    │ • 5-token    │    │ • ROC sweep │    │ • Confidence │
│   collapse  │    │   Clustering │    │   context    │    │ • C_fn=10×  │    │   threshold  │
│ • Chrono    │    │ • Silhouette │    │ • Temporal   │    │   C_fp      │    │              │
│   split     │    │   /NMI eval  │    │   leakage    │    │             │    │              │
│             │    │              │    │   guard ✓    │    │             │    │              │
└─────────────┘    └──────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
```

### Anti-Leakage Defense (3 Layers)

| Layer | Mechanism |
|---|---|
| **Split** | Strict chronological split by timestamp, not random |
| **Tokenizer** | Vocabulary fit on training data only; unseen tokens → `[UNK]` |
| **Dataset** | `__getitem__` asserts every input timestamp < label timestamp |

---

## 🚀 Quick Start

```bash
# 1. Set up environment
uv venv --python 3.12
source .venv/bin/activate

# 2. Install
uv pip install -e ".[dev]"

# 3. Generate synthetic data
python scripts/generate_mock_data.py --n_sessions 500

# 4. Run tests
make test        # fast tests (43 tests, ~3s)
make test-slow   # full suite including training (59 tests, ~15s)
```

---

## 📊 Performance

| Metric | Value | Test |
|---|---|---|
| **NMI (disjoint tasks)** | 1.0 (perfect) | `test_disjoint_tasks_perfect_separation` |
| **Silhouette score** | ≥ 0.85 | `test_silhouette_score_in_valid_range` |
| **Top-3 next-action accuracy** | ~82% | Integration smoke test |
| **Inference latency (avg)** | < 15ms | `test_inference_latency_below_15ms` |
| **Inference latency (P99)** | < 20ms | `test_p99_latency_below_20ms` |
| **Overfit capability** | Loss < 0.05 on 5 sequences | `test_overfit_micro_batch` |
| **Temporal leakage** | 0 violations | `test_temporal_leakage.py` (4 tests) |
| **UI noise reduction** | ~25% | Loop collapse removes consecutive duplicates |

---

## 📁 Project Structure

```
src/agentic_workflow_discovery/
├── ingestion/      # Pydantic schemas, UI loop collapse, chronological split
├── graph/          # NetworkX transition graph, sklearn spectral clustering
├── model/          # GRU/LSTM, PyTorch Dataset with leakage guard, trainer, predictor
├── optimization/   # Asymmetric cost matrix, ROC threshold sweep, operating point
└── pipeline/       # Orchestrator — wires all stages together

tests/
├── test_ingestion/     # 14 tests — cleaning edge cases, split boundary
├── test_graph/         # 17 tests — NMI=1.0, silhouette, weight accumulation
├── test_model/         # 16 tests — leakage guard, overfitting, latency budget
├── test_optimization/  # 10 tests — cost ratios, threshold shift, ROC monotonicity
└── test_pipeline/      # 2 tests — end-to-end smoke test

scripts/
├── generate_mock_data.py  # Synthetic enterprise UI traces with 3 embedded macro-tasks
```

---

## 🧪 Running the Full Test Suite

```bash
# All fast tests
make test

# Including slow (training) and benchmark (latency) tests
make test-slow

# Latency budget only
make bench

# Lint checks
make lint

# Type checks
make typecheck
```

---

## 🔑 Key Design Decisions

| Decision | Rationale |
|---|---|
| **GRU over Transformer** | 5-token context — Transformer's self-attention has no advantage; GRU is 5× faster with ½ the parameters |
| **Spectral clustering** | UI workflows are chains (A→B→C), not spheres. K-Means fails; graph Laplacian finds communities |
| **Time-based split** | Random split leaks future context; chronological split simulates real deployment |
| **C_fn=10, C_fp=1** | Missed automation = user disappointment; false alarm = minor annoyance. 10:1 is a sensible default |
| **15ms latency budget** | Human perception threshold for "instantaneous" interaction is ~20ms; 15ms leaves headroom |

---

## 📚 Interview Prep

Each module file contains inline `# Design rationale` comments explaining *why* each choice was made. Key questions you can answer after studying this project:

- *"How do you prevent data leakage in a time-series model?"*
- *"Why spectral clustering over K-Means for UI sequence data?"*
- *"When would you choose a GRU over a Transformer?"*
- *"How do you set a decision threshold under asymmetric costs?"*
- *"What makes this pipeline production-grade?"*

---

<div align="center">
  <sub>Built with Python, PyTorch, NetworkX, scikit-learn & PyTorch</sub>
  <br>
  <sub>MIT License</sub>
</div>
