<div align="center">
  <h1>🔄 Agentic Workflow Discovery Engine</h1>
  <p><em>A production-grade pipeline that discovers macro-tasks from raw UI event streams and predicts the next user action — with zero data leakage and sub-15ms inference.</em></p>

  <p>
    <img src="https://img.shields.io/badge/python-3.11%2B-blue?logo=python" alt="Python 3.11+">
    <img src="https://img.shields.io/badge/tests-59%20passing-brightgreen" alt="59 tests passing">
    <img src="https://img.shields.io/badge/latency-%3C15ms-brightgreen" alt="Inference <15ms">
    <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License">
  </p>

  <p>
    <strong>PyTorch</strong> ·
    <strong>scikit-learn</strong> ·
    <strong>NetworkX</strong> ·
    <strong>Pydantic</strong>
  </p>
</div>

---

## 📋 Overview

Enterprise desktop activity logs are noisy, redundant, and unstructured. This project builds an **automated pipeline** that ingests raw click/keystroke/focus event streams, cleans them, discovers hidden "macro-tasks" (invoicing, email, code review), and trains a real-time sequence model to predict the user's next action — enabling intelligent desktop automation.

### What it does

| Stage | Input | Output | Method |
|---|---|---|---|
| **Ingestion** | Messy event logs | Cleaned, chronologically split streams | Pydantic validation + UI loop collapse |
| **Graph Clustering** | Event sequences | Partitioned macro-tasks | NetworkX + Spectral Clustering |
| **Next-Action Prediction** | 5-event context windows | Top-3 next action predictions | PyTorch GRU |
| **Decision Tuning** | Model confidence scores | Optimal automation threshold | Asymmetric cost optimisation |

---

## 🔬 Key Results

| Metric | Value | How It's Verified |
|---|---|---|
| **Noise reduction** | ~25% | Consecutive duplicate state collapse |
| **Task discovery accuracy** | NMI = 1.0 (perfect) | Spectral clustering on disjoint synthetic tasks |
| **Cluster quality** | Silhouette ≥ 0.85 | sklearn `silhouette_score` on transition graph |
| **Next-action accuracy** | ~82% top-3 | End-to-end integration test |
| **Inference latency** | **<15ms avg, <20ms P99** | `test_latency_budget.py` (100 timed runs on M-series CPU) |
| **Model overfitting sanity** | Loss < 0.05 on 5 samples | `test_overfit_micro_batch` |
| **Data leakage** | **Zero violations** | 3-layer guard: chronological split + train-only tokenizer + timestamp assert in Dataset |

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

### Anti-Leakage Architecture

Temporal data leakage is the #1 failure mode in time-series ML. This pipeline prevents it at three independent layers:

1. **Chronological split** — train/val/test divided by timestamp quantiles, not randomly
2. **Train-only tokenizer** — vocabulary built exclusively from training events; unseen production states map to `[UNK]`
3. **Dataset assert** — every training sample verifies all input timestamps are strictly before the label timestamp

---

## 💻 Tech Stack

| Category | Library | Purpose |
|---|---|---|
| **Language** | Python 3.11+ | Type-hinted throughout |
| **Validation** | Pydantic v2 | Strict schema enforcement (`frozen=True`, `extra="forbid"`) |
| **Graph** | NetworkX | Directed transition graph with weighted edges |
| **Clustering** | scikit-learn | Spectral clustering on graph Laplacian |
| **Deep Learning** | PyTorch 2.x | GRU sequence model with MPS/CPU support |
| **Optimisation** | NumPy/SciPy | Asymmetric cost sweep, ROC curve |
| **Testing** | pytest | 59 tests incl. latency benchmarks and leakage detection |

---

## 🧪 Test Suite

**59 tests — all passing** — covering mathematical correctness, ML invariants, and real-world constraints.

```
tests/
├── test_ingestion/     (14 tests)  — Pydantic validation, loop collapse, split boundaries
├── test_graph/         (17 tests)  — NMI=1.0, silhouette, edge weights, reproducibility
├── test_model/         (16 tests)  — Leakage guard, overfitting, 15ms latency budget
├── test_optimization/  (10 tests)  — Cost ratios, threshold shift, ROC monotonicity
└── test_pipeline/      ( 2 tests)  — End-to-end smoke test with synthetic data
```

```bash
make test        # fast tests (43 tests, ~3s)
make test-slow   # full suite (59 tests, ~15s)
make bench       # latency benchmarks only
```

---

## 🚀 Getting Started

```bash
# Set up environment
uv venv --python 3.12
source .venv/bin/activate

# Install with dev dependencies
uv pip install -e ".[dev]"

# Generate 50,000+ synthetic UI events
python scripts/generate_mock_data.py --n_sessions 500

# Run the full pipeline
python -c "
from agentic_workflow_discovery.pipeline.orchestrator import Orchestrator
from scripts.generate_mock_data import generate_dataset

events, _ = generate_dataset(n_sessions=10, seed=42)
orch = Orchestrator()
orch.ingest(events)
orch.build_graph()
orch.train_model(max_epochs=20)
print(orch.predict(events[:5]))
"
```

---

## 📁 Project Structure

```
src/agentic_workflow_discovery/
├── ingestion/      # Schemas, cleaner, splitter, pipeline
├── graph/          # Builder, cluster, metrics
├── model/          # Tokenizer, dataset, GRU, trainer, predictor
├── optimization/   # Cost, threshold, controller
└── pipeline/       # Orchestrator (wires all stages)

scripts/
├── generate_mock_data.py   # Synthetic data generator (3 macro-tasks + noise)
```

---

## 🔑 Engineering Decisions

| Decision | Trade-off |
|---|---|
| **GRU over Transformer** | Context window is 5 tokens — Transformer's self-attention offers no benefit at this length. GRU is faster, smaller, and equally accurate. Would switch to Transformer for >32-token contexts. |
| **Spectral over K-Means** | UI workflows form chain-like DAGs (A→B→C), not spherical clusters. The graph Laplacian captures community structure where K-Means fails. |
| **Time-based train/test split** | Random split would leak future task patterns into the training set. Chronological split simulates real-world deployment. |
| **10:1 cost asymmetry** | False negatives (missed automations) cost 10× more than false alarms. The decision threshold is tuned to this ratio via expected-cost minimisation. |
| **15ms latency budget** | Under 20ms is imperceptible to users. 15ms leaves headroom for tokenisation and post-processing. Verified by P99 benchmark. |

---

<div align="center">
  <sub>Python · PyTorch · NetworkX · scikit-learn · Pydantic · pytest</sub>
  <br>
  <sub>MIT License</sub>
</div>
