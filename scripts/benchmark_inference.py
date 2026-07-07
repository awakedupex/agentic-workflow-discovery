"""Benchmark GRU inference latency against the 15ms budget.

Runs 1,000 forward passes and reports avg, P50, P95, P99 latency.

Usage:
    python scripts/benchmark_inference.py
"""

import time
import torch
from agentic_workflow_discovery.model.lstm_model import SequenceLSTM


def main():
    print("─" * 50)
    print("Inference Latency Benchmark")
    print("─" * 50)

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = SequenceLSTM(
        vocab_size=1000,
        embedding_dim=128,
        hidden_dim=256,
        n_layers=2,
        dropout=0.0,
    ).to(device)
    model.eval()

    input_ids = torch.randint(0, 1000, (1, 5), dtype=torch.long, device=device)

    # Warmup
    for _ in range(10):
        _ = model(input_ids)

    # Timed runs
    n_runs = 1000
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model(input_ids)
        times.append((time.perf_counter() - start) * 1000)

    times.sort()
    avg = sum(times) / len(times)
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    p99 = times[int(len(times) * 0.99)]

    print(f"\nDevice: {device}")
    print(f"Runs:   {n_runs}")
    print(f"───")
    print(f"Avg:    {avg:.2f} ms")
    print(f"P50:    {p50:.2f} ms")
    print(f"P95:    {p95:.2f} ms")
    print(f"P99:    {p99:.2f} ms")
    budget = 15.0
    print(f"\nBudget: {budget:.0f} ms  →  {'PASS' if avg < budget else 'FAIL'}")

    return avg


if __name__ == "__main__":
    main()
