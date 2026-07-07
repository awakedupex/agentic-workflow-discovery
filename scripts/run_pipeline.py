"""Run the full Agentic Workflow Discovery pipeline from synthetic data
to next-action prediction — one command, zero configuration.

Usage:
    python scripts/run_pipeline.py
"""

from scripts.generate_mock_data import generate_dataset
from agentic_workflow_discovery.pipeline.orchestrator import Orchestrator
from agentic_workflow_discovery.graph.metrics import evaluate_partition


def main():
    print("─" * 50)
    print("Agentic Workflow Discovery — Full Pipeline")
    print("─" * 50)

    # 1. Generate data
    print("\n[1/5] Generating synthetic event data...")
    events, _ = generate_dataset(n_sessions=10, seed=42)
    print(f"       {len(events)} events generated")

    # 2. Ingest & clean
    print("[2/5] Running ingestion pipeline...")
    orch = Orchestrator(context_length=5, embedding_dim=32, hidden_dim=64, n_layers=1)
    orch.ingest(events)
    print(f"       Train: {len(orch.split.train)}  Val: {len(orch.split.val)}  Test: {len(orch.split.test)}")

    # 3. Graph clustering
    print("[3/5] Building transition graph & clustering...")
    orch.build_graph(n_clusters=3)
    metrics = evaluate_partition(orch.graph, orch.state_to_cluster)
    print(f"       Nodes: {orch.graph.number_of_nodes()}  Edges: {orch.graph.number_of_edges()}")
    print(f"       Silhouette: {metrics['silhouette']:.3f}  Clusters: {int(metrics['n_clusters'])}")

    # 4. Train model
    print("[4/5] Training GRU model...")
    losses = orch.train_model(vocab_size=200, max_epochs=20, batch_size=16)
    print(f"       Final loss: {losses[-1]:.4f}")

    # 5. Predict
    print("[5/5] Running sample prediction...")
    context = orch.split.test[:5]
    if len(context) == 5:
        result = orch.predict(context)
        print(f"       Predicted token: {result['token_id']}  (confidence: {result['confidence']:.2%})")
        print("       Top-3:")
        for item in result["top_k"]:
            print(f"         {item['token_id']:4d}  {item['score']:.2%}  {item['state']}")

    print("\nDone.")


if __name__ == "__main__":
    main()
