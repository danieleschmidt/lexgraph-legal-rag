from lexgraph_legal_rag.metrics import start_metrics_server


def test_start_metrics_server():
    # start on ephemeral port; should not raise
    start_metrics_server(0)
