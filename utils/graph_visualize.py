"""Attack graph visualization utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def save_graph_png(
    graph: nx.DiGraph,
    out_path: str | Path = "reports/attack_graph.png",
    figsize: tuple[float, float] = (10, 8),
) -> Path:
    """Save attack graph as PNG image.

    Args:
        graph: NetworkX directed graph.
        out_path: Output file path.
        figsize: Figure size in inches.

    Returns:
        Path to saved PNG file.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=figsize)
    if graph.number_of_nodes() == 0:
        plt.title("Attack Graph (empty)")
        plt.savefig(path)
        plt.close()
        return path

    pos = nx.spring_layout(graph, seed=42)
    nx.draw(graph, pos, with_labels=False, node_size=50)
    labels = {n: n.split("/")[-1] or n for n in list(graph.nodes())[:50]}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=6)
    plt.title("Attack Graph")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def save_graph_gexf(
    graph: nx.DiGraph, out_path: str | Path = "reports/attack_graph.gexf"
) -> Path:
    """Save attack graph in GEXF format for Gephi/analysis tools."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_gexf(graph, path)
    return path
