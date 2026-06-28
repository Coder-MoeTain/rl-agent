"""Tests for graph visualization."""

import networkx as nx

from utils.graph_visualize import save_graph_gexf, save_graph_png


def test_save_graph_png_empty(tmp_path):
    g = nx.DiGraph()
    path = save_graph_png(g, tmp_path / "empty.png")
    assert path.exists()


def test_save_graph_png_with_nodes(tmp_path):
    g = nx.DiGraph()
    g.add_edge("http://localhost:3000/", "http://localhost:3000/login")
    path = save_graph_png(g, tmp_path / "graph.png")
    assert path.exists()


def test_save_graph_gexf(tmp_path):
    g = nx.DiGraph()
    g.add_node("http://localhost:3000/")
    path = save_graph_gexf(g, tmp_path / "graph.gexf")
    assert path.exists()
