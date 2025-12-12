\
import networkx as nx
import matplotlib.pyplot as plt
import os

def save_graph_png(G, out_path='reports/attack_graph.png', figsize=(10,8)):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=False, node_size=50)
    labels = {n: n.split('/')[-1] or n for n in list(G.nodes())[:50]}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=6)
    plt.title('Attack Graph')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path

def save_graph_gexf(G, out_path='reports/attack_graph.gexf'):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nx.write_gexf(G, out_path)
    return out_path
