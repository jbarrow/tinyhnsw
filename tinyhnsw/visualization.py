from tinyhnsw import HNSWIndex

import random
import networkx
import matplotlib.pyplot as plt


def visualize_hnsw_index(index: HNSWIndex):
    """
    Use this to visualize the different layers of HNSW graphs. The nodes
    maintain consistent locations between layers, and the layers are
    plotted next to each other.
    """

    _, axs = plt.subplots(1, len(index.layers), figsize=(len(index.layers) * 5, 5))

    layout = networkx.spring_layout(index.layers[0].G)
    node_color = ["r" if index.ep == node else "c" for node, _ in layout.items()]

    for i, layer in enumerate(index.layers):
        graph = layer.G

        graph_layout = {k: v for k, v in layout.items() if k in graph}
        graph_node_color = [node_color[k] for k, _ in graph_layout.items()]
        networkx.draw(
            graph, graph_layout, ax=axs[i], node_size=25, node_color=graph_node_color
        )
        axs[i].set_title(f"Layer {i}")

    plt.show()


def generate_tiny_world(n_nodes: int, width: int, height: int):
    """
    'tiny world' is a set of 2d points (snapped to an integer grid) that we're
    going to use to visualize HNSW in 2 dimensions.
    """
    xs = [random.randint(0, width-1) for _ in range(n_nodes)]
    ys = [random.randint(0, height-1) for _ in range(n_nodes)]

    return {ix: (xs[ix], ys[ix]) for ix in range(n_nodes)}


if __name__ == "__main__":
    G = networkx.random_lobster(5, 0.6, 0.3)
    layout = generate_tiny_world(len(G), 100, 100)
    networkx.draw(G, layout)

    plt.show()