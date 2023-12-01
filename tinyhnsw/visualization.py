from tinyhnsw import HNSWIndex

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


def generate_tiny_world():
    """
    'tiny world' is a set of 2d points (snapped to an integer grid) that we're
    going to use to visualize HNSW in 2 dimensions.
    """
    pass