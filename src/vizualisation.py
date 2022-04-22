import networkx as nx
import matplotlib as plt
import matplotlib.pyplot as plt


def draw_hg(hg):
    # Get the largest component of the graph
    components = nx.connected_components(hg)
    largest_component_size = max(components, key=len)
    hg_lcc = hg.subgraph(largest_component_size)

    # Draw the network
    from matplotlib.pyplot import figure
    pos = nx.spring_layout(hg, seed=7)  # positions for all nodes - seed for reproducibility
    figure(figsize=(20, 10), dpi=80)

    subax1 = plt.subplot(121)
    plt.title("The full network", fontsize=20)
    nx.draw(hg, pos, node_size = 80, with_labels=False)
    subax2 = plt.subplot(122)

    plt.title("Largest connected component (zoomed in), using weighted edges", fontsize=20)
    # Draw the nodes and the weighted edges
    elarge = [(u, v) for (u, v, d) in hg_lcc.edges(data=True) if d["weight"] > 2]
    esmall = [(u, v) for (u, v, d) in hg_lcc.edges(data=True) if d["weight"] <= 2]
    nx.draw_networkx_nodes(hg_lcc, pos, node_size=80)
    nx.draw_networkx_edges(hg_lcc, pos, edgelist=elarge, width=1)
    nx.draw_networkx_edges(
        hg_lcc, pos, edgelist=esmall, width=2, alpha=0.5, edge_color="b", style="dashed"
    )
    plt.axis("off")
    nx.draw(hg_lcc, node_size = 80, with_labels=False)