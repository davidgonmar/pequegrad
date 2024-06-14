import networkx as nx
import matplotlib.pyplot as plt


def viz(tensor, viz=True):
    G = nx.DiGraph()

    def add_node(tensor):
        if tensor not in seen:
            if tensor.ad_context() == "AsType":
                print(tensor.children())
            node_name = tensor.id
            G.add_node(node_name, label=f"{node_name}\n({tensor.ad_context()})")
            seen.add(tensor)
            for child in tensor.children():
                child_name = child.id
                G.add_node(child_name, label=f"{child_name}\n({child.ad_context()})")
                G.add_edge(child_name, node_name)
                add_node(child)

    seen = set()
    add_node(tensor)

    if viz:
        pos = nx.spring_layout(G)
        labels = nx.get_node_attributes(G, "label")
        nx.draw(
            G,
            pos,
            labels=labels,
            with_labels=True,
            node_size=3000,
            node_color="lightblue",
            font_size=10,
            font_weight="bold",
            arrows=True,
        )
        plt.show()

    return G
