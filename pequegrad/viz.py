import networkx as nx
from pyvis.network import Network


def viz(tensor_or_tensors, viz=True, name="graph"):
    G = nx.DiGraph()

    def add_node(tensor):
        if tensor not in seen:
            node_name = tensor.id
            additional = (
                tensor.eval(False).detach().numpy()
                if tensor.numel() < 10
                else tensor.eval(False).detach().numpy().sum()
            )
            G.add_node(
                node_name,
                label=f"{node_name}\n({tensor.ad_context()}), shape: {tensor.shape}, evaled: {tensor.is_evaled()} {additional}",
            )
            seen.add(tensor)
            for child in tensor.children():
                additional = (
                    child.eval(False).detach().numpy()
                    if child.numel() < 10
                    else child.eval(False).detach().numpy().sum()
                )
                child_name = child.id
                G.add_node(
                    child_name,
                    label=f"{child_name}\n({child.ad_context()}), shape: {child.shape}, evaled: {child.is_evaled()} {additional}",
                )
                G.add_edge(child_name, node_name)
                add_node(child)

    seen = set()
    for tensor in (
        tensor_or_tensors
        if isinstance(tensor_or_tensors, (list, tuple))
        else [tensor_or_tensors]
    ):
        add_node(tensor)

    if viz:
        net = Network(notebook=True, directed=True)

        for node, data in G.nodes(data=True):
            net.add_node(
                node, label=data["label"], title=data["label"], shape="ellipse"
            )
        for source, target in G.edges():
            net.add_edge(source, target, arrows="to")

        net.set_options(
            """
        var options = {
          "layout": {
            "hierarchical": {
              "enabled": true,
              "direction": "UD",
              "sortMethod": "directed",
              "nodeSpacing": 300
            }
          },
          "edges": {
            "smooth": {
              "type": "cubicBezier",
              "forceDirection": "vertical",
              "roundness": 0.4
            }
          },
          "physics": {
            "hierarchicalRepulsion": {
              "nodeDistance": 300
            },
            "minVelocity": 0.75
          }
        }
        """
        )

        net.show(name + ".html")

    return G
