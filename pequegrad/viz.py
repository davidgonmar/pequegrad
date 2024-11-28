import networkx as nx
from pyvis.network import Network
from pequegrad.tensor import Tensor


def viz(tensor_or_tensors, viz=True, name="graph"):
    G = nx.DiGraph()
    tensor_or_tensors = (
        tensor_or_tensors
        if isinstance(tensor_or_tensors, (list, tuple))
        else [tensor_or_tensors]
    )
    tensor_or_tensors = [t for t in tensor_or_tensors if isinstance(t, Tensor)]

    def add_node(tensor):
        if tensor not in seen:
            node_name = tensor.id
            G.add_node(
                node_name,
                label=f"{node_name}\n({tensor.ad_context() + str(tensor.position)}), shape: {tensor.shape}, strides: {tensor.strides}",
            )
            seen.add(tensor)
            for child in tensor.children():
                child_name = child.id
                G.add_node(
                    child_name,
                    label=f"{child_name}\n({child.ad_context() + str(child.position)}), shape: {child.shape}, strides: {child.strides}",
                )
                G.add_edge(child_name, node_name, relation="child")
                add_node(child)

            for sibling in tensor.siblings():
                sibling_name = sibling.id
                G.add_node(
                    sibling_name,
                    label=f"{sibling_name}\n({sibling.ad_context() + str(sibling.position)}), shape: {sibling.shape}, strides: {sibling.strides}",
                )
                G.add_edge(sibling_name, node_name, relation="sibling")
                add_node(sibling)

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

        for source, target, data in G.edges(data=True):
            edge_options = {"arrows": "to"}
            if data.get("relation") == "child":
                edge_options["color"] = {"color": "blue"}
                edge_options["dashes"] = False
            elif data.get("relation") == "sibling":
                edge_options["color"] = {"color": "red"}
                edge_options["dashes"] = True

            net.add_edge(source, target, **edge_options)

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
            "enabled": false
          }
        }
        """
        )

        net.show(name + ".html")

    return G
