from pequegrad.tensor import Tensor
import networkx as nx


ID = 0


def get_tensor_repr(node: Tensor):
    global ID
    if not hasattr(node, "id"):
        node.id = ID
        ID += 1
    s = (
        "shape: " + str(node.shape)
        if not is_scalar(node)
        else "value: " + str(node.numpy())
    )
    return str(node.id) + "\n" + s


def is_scalar(node: Tensor):
    return len(node.shape) == 0


def build_graph(root: Tensor):
    G = nx.DiGraph()
    visited = set()
    stack = [root]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            color = "green" if is_scalar(node) else "blue"
            G.add_node(get_tensor_repr(node), color=color)
            for c in node._ctx.children if node._ctx else []:
                stack.append(c)
                G.add_edge(
                    get_tensor_repr(c),
                    get_tensor_repr(node),
                    label=node._ctx.__class__.__name__,
                )
    return G
