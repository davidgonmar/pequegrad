from collections.abc import Mapping, Sequence
from collections import namedtuple
from pequegrad.modules import StatefulModule, NonStatefulModule


PyTreeDef = namedtuple("PyTreeDef", ["type", "structure"])

is_module = lambda x: isinstance(x, (StatefulModule, NonStatefulModule)) or issubclass(
    type(x), (StatefulModule, NonStatefulModule)
)


def is_leaf(node):
    """Check if a node is a leaf in the PyTree."""
    return not (isinstance(node, (Mapping, Sequence)) or is_module(node))


def make_pytree_list(l):
    lfs = [PyTreeDef(type=None, structure=None) for _ in l]
    return PyTreeDef(type=list, structure=lfs)


def make_pytree_nested_list(rows, cols):
    lfs = [make_pytree_list(range(cols)) for _ in range(rows)]
    return PyTreeDef(type=list, structure=lfs)


def is_pytree(x):
    """Check if x is a PyTree type."""
    return isinstance(x, PyTreeDef)


def tree_flatten(pytree):
    """Flatten a PyTree into a list of leaves and a PyTreeDef."""
    if is_pytree(pytree):
        # if is None, return empty list
        if pytree.type is None:
            return [pytree], pytree

        leaves = []
        for i, child_def in pytree.structure:
            leaves.extend(tree_flatten(i)[0])
        return leaves, pytree

    if is_module(pytree):
        params = pytree.parameters()
        child_struct = [PyTreeDef(type=None, structure=None) for _ in params]
        return params, PyTreeDef(type=list, structure=child_struct)
    if is_leaf(pytree):
        return [pytree], PyTreeDef(type=None, structure=None)
    if isinstance(pytree, Mapping):
        leaves = []
        structure = []
        for key, value in pytree.items():
            flattened, child_structure = tree_flatten(value)
            leaves.extend(flattened)
            structure.append((key, child_structure))
        return leaves, PyTreeDef(type=dict, structure=structure)
    elif isinstance(pytree, Sequence):
        leaves = []
        structure = []
        for i, item in enumerate(pytree):
            flattened, child_structure = tree_flatten(item)
            leaves.extend(flattened)
            structure.append(child_structure)
        return leaves, PyTreeDef(type=type(pytree), structure=structure)


def tree_unflatten(pytree_def, leaves):
    """Reconstruct a PyTree from its flattened version."""
    if pytree_def.type is None:
        return leaves[0]

    if pytree_def.type is dict:
        result = {}
        idx = 0
        for key, child_def in pytree_def.structure:
            num_leaves = count_leaves(child_def)
            result[key] = tree_unflatten(child_def, leaves[idx : idx + num_leaves])
            idx += num_leaves
        return result

    elif issubclass(pytree_def.type, Sequence):
        result = []
        idx = 0
        for child_def in pytree_def.structure:
            num_leaves = count_leaves(child_def)
            result.append(tree_unflatten(child_def, leaves[idx : idx + num_leaves]))
            idx += num_leaves
        return pytree_def.type(result)

    elif pytree_def.type == "_module":
        raise NotImplementedError("Module unflattening not implemented yet.")


def count_leaves(pytree_def):
    """Count the number of leaves in a PyTreeDef."""
    if pytree_def.type is None:
        return 1

    return (
        sum(count_leaves(child_def) for _, child_def in pytree_def.structure)
        if pytree_def.type is dict
        else sum(count_leaves(child_def) for child_def in pytree_def.structure)
    )
