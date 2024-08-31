from collections.abc import Mapping, Sequence
from collections import namedtuple
from pequegrad.modules import StatefulModule, NonStatefulModule

is_module = lambda x: isinstance(x, (StatefulModule, NonStatefulModule)) or issubclass(
    type(x), (StatefulModule, NonStatefulModule)
)

PyTreeDef = namedtuple("PyTreeDef", ["type", "structure"])


def is_leaf(node):
    """Check if a node is a leaf in the PyTree."""
    return not (isinstance(node, (Mapping, Sequence)) or is_module(node))


def tree_flatten(pytree):
    """Flatten a PyTree into a list of leaves and a PyTreeDef."""
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
    elif is_module(pytree):
        params = pytree.parameters()
        return params, PyTreeDef(type="_module", structure=None)


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
        raise NotImplementedError(
            "tree_unflatten for modules is not implemented. Can only flatten modules."
        )


def count_leaves(pytree_def):
    """Count the number of leaves in a PyTreeDef."""
    if pytree_def.type is None:
        return 1

    return (
        sum(count_leaves(child_def) for _, child_def in pytree_def.structure)
        if pytree_def.type is dict
        else sum(count_leaves(child_def) for child_def in pytree_def.structure)
    )
