from collections.abc import Mapping, Sequence
from collections import namedtuple
from pequegrad.modules import StatefulModule, NonStatefulModule


PyTreeDef = namedtuple("PyTreeDef", ["type", "structure"])

is_module = lambda x: isinstance(x, (StatefulModule, NonStatefulModule)) or issubclass(
    type(x), (StatefulModule, NonStatefulModule)
)


def is_pytree(x):
    """Check if x is a PyTree type."""
    return isinstance(x, PyTreeDef)


def is_leaf(node):
    """Check if a node is a leaf in the PyTree."""
    return not isinstance(node, (Mapping, Sequence)) or isinstance(node, (str, bytes))


def make_pytree_list(l):
    lfs = [PyTreeDef(type=None, structure=None) if not is_pytree(i) else i for i in l]
    return PyTreeDef(type=list, structure=lfs)


def make_pytree_nested_list(rows, cols):
    lfs = [make_pytree_list(range(cols)) for _ in range(rows)]
    return PyTreeDef(type=list, structure=lfs)


def first_tensor_pytree(pytree):
    return tree_flatten(pytree)[0][0]


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
        params_dict = pytree.tree_flatten()
        leaves = []
        structure = []
        for key, value in params_dict.items():
            flattened, child_structure = tree_flatten(value)
            leaves.extend(flattened)
            structure.append((key, child_structure))
        return leaves, PyTreeDef(type=dict, structure=structure)
    if hasattr(pytree, "_raw_struct_for_tree_flatten"):
        return tree_flatten(pytree._raw_struct_for_tree_flatten())

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


def pytree_def_to_dict(pytree_def):
    if isinstance(pytree_def, (list, tuple)) and not is_pytree(pytree_def):
        return [pytree_def_to_dict(child_def) for child_def in pytree_def]

    # leaves will be set to None
    if pytree_def.type is None:
        return None

    if pytree_def.type is dict:
        return {
            key: pytree_def_to_dict(child_def)
            for key, child_def in pytree_def.structure
        }

    if issubclass(pytree_def.type, Sequence):
        return [pytree_def_to_dict(child_def) for child_def in pytree_def.structure]

    if pytree_def.type == "_module":
        raise NotImplementedError("Module flattening not implemented yet.")


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


def tree_map(f, *structs):
    """Map a function f over one or more PyTrees."""
    leaves = [tree_flatten(pytree)[0] for pytree in structs]
    result_leaves = [f(*leaves) for leaves in zip(*leaves)]
    single_res_leave_pytree = tree_flatten(result_leaves[0])[1]
    # same as input pytree, but leaves are single_res_leave_pytree
    inp_pytree = tree_flatten(structs[0])[1]
    res_pytree = inp_pytree  # TODO
    return tree_unflatten(res_pytree, result_leaves)


def _check_same_structure(pytree1, pytree2):
    """Check if two PyTrees have the same structure."""
    if pytree1.type != pytree2.type:
        return False

    if pytree1.type is None:
        return True

    if pytree1.type is dict:
        return all(
            k1 == k2 and _check_same_structure(v1, v2)
            for (k1, v1), (k2, v2) in zip(pytree1.structure, pytree2.structure)
        )

    return all(
        _check_same_structure(c1, c2)
        for c1, c2 in zip(pytree1.structure, pytree2.structure)
    )


def check_same_structure(anys):
    """Check if all PyTrees have the same structure."""
    pytrees = [tree_flatten(pytree)[1] for pytree in anys]  # get the PyTreeDef
    return all(_check_same_structure(pytrees[0], pytree) for pytree in pytrees)
