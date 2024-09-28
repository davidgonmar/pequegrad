import pequegrad.ops as ops


def tree_all_reduce(tensors, op="sum"):
    num_tensors = len(tensors)
    if num_tensors == 1:
        return tensors[0]

    while num_tensors > 1:
        new_tensors = []
        for i in range(0, num_tensors, 2):
            if i + 1 < num_tensors:
                reduced_tensor = tensors[i] + tensors[i + 1].to(tensors[i].device)
                new_tensors.append(reduced_tensor)
            else:
                new_tensors.append(tensors[i])
        tensors = new_tensors
        num_tensors = len(tensors)

    total_sum = tensors[0]

    if op == "avg":
        total_sum = total_sum / len(tensors)
    elif op != "sum":
        raise ValueError(f"Unknown op {op}")

    # broadcast back
    for i in range(len(tensors)):
        tensors[i].assign(total_sum.to(tensors[i].device))

    return total_sum


def naive_all_reduce(tensors, op="sum"):
    total_sum = tensors[0]
    for tensor in tensors[1:]:
        total_sum = total_sum + tensor.to(total_sum.device)
    if op == "sum":
        pass
    elif op == "avg":
        total_sum = total_sum / len(tensors)
    else:
        raise ValueError(f"Unknown op {op}")
    for tensor in tensors:
        tensor.assign(total_sum.to(tensor.device))
    return tensor


def reduce_to_one_device(tensors, op="sum"):
    total_sum = tensors[0]
    for tensor in tensors[1:]:
        total_sum = total_sum + tensor.to(total_sum.device)
    if op == "sum":
        pass
    elif op == "avg":
        total_sum = total_sum / len(tensors)
    else:
        raise ValueError(f"Unknown op {op}")
    return total_sum


all_reduce = tree_all_reduce
