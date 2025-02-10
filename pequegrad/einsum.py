from typing import List
from pequegrad.tensor import Tensor
from functools import reduce


class EinsumIdx:
    def __init__(self, idx: str):
        self.idx = idx

    def __repr__(self):
        return self.idx

    def __str__(self):
        return self.idx

    def __eq__(self, other):
        return self.idx == other.idx

    def __hash__(self):
        return hash(self.idx)

    @staticmethod
    def from_str_multiple(s: str):
        return [EinsumIdx(c) for c in s]


def extract_indices(eq: str) -> List[List[str]]:
    return [EinsumIdx.from_str_multiple(s) for s in eq.split("->")[0].split(",")] + [
        EinsumIdx.from_str_multiple(eq.split("->")[1])
    ]


def partial_einsum(
    input1: Tensor,
    input1_idxs: List[EinsumIdx],
    input2: Tensor,
    input2_idxs: List[EinsumIdx],
    output_indices: List[EinsumIdx],
) -> Tensor:
    # determine if there are non-free shared indices (indices that are in both inputs and output)
    # if there are, we need to reduce over them, else, we can cas the operation as a matrix multiplication (with reshaping and stuff)

    non_free_shared_indices = (
        len(set(input1_idxs) & set(input2_idxs) & set(output_indices)) > 0
    )

    if not non_free_shared_indices:
        op1, op2 = input1, input2
        input_indices = [input1_idxs, input2_idxs]
        # reduction indices are the intersection of the input indices that are not in the output indices

        inters = set(input_indices[0]) & set(input_indices[1])
        reduction_indices = [idx for idx in inters if idx not in output_indices]

        # get the position of the reduction indices in the input indices
        position1 = [input_indices[0].index(idx) for idx in reduction_indices]
        position2 = [input_indices[1].index(idx) for idx in reduction_indices]

        # for input 1, we want to bring the reduction indices to the last dimension, and for the second input, we want to bring them to the first dimension
        # so we need to permute the dimensions of the inputs
        perm1 = [
            i for i in range(len(input_indices[0])) if i not in position1
        ] + position1
        perm2 = position2 + [
            i for i in range(len(input_indices[1])) if i not in position2
        ]

        def _prod(shp, from_idx, to_idx):
            return reduce(lambda x, y: x * y, shp[from_idx:to_idx], 1)

        # we want to transform the equation into a matrix multiplication, so, after permuting, we need to reshape the inputs into (non_reduction_indices, reduction_indices)
        permuted1 = op1.permute(*perm1)
        permuted2 = op2.permute(*perm2)

        reshaped1 = permuted1.reshape(
            (
                -1,
                _prod(
                    permuted1.shape,
                    len(op1.shape) - len(reduction_indices),
                    len(op1.shape),
                ),
            )
        )
        reshaped2 = permuted2.reshape(
            (_prod(permuted2.shape, 0, len(reduction_indices)), -1)
        )

        out = reshaped1 @ reshaped2

        non_reduced_shape1 = [
            op1.shape[input_indices[0].index(idx)]
            for idx in input_indices[0]
            if idx not in reduction_indices
        ]
        non_reduced_shape2 = [
            op2.shape[input_indices[1].index(idx)]
            for idx in input_indices[1]
            if idx not in reduction_indices
        ]

        ret_shape = non_reduced_shape1 + non_reduced_shape2

        out = out.reshape(ret_shape)

        # out.shape is now (non_reduction_indices_1, non_reduction_indices_2)
        # we need to find the permutation that brings it to the desired shape
        # the permutation is given by the output indices
        symbolic_out_result_shape = [
            input_indices[0][i]
            for i in range(len(input_indices[0]))
            if input_indices[0][i] not in reduction_indices
        ] + [
            input_indices[1][i]
            for i in range(len(input_indices[1]))
            if input_indices[1][i] not in reduction_indices
        ]

        return out, symbolic_out_result_shape

    else:
        # put input 1 in the form (non_common_indices, common_indices) and input 2 in the form (common_indices, non_common_indices)

        common_indices = list(set(input1_idxs) & set(input2_idxs))

        non_common_indices1 = [idx for idx in input1_idxs if idx not in common_indices]

        non_common_indices2 = [idx for idx in input2_idxs if idx not in common_indices]

        perm1 = [input1_idxs.index(idx) for idx in non_common_indices1] + [
            input1_idxs.index(idx) for idx in common_indices
        ]

        perm2 = [input2_idxs.index(idx) for idx in common_indices] + [
            input2_idxs.index(idx) for idx in non_common_indices2
        ]

        permuted1 = input1.permute(*perm1)

        permuted2 = input2.permute(*perm2)

        # permuted1 has shape (non_common_indices1, common_indices)
        # permuted2 has shape (common_indices, non_common_indices2)

        # we can broadcast permuted1 to (non_common_indices1, common_indices, [1] * len(non_common_indices2))
        # and permuted2 to ([1] * len(non_common_indices1), common_indices, non_common_indices2)

        # then we can multiply them and sum over the common indices THAT ARE NOT IN THE OUTPUT INDICES

        reshaped1 = permuted1.reshape(permuted1.shape + [1] * len(non_common_indices2))

        reshaped2 = permuted2.reshape([1] * len(non_common_indices1) + permuted2.shape)

        out = reshaped1 * reshaped2

        out_symbolic_shape = (
            [idx for idx in non_common_indices1]
            + common_indices
            + [idx for idx in non_common_indices2]
        )

        # sum over the common indices that are not in the output indices
        reduction_indices = [idx for idx in common_indices if idx not in output_indices]

        # get the actual axis

        reduction_axis = [out_symbolic_shape.index(idx) for idx in reduction_indices]

        out = out.sum(reduction_axis)

        out_symbolic_shape = [
            idx for idx in out_symbolic_shape if idx not in reduction_indices
        ]

        return out, out_symbolic_shape


def einsum(equation: str, operands: List[Tensor]) -> List[List[int]]:
    idxs = extract_indices(equation)

    inp1 = operands[0]
    idxs1, out_idxs_final = idxs[0], idxs[-1]
    curr_outs_idx = None
    for i in range(1, len(operands)):
        inp2 = operands[i]
        idxs2 = idxs[i]
        l = list(set(idxs1 + idxs2)) if i < len(operands) - 1 else out_idxs_final
        out, out_idxs = partial_einsum(inp1, idxs1, inp2, idxs2, l)
        inp1 = out
        idxs1 = out_idxs
        curr_outs_idx = out_idxs
    symbolic_out_shape = curr_outs_idx

    # find permutation that brings the output to the desired shape
    perm = [symbolic_out_shape.index(idx) for idx in out_idxs_final]
    return out.permute(*perm)
