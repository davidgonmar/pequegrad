from typing import List
from pequegrad.tensor import Tensor


def parse_einsum_input(
    equation: str, operand1: Tensor, operand2: Tensor
) -> List[List[int]]:
    """
    Gets an equation of the form 'ij,jk->ik' and a list of tensors (operands)
    It verifies that shapes are as expected.
    Returns:
        - Dim to sum over in the input tensors

    Example:
    equation = 'ij,jk->ik'
    a = Tensor(np.random.randn(10, 20))
    b = Tensor(np.random.randn(20, 30))
    res = parse_einsum_input(equation, a, b)
    print(res)  # [[1], [0]]
    """

    # Get the input and output labels
    input_labels, output_labels = equation.split("->")
    input_labels = input_labels.split(",")
    assert len(input_labels) == 2, "Only two operands supported"
    input_labels = list(map(lambda x: x.strip(), input_labels))
    output_labels = output_labels.strip()

    # Check that the input labels are unique
    if len(input_labels) != len(set(input_labels)):
        raise ValueError("Repeated labels in input")
    assert (
        operand1.dim == operand2.dim
    ), "Both operands must have the same number of dimensions, got {} and {}".format(
        operand1.ndim, operand2.ndim
    )

    inpl1, inpl2 = input_labels

    to_sum1 = []
    to_sum2 = []
    for i, (l1, l2) in enumerate(zip(inpl1, inpl2)):
        if l1 not in output_labels:
            assert (
                l1 in inpl2
            ), "got a dimension to reduce that is not on second operand, got {}".format(
                l1
            )
            to_sum1.append(i)
        if l2 not in output_labels:
            to_sum2.append(i)
            assert (
                l2 in inpl1
            ), "got a dimension to reduce that is not on first operand, got {}".format(
                l2
            )

    return [to_sum1, to_sum2]


def einsum(equation: str, *operands: Tensor) -> Tensor:
    assert len(operands) == 2, "Only two operands supported"
    to_sum = parse_einsum_input(equation, *operands)
    # now do a tensordot based on the to_sum
    return operands[0].tensordot(operands[1], dims=to_sum)
