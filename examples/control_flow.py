# Jittable control flow with pretty prints!
import pequegrad as pg
from pequegrad.tensor import Tensor
import pequegrad.ops as ops


def body(i, x):
    return i + 1, x + 1


print("\n--- While Loop ---")
print("Evaluating while_loop with initial i=0.0, x=10.0...")
final = pg.jit(ops.while_loop)(lambda i, x: i < 10, body, (Tensor(0.0), Tensor(10.0)))
print("Result:", final.numpy(), "# Expected: 20.0")

print("\nComputing gradient of while_loop with respect to input tuple...")
final = pg.fngrad(ops.while_loop, wrt=[2])(
    lambda i, x: i < 10, body, (Tensor(0.0), Tensor(10.0))
)
print("Gradients:", final[0][0].numpy(), final[0][1].numpy(), "# Expected: 0.0 1.0")

print("\n--- Fori Loop ---")
print("Evaluating fori_loop with range(0, 10), initial x=10.0...")
finalfori = pg.jit(ops.fori_loop)(
    Tensor(0), Tensor(10), lambda i, x: x + 1, (Tensor(10.0),)
)
print("Result:", finalfori.numpy(), "# Expected: 20.0")

print("\nComputing gradient of fori_loop with respect to i0, i1, and x...")
finalfori = pg.fngrad(ops.fori_loop, wrt=[0, 1, 3])(
    Tensor(0), Tensor(10), lambda i, x: x + 1, (Tensor(10.0),)
)
print(
    "Gradients:",
    finalfori[0].numpy(),
    finalfori[1].numpy(),
    finalfori[2][0].numpy(),
    "# Expected: 0.0 0.0 1.0",
)

print("\n--- If-Else Branching ---")


def _if(inp):
    return inp * 3


def _else(inp):
    return inp + 2


print("Evaluating ifelse with x=10.0 (should take else branch)...")
finalif = pg.jit(ops.ifelse)(lambda x: x < 10, _if, _else, (Tensor(10.0),))
print("Result:", finalif.numpy(), "# Expected: 12.0")

finalif = pg.jit(ops.ifelse)(lambda x: x < 10, _if, _else, (Tensor(5.0),))
print("Result:", finalif.numpy(), "# Expected: 15.0")

print("\nComputing gradient of ifelse at x=10.0 (else branch)...")
finalif = pg.fngrad(ops.ifelse, wrt=[3])(lambda x: x < 10, _if, _else, (Tensor(10.0),))
print("Gradient:", finalif[0][0].numpy(), "# Expected: 1.0")

print("Computing gradient of ifelse at x=5.0 (if branch)...")
finalif = pg.fngrad(ops.ifelse, wrt=[3])(lambda x: x < 10, _if, _else, (Tensor(5.0),))
print("Gradient:", finalif[0][0].numpy(), "# Expected: 3.0")
