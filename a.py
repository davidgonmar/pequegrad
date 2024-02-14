from pequegrad.tensor import Tensor

t1 = Tensor([1, 2, 3, 4]).to("cuda")


t2 = t1.sum(-1)
