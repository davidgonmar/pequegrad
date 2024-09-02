from pequegrad import Tensor, fngrad, checkpoint


@checkpoint
def f(args):
    a, b, c = args
    x = a * b
    return x * c


a, b, c = Tensor(2.0), Tensor(3.0), Tensor(4.0)

f_and_grad = fngrad(f, wrt=[0], return_outs=True)
res, grads = f_and_grad((a, b, c))


f_and_grad.print_trace()

print(res.numpy())  # 24.0

# print all grads
for g in grads:
    print(g.numpy())  # 12.0, 8.0, 6.0
