import numpy as np
from pequegrad.modules import Linear, Conv2d, Module
from pequegrad.extra.mnist import get_mnist_dataset
from pequegrad.tensor import Tensor


modelpath = "conv_mnist_model.pkl"


class ConvNet(Module):
    def __init__(self):
        # input size = 28x28
        self.conv1 = Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.conv2 = Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.fc1 = Linear(16 * 5 * 5, 10)

    def forward(self, input):
        input = input.reshape((-1, 1, 28, 28))  # shape: (28, 28)
        input = (
            self.conv1.forward(input).relu().max_pool2d(kernel_size=(2, 2))
        )  # shape: (28, 28) -> (26, 26) -> (13, 13)
        input = (
            self.conv2.forward(input).relu().max_pool2d(kernel_size=(2, 2))
        )  # shape: (13, 13) -> (11, 11) -> (5, 5)
        input = input.reshape((-1, 16 * 5 * 5))
        return self.fc1.forward(input)


model = ConvNet()

model.load(modelpath)

CUDA = model.parameters()[0].storage_type == "cuda"

print(f"Using {'CUDA' if CUDA else 'CPU'} for computations")

X_train, Y_train, X_test, Y_test = get_mnist_dataset()

batch_size = 512

# Evaluate the model
correct = 0
for i in range(0, len(X_test), batch_size):
    end_idx = min(i + batch_size, len(X_test))
    batch_X = Tensor(X_test[i:end_idx], storage="cuda" if CUDA else "np")
    batch_Y = Tensor(Y_test[i:end_idx], storage="cuda" if CUDA else "np")

    prediction = model.forward(batch_X)
    correct += (np.argmax(prediction.numpy(), axis=1) == batch_Y.numpy()).sum()

print(f"Accuracy: {correct}/{len(X_test)} = {correct/len(X_test)*100:.2f}%")
