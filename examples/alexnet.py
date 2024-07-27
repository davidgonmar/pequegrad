from pequegrad import Tensor, device, grads
import pequegrad.modules as nn
from pequegrad.optim import SGD, Adam, JittedAdam, JittedSGD
import argparse
import numpy as np
import pequegrad.transforms as transforms
from pequegrad.data.dataloader import DataLoader
from pequegrad.extra.cifar_100 import CIFAR100Dataset
from pequegrad.compile import jit


class AlexNet(nn.StatefulModule):
    """AlexNet model

    Characteristics (follows the paper):
        - Response normalization layers follow the first and second convolutional layers.
        - Max-pooling layers follow both response normalization layers as well as the fifth convolutional layer.
        - The ReLU non-linearity is applied to the output of every convolutional and fully-connected layer.
        - We use dropout in the first two fully-connected layers.

    The model expects input of size N x 3 x 224 x 224 and returns output of size N x num_classes.
    """

    def __init__(self, num_classes=100):
        super(AlexNet, self).__init__()

        # Features: N x 3 x 224 x 224 -> N x 256 x 6 x 6
        self.features = nn.Sequential(
            # Block 1: N x 3 x 224 x 224 -> N x 96 x 55 x 55
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Block 2: N x 96 x 55 x 55 -> N x 256 x 27 x 27
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Block 3: N x 256 x 27 x 27 -> N x 384 x 13 x 13
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            # Block 4: N x 384 x 13 x 13 -> N x 384 x 13 x 13
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            # Block 5: N x 384 x 13 x 13 -> N x 256 x 6 x 6
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Classifier: N x 256 x 6 x 6 -> N x num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        # Flattens the tensor from N x 256 x 6 x 6 to N x 9216
        x = x.reshape((x.shape[0], 256 * 6 * 6))
        x = self.classifier(x)
        return x


# allow to continue training from a checkpoint
parser = argparse.ArgumentParser(description="Train AlexNet on CIFAR-100")

parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to a checkpoint file to load and continue training",
)

parser.add_argument(
    "--test",
    action="store_true",
    help="Run the model on the test set after training",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    help="Number of epochs to train the model",
)

parser.add_argument(
    "--lr",
    type=float,
    default=0.005,
    help="Learning rate for the optimizer",
)

parser.add_argument(
    "--jit",
    action="store_true",
    default=False,
    help="Use CUDA for computations",
)

args = parser.parse_args()
transform = transforms.Compose(
    [
        transforms.ToTensor(device=device.cuda),
        transforms.JitCompose(
            [
                transforms.PermuteFromTo((0, 1, 2, 3), (0, 3, 1, 2)),  # NHWC -> NCHW
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ],
            enabled=args.jit,
        ),
        transforms.Resize((224, 224)),
        transforms.EvalAndDetach(),
    ],
)
trainset = CIFAR100Dataset(train=True, transform=transform)

trainloader = DataLoader(
    trainset, batch_size=30, shuffle=True
)  # jit allows about 30, no jit 45

testset = CIFAR100Dataset(train=False, transform=transform)

testloader = DataLoader(testset, batch_size=40, shuffle=False)


DEVICE = device.cuda
model = AlexNet(num_classes=100).to(DEVICE)
print("Number of parameters:", sum([p.numel() for p in model.parameters()]))
print("Size in MB:", sum([p.numel() * 4 for p in model.parameters()]) / 1024 / 1024)


if args.checkpoint is not None:
    model.load(args.checkpoint)

use_sgd = True
optims = {
    "compiled": {
        "adam": JittedAdam,
        "sgd": JittedSGD,
    },
    "uncompiled": {
        "adam": Adam,
        "sgd": SGD,
    },
}
if not args.test:
    str1 = "compiled" if args.jit else "uncompiled"
    str2 = "adam" if use_sgd else "sgd"
    optim = optims[str1][str2](model.parameters(), lr=args.lr)

    def train_step(x, y):
        outs = model(x)
        loss = outs.cross_entropy_loss_probs(y)
        g = grads(model.parameters(), loss)
        return [loss] + g

    use_jit = args.jit  # does not work yet
    train_step = (
        jit(train_step, externals=model.parameters()) if use_jit else train_step
    )
    import time

    for epoch in range(args.epochs):
        for i, data in enumerate(trainloader, 0):
            st = time.time()
            inputs, labels = data

            inputs = inputs.to(DEVICE)
            labels = Tensor(labels, device=DEVICE)

            batch_y_onehot = Tensor.one_hot(100, labels, device=DEVICE)
            outs = train_step(inputs, batch_y_onehot)
            # import pequegrad.viz as viz
            # viz.viz(outs, name="outs")
            loss = outs[0]
            g = outs[1:]
            optim.step(g)
            print(
                f"Epoch {epoch}, iter {i}, loss: {loss.numpy()}, time: {time.time() - st}"
            )
            if i == 10:
                raise Exception("stop")
            if i % 100 == 0:
                model.save("alexnet_checkpoint.pkl")

if args.test:
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        images = Tensor(images.numpy().astype("float32"), device=DEVICE)
        labels = Tensor(labels.astype("float32"), device=DEVICE)
        outputs = model(images)
        total += labels.shape[0]
        correct += np.sum(outputs.numpy().argmax(1) == labels.numpy())

        print(
            "Accuracy of the network on the 10000 test images: %d %%"
            % (100 * correct / total)
        )
    print(
        "Accuracy of the network on the 10000 test images: %d %%"
        % (100 * correct / total)
    )
    print(
        "Accuracy of the network on the 10000 test images: %d %%"
        % (100 * correct / total)
    )
