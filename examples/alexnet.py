from pequegrad import Tensor, device, grads
import pequegrad.modules as nn
from pequegrad.optim import SGD
import argparse
import torch
import torchvision
import numpy as np


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


# download cifar from torch


transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        torchvision.transforms.Resize((224, 224)),
    ]
)

trainset = torchvision.datasets.CIFAR100(
    root="./data", train=True, download=True, transform=transform
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)

testset = torchvision.datasets.CIFAR100(
    root="./data", train=False, download=True, transform=transform
)

testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)


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

DEVICE = device.cuda
model = AlexNet(num_classes=100).to(DEVICE)
print("Number of parameters:", sum([p.numel() for p in model.parameters()]))
print("Size in MB:", sum([p.numel() * 4 for p in model.parameters()]) * 1e6)
args = parser.parse_args()

if args.checkpoint is not None:
    model.load(args.checkpoint)

if not args.test:
    optim = SGD(model.parameters(), lr=0.01)

    for epoch in range(args.epochs):
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            inputs = Tensor(inputs.numpy().astype("float32"), device=DEVICE)
            labels = Tensor(labels.numpy().astype("float32"), device=DEVICE)

            outputs = model(inputs)
            loss = outputs.cross_entropy_loss_indices(labels)
            print(f"Epoch {epoch}, iter {i}, loss: {loss.numpy()}")
            # format day_month_hour_minute
            model.save("alexnet_checkpoint.pkl")
            g = grads(model.parameters(), loss)
            optim.step(g)


if args.test:
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        images = Tensor(images.numpy().astype("float32"), device=DEVICE)
        labels = Tensor(labels.numpy().astype("float32"), device=DEVICE)
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
