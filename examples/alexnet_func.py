from pequegrad import Tensor, fngrad, maybe
import pequegrad.modules as nn
from pequegrad.optim import SGDState, sgd
import argparse
import pequegrad.ds_transforms as transforms
from pequegrad.data.dataloader import DataLoader
from pequegrad.extra.cifar_100 import CIFAR100Dataset
import pequegrad as pg
from pequegrad.transforms.pytree import tree_map


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
    default=0.0005,
    help="Learning rate for the optimizer",
)

parser.add_argument(
    "--bs",
    type=int,
    default=128,
    help="Batch size for training",
)

args = parser.parse_args()

transform = transforms.Compose(
    [
        transforms.ToTensor(device="cuda"),
        transforms.JitCompose(
            [
                transforms.PermuteFromTo((0, 1, 2, 3), (0, 3, 1, 2)),  # NHWC -> NCHW
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Resize((224, 224)),
            ],
        ),
        transforms.EvalAndDetach(),
    ],
)

trainset = CIFAR100Dataset(train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=args.bs, shuffle=True)
testset = CIFAR100Dataset(train=False, transform=transform)
testloader = DataLoader(testset, batch_size=args.bs, shuffle=False)

model = AlexNet(num_classes=100).to("cuda")
print("Number of parameters:", sum([p.numel() for p in model.parameters()]))
print("Size in MB:", sum([p.numel() * 4 for p in model.parameters()]) / 1024 / 1024)


if not args.test:
    do_jit = True

    @maybe(pg.jit.withargs(allocator="custom"), do_jit)
    def update_step(state, params_dict, x, y):
        x = x.to("cuda")
        y = y.to("cuda")
        y = Tensor.one_hot(100, y)

        def get_loss(x, y, params_dict):
            outs = nn.apply_to_module(model, params_dict, x)
            return outs.cross_entropy_loss_probs(y)

        loss, (grads,) = fngrad(get_loss, wrt=[2], return_outs=True)(x, y, params_dict)
        new_state, params_dict = sgd(params_dict, grads, state)
        # if we are not jitting, evaluate the lazy new state and params_dict
        if not do_jit:
            tree_map(lambda x: x.eval() if isinstance(x, Tensor) else x, new_state)
            tree_map(lambda x: x.eval() if isinstance(x, Tensor) else x, params_dict)
            loss.eval()
        return new_state, params_dict, loss

    import time

    state = SGDState(model, lr=args.lr)
    params_dict = model.tree_flatten()
    for epoch in range(args.epochs):
        for i, data in enumerate(trainloader, 0):
            st = time.time()
            inputs, labels = data
            state, params_dict, loss = update_step(state, params_dict, inputs, labels)
            print(
                f"Epoch {epoch}, iter {i}, loss: {loss.numpy()}, time: {time.time() - st}"
            )
            if i % 100 == 0:
                model.save("alexnet_checkpoint.pkl")
