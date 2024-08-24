from pequegrad import shard_tensor, device, Tensor, topology_diagram
import numpy as np

devices = [device.cuda, device.cpu, device.cpu, device.cuda]

t = Tensor(np.random.randn(120, 60))
t2 = Tensor(np.random.randn(60, 120))

sharded1 = shard_tensor(t, 4, dim=0, devices=devices)
sharded2 = shard_tensor(t2, 4, None, devices=devices)

topology_diagram(sharded1)
topology_diagram(sharded2)

assert np.allclose((sharded1 @ sharded2).numpy(), t.numpy() @ t2.numpy())

sharded3 = shard_tensor(t, 4, dim=None, devices=devices)
sharded4 = shard_tensor(t2, 4, dim=1, devices=devices)

assert np.allclose((sharded3 @ sharded4).numpy(), t.numpy() @ t2.numpy())
