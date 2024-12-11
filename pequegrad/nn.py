import pequegrad.ops as ops
from pequegrad.tensor import Tensor
from pequegrad.transforms.pytree import tree_map


class BaseModule:
    def __init__(self):
        pass

    def _get_submodules(self, root=True, di={}, curr_key=None, ret_params=False):
        # search in dict
        self_di = self.__dict__
        for key, value in self_di.items():
            if isinstance(value, BaseModule):
                if not ret_params:
                    if root:
                        di[key] = value
                    else:
                        di[curr_key][key] = value

                else:
                    if root:
                        di[key] = value.init()
                    else:
                        di[curr_key][key] = value.init()

            if isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, BaseModule):
                        if root:
                            di[key] = {}
                        else:
                            di[curr_key][key] = {}
                        item._get_submodules(
                            root=False, di=di, curr_key=key, ret_params=ret_params
                        )
            if isinstance(value, dict):
                for key2, value2 in value.items():
                    if isinstance(value2, BaseModule):
                        if root:
                            di[key] = {}
                        else:
                            di[curr_key][key] = {}
                        value2._get_submodules(
                            root=False, di=di, curr_key=key, ret_params=ret_params
                        )
        return di

    def init(self, device=None):
        mods = self._get_submodules(ret_params=True)
        if device is not None:
            mods = tree_map(lambda x: x.to(device), mods)
        return mods

    def _apply_params_dict(self, params_dict):
        # we need to set eachs module's params
        self_di = self.__dict__
        for key, value in self_di.items():
            if isinstance(value, BaseModule):
                value.params = params_dict[key]
            if isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, BaseModule):
                        item.params = params_dict[key][i]
            if isinstance(value, dict):
                for key2, value2 in value.items():
                    if isinstance(value2, BaseModule):
                        value2.params = params_dict[key][key2]

    def _reset_params_dict(self):
        self_di = self.__dict__
        for key, value in self_di.items():
            if isinstance(value, BaseModule):
                del value.params
            if isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, BaseModule):
                        del item.params
            if isinstance(value, dict):
                for key2, value2 in value.items():
                    if isinstance(value2, BaseModule):
                        del value2.params

    def __repr__(self) -> str:
        return "BaseModule()"

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)


class Linear(BaseModule):
    def __init__(self, in_size: int, out_size: int):
        self.in_size = in_size
        self.out_size = out_size

    def init(self):
        return {
            "w": ops.randn((self.in_size, self.out_size)),
            "b": ops.zeros((self.out_size,)),
        }

    def apply(self, x: Tensor):
        return ops.add(ops.matmul(x, self.params["w"]), self.params["b"])

    def __repr__(self) -> str:
        return f"Linear({self.in_size}, {self.out_size})"


def apply_fn(mod: BaseModule, *args, params_dict, **kwargs):
    mod._apply_params_dict(params_dict)
    ret = mod.apply(*args, **kwargs)
    mod._reset_params_dict()
    return ret
