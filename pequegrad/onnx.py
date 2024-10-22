import onnx
import pequegrad as pg
import numpy as np
from pequegrad.transforms.lazyfn import topo_recurse_until_reach_inputs, Identity

onnx_dt_to_pg_dt = {
    onnx.TensorProto.FLOAT: pg.dt.float32,
    onnx.TensorProto.INT32: pg.dt.int32,
}

onnx_dt_to_np_dt = {
    onnx.TensorProto.FLOAT: np.float32,
    onnx.TensorProto.INT32: np.int32,
    onnx.TensorProto.INT64: np.int64,
}

pg_dt_to_onnx_dt = {v: k for k, v in onnx_dt_to_pg_dt.items()}


def handle_gemm(op: onnx.NodeProto, tensor_dict: dict):
    transA = None
    transB = None
    for attr in op.attribute:
        if attr.name == "transA":
            transA = attr.i
        if attr.name == "transB":
            transB = attr.i

    x, w, b = op.input if len(op.input) == 3 else list(op.input) + [None]
    x, w, b = (
        tensor_dict[x],
        tensor_dict[w],
        (tensor_dict[b] if b is not None else None),
    )
    if transA:
        x = pg.transpose(x, -1, -2)
    if transB:
        w = pg.transpose(w, -1, -2)
    out = pg.matmul(x, w)
    if b is not None:
        out = out + b
    tensor_dict[op.output[0]] = out


def handle_constant(op: onnx.NodeProto, tensor_dict: dict):
    for attr in op.attribute:
        if attr.name == "value":
            tensor = attr.t
            npbuff = onnx.numpy_helper.to_array(tensor)
            assert npbuff.shape != (0,)  # check if shape is not empty
            # cast int64 to int32
            if tensor.data_type == onnx.TensorProto.INT64:
                npbuff = npbuff.astype(np.int32)
            tensor_dict[op.output[0]] = pg.Tensor(npbuff).reshape(tensor.dims)


def handle_relu(op: onnx.NodeProto, tensor_dict: dict):
    x = tensor_dict[op.input[0]]
    tensor_dict[op.output[0]] = pg.relu(x)


def handle_reshape(op: onnx.NodeProto, tensor_dict: dict):
    x, shape = op.input
    x, shape = tensor_dict[x], tensor_dict[shape]
    shape = (
        shape.numpy().astype(np.int32).tolist()
        if isinstance(shape, pg.Tensor)
        else shape
    )
    tensor_dict[op.output[0]] = x.reshape(shape)


def handle_transpose(op: onnx.NodeProto, tensor_dict: dict):
    x = tensor_dict[op.input[0]]
    tensor_dict[op.output[0]] = pg.transpose(x, -1, -2)


def handle_identity(op: onnx.NodeProto, tensor_dict: dict):
    tensor_dict[op.output[0]] = tensor_dict[op.input[0]]


def handle_elementwise(op: onnx.NodeProto, tensor_dict: dict, fn):
    x, y = op.input
    x, y = tensor_dict[x], tensor_dict[y]
    tensor_dict[op.output[0]] = fn(x, y)


op_dict = {
    "Gemm": handle_gemm,
    "Constant": handle_constant,
    "Relu": handle_relu,
    "Reshape": handle_reshape,
    "Transpose": handle_transpose,
    "Identity": handle_identity,
    "Add": lambda op, tensor_dict: handle_elementwise(
        op, tensor_dict, lambda x, y: x + y
    ),
    "Max": lambda op, tensor_dict: handle_elementwise(op, tensor_dict, pg.max),
}


class OnnxModel(pg.StatefulModule):
    def __init__(self, model_proto: onnx.ModelProto):
        self.graph = model_proto.graph
        self.param_dict = {}
        for tensor in self.graph.initializer:
            self.param_dict[tensor.name] = pg.Tensor(
                onnx.numpy_helper.to_array(tensor)
            ).reshape(tensor.dims)

    @staticmethod
    def _run_onnx_graph(graph: onnx.GraphProto, inputs: dict, preloadeds: dict):
        for input_tensor in graph.input:
            input_name = input_tensor.name
            if input_name not in inputs:
                raise ValueError(
                    f"Input tensor '{input_name}' not found in inputs dictionary"
                )
            input_shape = tuple(
                d.dim_value for d in input_tensor.type.tensor_type.shape.dim
            )
            if tuple(inputs[input_name].shape) != input_shape:
                raise ValueError(
                    f"Input tensor '{input_name}' has incorrect shape. Expected: {input_shape}, Got: {inputs[input_name].shape}"
                )
        tensor_dict = {}
        for tensor in graph.initializer:
            tensor_dict[tensor.name] = preloadeds[tensor.name]
        for input_tensor in graph.input:
            tensor_dict[input_tensor.name] = inputs[input_tensor.name]
        for node in graph.node:
            op_dict[node.op_type](node, tensor_dict)
        return {out.name: tensor_dict[out.name] for out in graph.output}

    def forward(self, inputs: dict):
        return self._run_onnx_graph(self.graph, inputs, self.param_dict)


def from_onnx_path(onnx_model_path):
    onnx_model = onnx.load(onnx_model_path)
    return OnnxModel(onnx_model)


def from_onnx_model(onnx_model):
    return OnnxModel(onnx_model)


# =========================== Exporting to ONNX ===========================


def export_to_onnx(model, dummy_input, out_names):
    def fn(inputs, params_dict):
        return pg.apply_to_module(model, params_dict, inputs)

    traced = Identity(fn)

    nodes = []
    inputs = []
    outputs = []
    initializers = []
    value_info = []
    params_dict = model.tree_flatten()
    tensor_dict = {}

    input_names = []
    param_names = []

    # flatten the params_dict keys!
    def _flatten_dict_keys(d):
        for k, v in d.items():
            if isinstance(v, dict):
                for subk, subv in _flatten_dict_keys(v):
                    yield (k + "." + subk, subv)
            else:
                yield (k, v)

    params_dict_flattened = dict(_flatten_dict_keys(params_dict))
    for name, data in dummy_input.items():
        input_names.append(name)
        tensor_dict[name] = data

    for name, data in params_dict_flattened.items():
        param_names.append(name)
        tensor_dict[name] = data

    i = 0

    def make_name():
        nonlocal i
        i += 1
        return "node_" + str(i)

    def tensor_to_name(tensor):
        return {tensor.id: name for name, tensor in tensor_dict.items()}[tensor.id]

    def add_node_from_pg_tensor(tensor):
        name = make_name()
        tensor_dict[name] = tensor

        children = tensor.children()
        inputs = []

        for child in children:
            try:
                inputs.append(tensor_to_name(child))
            except KeyError:
                # must be an input
                trace_inputs = traced.get_last_trace().input_tensors
                for idx, trace_input in enumerate(trace_inputs):
                    if child.id == trace_input.id:
                        inputs.append((input_names + param_names)[idx])
                        break
        primitive = tensor.ad_context()

        prim_to_onnx = {
            "MatMul": "Gemm",
            "Relu": "Relu",
            "Reshape": "Reshape",
            "Permute": "Transpose",
            "Broadcast": "Identity",  # TODO
            "Add": "Add",
            "Max": "Max",
        }

        if primitive in prim_to_onnx or primitive.startswith("Fill"):
            # if is fill (starts with fill) then it is a constant
            if primitive.startswith("Fill"):
                data = tensor.numpy()
                if data.shape == ():
                    data = data.reshape(
                        (1,)
                    )  # make it a 1d tensor (onnx does not support scalars)
                add_initializer(name, data, data.shape, pg_dt_to_onnx_dt[tensor.dtype])
            elif primitive == "Reshape":
                shape = tensor.shape
                shapenp = np.array(shape, dtype=np.int64)
                # add CONSTANT node for shape
                node = onnx.helper.make_node(
                    "Constant",
                    [],
                    [name + "_shape"],
                    value=onnx.helper.make_tensor(
                        name + "_shape", onnx.TensorProto.INT64, (len(shape),), shapenp
                    ),
                )
                nodes.append(node)
                nodes.append(
                    onnx.helper.make_node(
                        prim_to_onnx[primitive], [inputs[0], name + "_shape"], [name]
                    )
                )
            else:
                nodes.append(
                    onnx.helper.make_node(prim_to_onnx[primitive], inputs, [name])
                )
        else:
            raise ValueError(f"Primitive {primitive} not supported for ONNX export")

    def add_input(name, shape, dtype):
        inputs.append(onnx.helper.make_tensor_value_info(name, dtype, shape))

    def add_output(name, shape, dtype):
        outputs.append(onnx.helper.make_tensor_value_info(name, dtype, shape))

    def add_initializer(name, data, shape, dtype):
        tensor = onnx.helper.make_tensor(name, dtype, shape, data)
        initializers.append(tensor)

    def add_value_info(name, shape, dtype):
        value_info.append(onnx.helper.make_tensor_value_info(name, dtype, shape))

    output = traced(dummy_input, params_dict)

    # add params
    for name, data in params_dict_flattened.items():
        add_initializer(name, data.numpy(), data.shape, pg_dt_to_onnx_dt[data.dtype])

    topo_recurse_until_reach_inputs(
        traced.get_last_trace().outputs,
        add_node_from_pg_tensor,
        traced.get_last_trace().input_tensors,
        do_for_input=False,
    )

    # connect output with output from pg

    traced_output = traced.get_last_trace().outputs

    assert len(output) == len(traced_output)

    assert len(output) == len(out_names)
    for tensor, out_name in zip(traced_output, out_names):
        name = tensor_to_name(tensor)
        # connect the onnx output with this node
        add_output(out_name, tensor.shape, pg_dt_to_onnx_dt[tensor.dtype])
        nodes.append(onnx.helper.make_node("Identity", [name], [out_name]))

    # for everything with 'input' in the name, add it to the inputs
    for name, tensor in dummy_input.items():
        add_input(name, tensor.shape, pg_dt_to_onnx_dt[tensor.dtype])

    graph = onnx.helper.make_graph(
        nodes, "custom_pequegrad_model", inputs, outputs, initializers
    )

    model = onnx.helper.make_model(graph)

    # validate
    onnx.checker.check_model(model)

    return model
