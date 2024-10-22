import onnx
import pequegrad as pg
import numpy as np

onnx_dt_to_pg_dt = {
    onnx.TensorProto.FLOAT: pg.dt.float32,
    onnx.TensorProto.INT32: pg.dt.int32,
}

onnx_dt_to_np_dt = {
    onnx.TensorProto.FLOAT: np.float32,
    onnx.TensorProto.INT32: np.int32,
    onnx.TensorProto.INT64: np.int64,
}


def handle_gemm(op: onnx.NodeProto, tensor_dict: dict):
    transA = None
    transB = None
    for attr in op.attribute:
        if attr.name == "transA":
            transA = attr.i
        if attr.name == "transB":
            transB = attr.i

    x, w, b = op.input if len(op.input) == 3 else op.input + [None]
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
            npbuff = np.frombuffer(
                tensor.raw_data, dtype=onnx_dt_to_np_dt[tensor.data_type]
            )
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


op_dict = {
    "Gemm": handle_gemm,
    "Constant": handle_constant,
    "Relu": handle_relu,
    "Reshape": handle_reshape,
}


class OnnxModel:
    def __init__(self, model_proto: onnx.ModelProto):
        self.graph = model_proto.graph
        self.param_dict = {}
        for tensor in self.graph.initializer:
            self.param_dict[tensor.name] = pg.Tensor(
                np.frombuffer(
                    tensor.raw_data, dtype=onnx_dt_to_np_dt[tensor.data_type]
                ).reshape(tensor.dims)
            )

    @pg.jit.withargs(
        assume_static_argnums=(0,), opts={"fuser": False, "common_subexpr_elim": False}
    )
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
