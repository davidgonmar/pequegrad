from pequegrad.backend.c import (
    compile,
    Tensor,
    sync_cuda_device,
)  # noqa
from contextvars import ContextVar
from .lazyfn import (
    GraphTrace,
    LazyFunction,
    extract_input_tensors as extract_tensors,
    topo_recurse_until_reach_inputs,
    get_random_possible_toposorts,
    bridge_args_to_lazy_fn,
)
from typing import Callable, Tuple
from pequegrad.ops import fill, dt, device

inside_jit = ContextVar("inside_jit", default=False)


def make_pattern_matcher(fn: Callable, example_shapes: Tuple, match_inps=dict()):
    def match(t: Tensor):
        fake_tensors = [
            fill(shape, dt.float32, 0, device.cpu(0)) for shape in example_shapes
        ]
        fake_tensor_ids = [t.id for t in fake_tensors]
        out = fn(*fake_tensors)
        # we will compare the graph structure of out with t
        matched_children = []
        matched_children_out = []
        appended_ids = set()

        def _recursive_match(t, out):
            if out.id in fake_tensor_ids:
                if t.id not in appended_ids:
                    matched_children.append(t)
                    matched_children_out.append(out)
                    appended_ids.add(t.id)
                return True
            if t.ad_context() != out.ad_context():
                return False
            # for dtype, check dtype_to is same
            if "AsType" in t.ad_context():
                dtype_t = t.dtype
                dtype_out = out.dtype
                if dtype_t != dtype_out:
                    return False
            if len(t.children()) != len(out.children()):
                return False
            for i, j in zip(t.children(), out.children()):
                if not _recursive_match(i, j):
                    return False
            return True

        matched = _recursive_match(t, out)
        if not matched:
            return []
        # reorder matched_children so that it matches the order of fake_tensors
        matched_children_ = []
        for i in fake_tensor_ids:
            idx = None
            for j, k in enumerate(matched_children_out):
                if k.id == i:
                    idx = j
                    break
            matched_children_.append(matched_children[idx])
        # additional checks
        for i, child in enumerate(matched_children_):
            if not match_inps.get(i, lambda _: True)(child):
                return []
        return matched_children_

    return match


def make_pattern(name: str, matcher: Callable, converter: Callable, precompute: bool):
    return (name, matcher, converter, precompute)


class jit(LazyFunction):
    def __init__(
        self,
        f,
        assume_static_argnums=None,
        eval_outs=True,
        opts=None,
        custom_patterns=[],
    ):
        super().__init__(f, assume_static_argnums)
        self.opts = opts if opts is not None else {}
        if opts and opts["experimental_toposort_optim"]:
            if not eval_outs:
                raise ValueError("experimental_toposort_optim requires eval_outs=True")
            self.toposort_optim = True
        else:
            self.toposort_optim = False
        self.eval_outs = eval_outs
        self.toposorted_indices = None
        self.custom_patterns = [(pat[1], pat[2], pat[3]) for pat in custom_patterns]

    def _transform_trace(self, trace: GraphTrace) -> GraphTrace:
        # same as autograd, but it just compiles the graph
        new_trace = GraphTrace(
            inputs=trace.inputs,
            inputs_pytree=trace.inputs_pytree,
            input_tensors=trace.input_tensors,
            outputs=trace.outputs,
            outputs_pytree=trace.outputs_pytree,
        )

        compile(extract_tensors(new_trace.outputs), self.opts, self.custom_patterns)

        if self.toposort_optim:
            toposorted_tensors = []

            def _fn(tensor):
                toposorted_tensors.append(tensor)

            output_tensors = extract_tensors(new_trace.outputs)
            topo_recurse_until_reach_inputs(
                output_tensors, _fn, inputs=new_trace.input_tensors
            )
            self.toposorted_indices = range(len(toposorted_tensors))
            res = [
                get_random_possible_toposorts(
                    output_tensors, inputs=new_trace.input_tensors
                )
                for i in range(200)
            ]
            # rest is a list of permutations of the toposorted_tensors
            # we want to store the indices
            toposorted_tensors = [t.id for t in toposorted_tensors]
            toposorted_indices = []
            for r in res:
                toposorted_indices.append([toposorted_tensors.index(t.id) for t in r])

            self.toposorted_indices = toposorted_indices
            self.already_benchmarked = False
        return new_trace

    def post_process_outs(self, outs, args, input_tensors):
        if not self.toposort_optim:
            if self.eval_outs:
                outs = [o.eval() if isinstance(o, Tensor) else o for o in outs]
            return outs
        else:
            toposorted_tensors = []

            def _fn(tensor):
                toposorted_tensors.append(tensor)

            output_tensors = extract_tensors(
                [out for out in outs if isinstance(out, Tensor)]
            )
            topo_recurse_until_reach_inputs(output_tensors, _fn, inputs=input_tensors)
            del output_tensors
            import gc

            gc.collect()
            list(
                map(lambda x: x.eval(), input_tensors)
            )  # make sure inputs are evaluated
            if not self.already_benchmarked:
                totry = self.toposorted_indices
                times = []
                import time

                for i in totry:
                    start = time.time()
                    trace = self._get_maybe_cached_transformed_trace(args)
                    bridge_args_to_lazy_fn(trace.input_tensors, input_tensors)
                    outs = trace.outputs
                    list(map(lambda x: x.eval() if isinstance(x, Tensor) else x, outs))
                    sync_cuda_device()
                    end = time.time()
                    times.append(end - start)
                    print(f"Time: {end-start}")
                self.toposorted_indices = totry[times.index(min(times))]
                self.already_benchmarked = True

            for i in self.toposorted_indices:
                t = toposorted_tensors[i]
                if not t.is_evaled():
                    t._eval_assume_inputs_evaled()
                    # remove the tensor from the list (at idx)
                    toposorted_tensors[i] = None
            return outs
