# Owner(s): ["module: dynamo"]
import functools
import unittest

import torch
import torch._dynamo.test_case
import torch.nn.functional as F
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_CUDNN_ATTENTION,
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests


def _grad(*args, **kwargs):
    return torch.autograd.grad(*args, **kwargs)


GPU_TYPE = "xpu"


class RematerializeACNodesPassTests(torch._dynamo.test_case.TestCase):
    def _compile_and_capture(self, fn, enable_pass, inputs):
        from torch._dynamo.backends.common import aot_autograd
        from torch._dynamo.testing import AotEagerAndRecordGraphs

        def fw_compiler(gm, example_inputs):
            if enable_pass:
                from torch._inductor.fx_passes.joint_graph import joint_graph_passes

                joint_graph_passes(gm)
            return gm.forward

        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=None,
            partition_fn=None,
        )

        with torch._dynamo.config.patch(trace_autograd_ops=True):
            compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
            result = compiled_fn(*inputs)

            # Also capture the graph with trace_autograd_ops enabled
            graphs = AotEagerAndRecordGraphs()
            torch.compile(fn, backend=graphs, fullgraph=True)(*inputs)

        return result, graphs.graphs[0] if graphs.graphs else None

    def count_op(self, gm, op):
        if gm is None:
            return 0
        count = 0
        for node in gm.graph.nodes:
            if node.target == op:
                count += 1
        return count

    @unittest.skipIf(
        not torch.get_device_module(GPU_TYPE).is_available(), "XPU not available"
    )
    @unittest.skip("XPU has known limitations on SDPA backends - see issue #133")
    def test_ac_rematerialize_with_sdpa_dropout_zero(self):
        from torch.nn.attention import sdpa_kernel, SDPBackend

        cases = []
        if PLATFORM_SUPPORTS_MEM_EFF_ATTENTION:
            cases.append((SDPBackend.EFFICIENT_ATTENTION, torch.float32))
        if PLATFORM_SUPPORTS_FLASH_ATTENTION:
            cases.append((SDPBackend.FLASH_ATTENTION, torch.float16))
        if PLATFORM_SUPPORTS_CUDNN_ATTENTION:
            cases.append((SDPBackend.CUDNN_ATTENTION, torch.float16))
        if not cases:
            self.skipTest("No fused SDPA backends available")
        sdpa_ops = {
            torch.ops.aten.scaled_dot_product_attention.default,
            torch.ops.aten._scaled_dot_product_cudnn_attention.default,
            torch.ops.aten._scaled_dot_product_flash_attention.default,
            torch.ops.aten._scaled_dot_product_efficient_attention.default,
            torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
        }

        def policy_fn(ctx, op, *args, **kwargs):
            if op in sdpa_ops:
                return torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE
            return torch.utils.checkpoint.CheckpointPolicy.PREFER_SAVE

        context_fn = functools.partial(
            torch.utils.checkpoint.create_selective_checkpoint_contexts, policy_fn
        )

        for backend, dtype in cases:
            # Fix: Convert enum to string to avoid pytest-xdist serialization error
            with self.subTest(backend=str(backend), dtype=dtype):
                torch._dynamo.reset()
                q = torch.randn(
                    2, 4, 128, 64, device=GPU_TYPE, dtype=dtype, requires_grad=True
                )
                k = torch.randn(
                    2, 4, 128, 64, device=GPU_TYPE, dtype=dtype, requires_grad=True
                )
                v = torch.randn(
                    2, 4, 128, 64, device=GPU_TYPE, dtype=dtype, requires_grad=True
                )

                def fwd_bwd_with_sdpa(q, k, v):
                    with sdpa_kernel(backend):
                        z = torch.utils.checkpoint.checkpoint(
                            lambda q, k, v: F.scaled_dot_product_attention(
                                q, k, v, dropout_p=0.0
                            ),
                            q,
                            k,
                            v,
                            use_reentrant=False,
                            context_fn=context_fn,
                        )
                        loss = z.sum()
                        dq, dk, dv = _grad(loss, (q, k, v))

                    return z.detach(), dq, dk, dv

                result_with, gm_with = self._compile_and_capture(
                    fwd_bwd_with_sdpa, True, (q, k, v)
                )
                torch._dynamo.reset()
                result_without, _ = self._compile_and_capture(
                    fwd_bwd_with_sdpa, False, (q, k, v)
                )
                eager_inputs = tuple(
                    t.detach().clone().requires_grad_(True) for t in (q, k, v)
                )
                result_eager = fwd_bwd_with_sdpa(*eager_inputs)

                for actual, expected in zip(result_with, result_without):
                    self.assertEqual(actual, expected)
                for actual, expected in zip(result_with, result_eager):
                    self.assertEqual(actual, expected)
                self.assertEqual(sum(self.count_op(gm_with, op) for op in sdpa_ops), 2)


instantiate_device_type_tests(
    RematerializeACNodesPassTests, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
