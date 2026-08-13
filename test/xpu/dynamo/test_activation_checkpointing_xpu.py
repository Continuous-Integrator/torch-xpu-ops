# Owner(s): ["module: dynamo"]
import sys
import unittest

import torch
import torch._dynamo
import torch._dynamo.config
import torch._dynamo.test_case
import torch.nn as nn
from torch.testing._internal.inductor_utils import GPU_TYPE


@unittest.skipIf(
    not torch.get_device_module(GPU_TYPE).is_available(), "GPU not available"
)
class RematerializeACNodesPassTests(torch._dynamo.test_case.TestCase):
    """Tests for AC reordering optimization in full graph (forward+backward in one graph)."""

    @torch._dynamo.config.patch(skip_fwd_side_effects_in_bwd_under_checkpoint=True)
    def test_attr_compile_submodules_in_checkpoint_wrapper(self):
        """Compiling submodules inside a checkpointed block should not hit the
        recompile limit due to WeakKeyDictionary guards in the pack_hook."""
        from torch.utils.checkpoint import checkpoint

        class Block(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.norm1 = nn.RMSNorm(dim)
                self.linear1 = nn.Linear(dim, dim, bias=False)
                self.norm2 = nn.RMSNorm(dim)
                self.linear2 = nn.Linear(dim, dim, bias=False)
                self.norm3 = nn.RMSNorm(dim)
                self.linear3 = nn.Linear(dim, dim, bias=False)

            def forward(self, x):
                x = x + self.linear1(self.norm1(x))
                x = x + self.linear2(self.norm2(x))
                x = x + self.linear3(self.norm3(x))
                return x

        class CheckpointedBlock(nn.Module):
            def __init__(self, block):
                super().__init__()
                self.block = block

            def forward(self, x):
                return checkpoint(self.block, x, use_reentrant=False)

        # Initialize the device before checkpoint to avoid device state initialization error
        torch.get_device_module(GPU_TYPE)._lazy_init()

        dim = 32
        block = Block(dim).to(GPU_TYPE)

        x_ref = torch.randn(4, dim, device=GPU_TYPE, requires_grad=True)
        ref = block(x_ref)
        ref.sum().backward()

        block_cp = Block(dim).to(GPU_TYPE)
        block_cp.load_state_dict(block.state_dict())
        wrapped = CheckpointedBlock(block_cp)

        for _, submod in wrapped.block.named_children():
            submod.compile(backend="aot_eager")

        with torch._dynamo.config.patch(recompile_limit=2):
            x_test = x_ref.detach().clone().requires_grad_(True)
            result = wrapped(x_test)
            result.sum().backward()

        self.assertEqual(ref, result)
        self.assertEqual(x_ref.grad, x_test.grad)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
