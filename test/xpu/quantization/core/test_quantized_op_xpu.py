# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# Owner(s): ["module: intel"]
import itertools

import torch
from torch.nn.modules.utils import _pair
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase

try:
    from xpu_test_utils import XPUPatchForImport
except Exception as e:
    import os
    import sys

    script_path = os.path.split(__file__)[0]
    sys.path.insert(0, os.path.realpath(os.path.join(script_path, "../..")))
    from xpu_test_utils import XPUPatchForImport

with XPUPatchForImport(False):
    from test_quantized_op import TestQuantizedOps


def _test_max_pool2d_pt2e(self):
    kernel_list = [2, 3]
    stride_list = [1, 2]
    padding_list = [0, 2]
    dilation_list = [1, 2]
    ceil_mode_list = [False, True]
    channels_last_input = [False, True]
    options = itertools.product(
        kernel_list,
        stride_list,
        padding_list,
        dilation_list,
        ceil_mode_list,
        channels_last_input,
    )
    for kernel, stride, padding, dilation, ceil_mode, channels_last in options:
        if padding >= (kernel // 2):
            # Continue with invalid input
            continue
        device = torch.device("xpu:0")
        input = torch.randint(0, 8, (1, 3, 8, 8), dtype=torch.uint8, device=device)
        if channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        a_pool = torch.nn.functional.max_pool2d(
            input.to(torch.float32),
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
        ).to(torch.uint8)
        a_hat = torch.ops.quantized.max_pool2d(
            input,
            kernel_size=_pair(kernel),
            stride=_pair(stride),
            padding=_pair(padding),
            dilation=_pair(dilation),
            ceil_mode=ceil_mode,
        )
        self.assertEqual(
            input.is_contiguous(),
            a_hat.is_contiguous(),
            msg="ops.quantized.max_pool2d input output diff memory format",
        )
        self.assertEqual(a_pool, a_hat, msg="ops.quantized.max_pool2d results are off")


TestQuantizedOps.test_max_pool2d_pt2e = _test_max_pool2d_pt2e


def _test_qsoftmax_qnnpack(self):
    """Override to inline the test logic since test_qsoftmax becomes test_qsoftmax_xpu."""
    import numpy as np
    import hypothesis.strategies as st
    from torch.testing._internal.common_quantized import _quantize, override_quantized_engine

    with override_quantized_engine('qnnpack'):
        # Inline the test_qsoftmax logic with sample dims
        dims = [3, 3, 3, 3, 3]  # Use a representative sample
        for (num_dims, dim, memory_format) in [
            (2, 1, torch.contiguous_format),
            (4, 3, torch.contiguous_format),
            (5, 2, torch.contiguous_format),
            (4, 3, torch.channels_last),
            (4, 1, torch.channels_last),
            (5, 1, torch.channels_last_3d),
        ]:
            size = dims[:num_dims]
            torch_dtype = torch.quint8
            np_dtype = np.uint8

            scale_X = 1.3
            zero_point_X = 5
            X = torch.rand(size=size, dtype=torch.float32) * 8 + zero_point_X
            X = X.to(memory_format=memory_format)

            scale_Y = 1 / 256
            zero_point_Y = 0

            qX = torch.quantize_per_tensor(X,
                                           scale=scale_X,
                                           zero_point=zero_point_X,
                                           dtype=torch_dtype)

            Y = torch.softmax(qX.dequantize(), dim=dim).numpy()
            qY = _quantize(Y, scale_Y, zero_point_Y, dtype=np_dtype)
            qY_hat = torch.ops.quantized.softmax(qX,
                                                 dim=dim,
                                                 output_scale=scale_Y,
                                                 output_zero_point=zero_point_Y)

            np.testing.assert_equal(qY, qY_hat.int_repr(),
                                    "Quantized softmax failed.")


TestQuantizedOps.test_qsoftmax_qnnpack = _test_qsoftmax_qnnpack

instantiate_device_type_tests(
    TestQuantizedOps, globals(), only_for="xpu", allow_xpu=True
)

if __name__ == "__main__":
    TestCase._default_dtype_check_enabled = True
    run_tests()
