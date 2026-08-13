# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

# BMG-specific skip list for Linux
# These tests hang (timeout >10 minutes) on BMG hardware
# See https://github.com/intel/torch-xpu-ops/issues/4947

skip_dict = {
    "extended/test_ops_xpu.py": (
        # Multi-head attention forward tests hang on BMG
        "test_forward_ad_nn_functional_multi_head_attention_forward_xpu_float32",
        "test_operator_nn_functional_multi_head_attention_forward_xpu_float32",
    ),
    "test_indexing_xpu.py": (
        # Index add fast path hangs on BMG with float64
        "test_index_add_fast_path_xpu_float64",
    ),
    "test_linalg_xpu.py": (
        # Linalg low-rank and lobpcg tests hang on BMG
        "test_lobpcg_ortho_xpu_float64",
        "test_pca_lowrank_xpu",
        "test_svd_lowrank_xpu_complex128",
        "test_svd_lowrank_xpu_float64",
    ),
    "test_ops_fwd_gradients_xpu.py": (
        # STFT forward-backward gradient test hangs on BMG with complex128
        "test_fn_fwgrad_bwgrad_stft_xpu_complex128",
    ),
    "test_ops_gradients_xpu.py": (
        # STFT backward gradient test hangs on BMG with complex128
        "test_fn_gradgrad_stft_xpu_complex128",
    ),
    "test_sparse_xpu.py": (
        # Sparse binary operations hang on BMG with complex128 and float64
        "test_binary_operation_mul_SparseBSC_xpu_complex128",
        "test_binary_operation_mul_SparseBSC_xpu_float64",
        "test_binary_operation_mul_SparseBSR_xpu_complex128",
        "test_binary_operation_mul_SparseCSC_xpu_complex128",
        "test_binary_operation_mul_SparseCSC_xpu_float64",
        "test_binary_operation_mul_SparseCSR_xpu_complex128",
        "test_binary_operation_mul_SparseCSR_xpu_float64",
        # Sparse index_select and to_sparse hang on BMG with complex128 and float64
        "test_index_select_empty_and_non_contiguous_index_xpu_float64",
        "test_index_select_exhaustive_index_large_xpu_float64",
        "test_to_sparse_xpu_complex128",
        "test_to_sparse_xpu_float64",
    ),
}
