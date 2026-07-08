/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <ATen/core/Tensor.h>

namespace at::native::xpu {

void triangular_solve_sparse_csr_mkl(
    const Tensor& A,
    const Tensor& B,
    Tensor& X,
    bool upper,
    bool transpose,
    bool unitriangular);

} // namespace at::native::xpu
