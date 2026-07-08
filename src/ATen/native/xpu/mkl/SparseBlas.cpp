/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// Sparse triangular solve for XPU using oneapi::mkl::sparse::trsm.
//
// optimize_trsm (wavefront/level-set analysis) is expensive (~3-10x the
// solve itself), so we cache the analysed matrix handle keyed on CSR data
// pointers, shape, and solve configuration.  When the same sparse matrix A
// is reused across calls the analysis is paid only once.
//
// Limitation: optimize_trsm does not support transpose::trans.  For the
// transposed case we fall back to to_dense + triangular_solve_out.

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/mkldnn/xpu/detail/LRUCache.h>
#include <ATen/native/xpu/mkl/SparseBlas.h>
#include <comm/SYCLContext.h>

#include <oneapi/mkl/spblas.hpp>

#include <array>
#include <memory>

namespace at::native::xpu {

namespace {

// ---------------------------------------------------------------------------
// Cache key
// ---------------------------------------------------------------------------

struct SpTrsmKey {
  const void* crow_ptr; // identity proxy for CSR structure
  const void* val_ptr; // identity proxy for CSR values
  int64_t nrows;
  int64_t nnz;
  int64_t nrhs; // optimize_trsm takes nrhs as a hint; cache per nrhs
  at::ScalarType dtype; // must be part of key to avoid type mismatch on reuse
  oneapi::mkl::uplo uplo_val;
  oneapi::mkl::diag diag_val;

  bool operator==(const SpTrsmKey& o) const {
    return crow_ptr == o.crow_ptr && val_ptr == o.val_ptr &&
        nrows == o.nrows && nnz == o.nnz && nrhs == o.nrhs &&
        dtype == o.dtype &&
        uplo_val == o.uplo_val && diag_val == o.diag_val;
  }
};

} // namespace

} // namespace at::native::xpu

namespace std {
template <>
struct hash<at::native::xpu::SpTrsmKey> {
  size_t operator()(const at::native::xpu::SpTrsmKey& k) const noexcept {
    size_t h = 0;
    auto mix = [&](size_t v) {
      h ^= v + 0x9e3779b9 + (h << 6) + (h >> 2);
    };
    mix(reinterpret_cast<uintptr_t>(k.crow_ptr));
    mix(reinterpret_cast<uintptr_t>(k.val_ptr));
    mix(static_cast<size_t>(k.nrows));
    mix(static_cast<size_t>(k.nnz));
    mix(static_cast<size_t>(k.nrhs));
    mix(static_cast<size_t>(k.dtype));
    mix(static_cast<size_t>(k.uplo_val));
    mix(static_cast<size_t>(k.diag_val));
    return h;
  }
};
} // namespace std

namespace at::native::xpu {

namespace {

// ---------------------------------------------------------------------------
// Cached handle
// SpTrsmHandle is move-only (non-copyable RAII), so the cache stores
// shared_ptr<SpTrsmHandle> which is copyable.
// ---------------------------------------------------------------------------

struct SpTrsmHandle {
  oneapi::mkl::sparse::matrix_handle_t handle{nullptr};
  sycl::queue* queue_ptr{nullptr}; // non-owning; identifies the queue

  SpTrsmHandle() = default;
  SpTrsmHandle(const SpTrsmHandle&) = delete;
  SpTrsmHandle& operator=(const SpTrsmHandle&) = delete;
  SpTrsmHandle(SpTrsmHandle&&) = delete;
  SpTrsmHandle& operator=(SpTrsmHandle&&) = delete;

  ~SpTrsmHandle() {
    if (handle && queue_ptr) {
      try {
        oneapi::mkl::sparse::release_matrix_handle(*queue_ptr, &handle)
            .wait();
      } catch (...) {
      }
    }
  }
};

// ---------------------------------------------------------------------------
// Per-device LRU cache
// ---------------------------------------------------------------------------

using SpTrsmCache = at::native::onednn::lru_cache<
    SpTrsmKey,
    std::shared_ptr<SpTrsmHandle>>;

static constexpr int kMaxDevices = 16;
static constexpr int kCacheCapacity = 128;

SpTrsmCache& get_cache(int device_id) {
  static thread_local std::array<SpTrsmCache, kMaxDevices> caches;
  auto& c = caches[device_id];
  if (c.max_size() == 0) {
    c.resize(kCacheCapacity);
  }
  return c;
}

// ---------------------------------------------------------------------------
// Typed core: B_cm and X_cm are column-major [nrows, nrhs], meaning
// element (i,j) is at offset j*nrows+i.  We achieve this by passing
// .t().contiguous() tensors whose shape is [nrhs, nrows] in row-major,
// which has the identical byte layout.
// ---------------------------------------------------------------------------

template <typename scalar_t, typename index_t>
void apply_sparse_trsm(
    sycl::queue& queue,
    int64_t nrows,
    int64_t nnz,
    int64_t nrhs,
    const index_t* d_crow,
    const index_t* d_col,
    const scalar_t* d_val,
    const scalar_t* d_B_cm, // col-major [nrows, nrhs], ld=nrows
    scalar_t* d_X_cm, // col-major [nrows, nrhs], ld=nrows
    at::ScalarType dtype,
    oneapi::mkl::uplo uplo_val,
    oneapi::mkl::diag diag_val,
    int device_id) {
  SpTrsmKey key{d_crow, d_val, nrows, nnz, nrhs, dtype, uplo_val, diag_val};
  auto& cache = get_cache(device_id);

  auto it = cache.find(key);
  if (it == cache.end()) {
    auto entry = std::make_shared<SpTrsmHandle>();
    oneapi::mkl::sparse::init_matrix_handle(&entry->handle);
    entry->queue_ptr = &queue;

    // set_csr_data is asynchronous; chain its event into optimize_trsm.
    auto e_set = oneapi::mkl::sparse::set_csr_data(
        queue,
        entry->handle,
        nrows,
        nrows,
        nnz,
        oneapi::mkl::index_base::zero,
        const_cast<index_t*>(d_crow),
        const_cast<index_t*>(d_col),
        const_cast<scalar_t*>(d_val),
        /*dependencies=*/{});

    auto e_opt = oneapi::mkl::sparse::optimize_trsm(
        queue,
        oneapi::mkl::layout::col_major,
        uplo_val,
        oneapi::mkl::transpose::nontrans,
        diag_val,
        entry->handle,
        nrhs,
        {e_set});
    e_opt.wait();

    cache.insert({key, entry});
    it = cache.find(key);
  }

  scalar_t alpha = scalar_t(1);
  auto e_solve = oneapi::mkl::sparse::trsm(
      queue,
      oneapi::mkl::layout::col_major,
      oneapi::mkl::transpose::nontrans,
      oneapi::mkl::transpose::nontrans,
      uplo_val,
      diag_val,
      alpha,
      it->second->handle,
      d_B_cm,
      nrhs,
      nrows, // ldx
      d_X_cm,
      nrows, // ldy
      /*dependencies=*/{});
  e_solve.wait();
}

} // namespace

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

void triangular_solve_sparse_csr_mkl(
    const Tensor& A,
    const Tensor& B,
    Tensor& X,
    bool upper,
    bool transpose,
    bool unitriangular) {
  // optimize_trsm does not support transpose::trans; fall back to dense.
  if (transpose) {
    Tensor temp_clone_A = at::empty({0}, A.options().layout(at::kStrided));
    at::triangular_solve_out(
        X, temp_clone_A, B, A.to_dense(), upper, transpose, unitriangular);
    return;
  }

  TORCH_CHECK(
      A.scalar_type() == ScalarType::Float ||
          A.scalar_type() == ScalarType::Double,
      "triangular_solve_sparse_csr_mkl: only float32 and float64 are "
      "supported, got ",
      A.scalar_type());

  const int64_t nrows = A.size(0);
  const int64_t nrhs = B.dim() == 1 ? 1 : B.size(1);
  const int64_t nnz = A._nnz();

  auto uplo_val =
      upper ? oneapi::mkl::uplo::upper : oneapi::mkl::uplo::lower;
  auto diag_val =
      unitriangular ? oneapi::mkl::diag::unit : oneapi::mkl::diag::nonunit;

  auto& queue = at::xpu::getCurrentSYCLQueue();
  int device_id = static_cast<int>(A.device().index());

  AT_DISPATCH_FLOATING_TYPES(
      A.scalar_type(), "triangular_solve_sparse_csr_xpu_mkl", [&] {
        // MKL sparse requires int32 indices.
        auto crow = A.crow_indices().to(kInt).contiguous();
        auto col = A.col_indices().to(kInt).contiguous();
        auto val = A.values().to(A.scalar_type()).contiguous();

        // Convert B to column-major: .t().contiguous() gives [nrhs, nrows]
        // row-major whose byte layout equals [nrows, nrhs] col-major.
        Tensor B_2d = B.dim() == 1 ? B.unsqueeze(1) : B;
        Tensor B_cm = B_2d.t().contiguous();
        Tensor X_cm = at::empty_like(B_cm);

        apply_sparse_trsm<scalar_t, int32_t>(
            queue,
            nrows,
            nnz,
            nrhs,
            crow.const_data_ptr<int32_t>(),
            col.const_data_ptr<int32_t>(),
            val.const_data_ptr<scalar_t>(),
            B_cm.const_data_ptr<scalar_t>(),
            X_cm.data_ptr<scalar_t>(),
            A.scalar_type(),
            uplo_val,
            diag_val,
            device_id);

        // X_cm is [nrhs, nrows] row-major = [nrows, nrhs] col-major.
        // X expects [nrows, nrhs] row-major, so copy X_cm.t().
        if (B.dim() == 1) {
          X.copy_(X_cm.squeeze(0));
        } else {
          X.copy_(X_cm.t());
        }
      });
}

} // namespace at::native::xpu
