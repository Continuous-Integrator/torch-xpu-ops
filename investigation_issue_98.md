# Investigation: Issue #98 — StateDictStager pin_memory failure on XPU

## Background Concepts

### Pinned Memory

Normal CPU memory is **pageable** — the OS can swap it to disk at any time. When an accelerator (GPU/XPU) wants to transfer data from CPU to device, the driver must first copy from pageable memory to an internal pinned buffer, then DMA to the device. Two copies.

**Pinned (page-locked) memory** tells the OS: "never swap this out." The accelerator can DMA directly from/to it. One copy. Faster transfers.

There are two ways to get pinned memory:

1. **Allocate as pinned from the start** — ask a special allocator to give you page-locked memory.
   - CUDA: `cudaHostAlloc()`
   - SYCL/XPU: `sycl::aligned_alloc_host()`

2. **Register existing memory in-place** — take a regular `malloc`'d pointer and tell the driver "lock these pages."
   - CUDA: `cudaHostRegister(ptr, size)` / `cudaHostUnregister(ptr)`
   - SYCL/XPU: **No equivalent exists.**

### Shared Memory

**Shared memory** means the memory region is accessible from multiple OS processes (via POSIX shared memory or memory-mapped files). This is used when multiple training processes on the same machine need to access the same staged checkpoint data without extra copies.

In PyTorch: `storage._new_shared(size)` allocates in a shared memory region. `tensor.is_shared()` checks if the underlying storage lives in such a region.

### Caching (in StateDictStager context)

During training, checkpoints are saved periodically (e.g., every 1000 steps). Each save requires copying model state from accelerator to CPU. **Caching** means the `StateDictStager` allocates CPU buffers once and reuses them across saves, avoiding repeated allocation/deallocation of large buffers.

```
Step 1000: allocate 10GB CPU buffer -> copy -> write to disk -> keep buffer
Step 2000: reuse same buffer -> copy -> write to disk -> keep buffer
Step 3000: reuse same buffer -> copy -> write to disk -> keep buffer
```

### Why caching forced the `cudaHostRegister` approach

The stager needs ONE buffer that is:
1. Allocated once (cached across saves)
2. Possibly shared (accessible from other processes)
3. Pinned (fast DMA from accelerator)

Using `tensor.pin_memory()` would create a **new** buffer each time — breaking caching. The CUDA developers needed to pin an **existing** cached buffer in-place, hence they used `cudaHostRegister` exposed via `torch.cuda._pin_memory_utils`.

---

## Timeline

### April 2024 — XPU gets pinned memory support

**PR #123080** (`a0466061e`) — "Support xpu host allocator"

- Introduces `XPUCachingHostAllocator` using `sycl::aligned_alloc_host()`
- Enables `tensor.pin_memory()` for XPU
- Uses allocate-new approach (no in-place registration)
- Commit message: *"Following CUDA, this PR adds a new API `getPinnedMemoryAllocator` to support the tensor's memory pinned."*

### December 2024 — StateDictStager introduced (CUDA-only)

**PR #155192** (`19ffdf4ea`) — "[dcp] add new checkpoint staging to preserve storage sharing and support mutable state_dicts"

- Author: Teja Rao (Meta)
- Introduces `StateDictStager` class
- Introduces `torch/cuda/_pin_memory_utils.py` (wraps `cudaHostRegister`/`cudaHostUnregister`)
- Introduces `test/distributed/checkpoint/test_state_dict_stager.py`
- **Everything is CUDA-only**: test uses `@requires_cuda` and `.cuda()`
- The stager's `__init__` checks `torch.cuda.is_available()` to gate pinning
- Pinning uses CUDA-specific `cudaHostRegister` for in-place registration

### January 2025 — Test generalized for all accelerators

**PR #159242** (`8f83b3e71`) — "add device generalization support for distributed checkpoint tests"

- Changes test decorator: `@requires_cuda` -> `@unittest.skipIf(not HAS_ACCELERATOR, ...)`
- Changes tensor creation: `.cuda()` -> `.to(device_type)`
- **Does NOT change `_state_dict_stager.py`** -- the implementation remains CUDA-hardcoded
- The test now runs on XPU, but the code it tests still only works with CUDA

### May 2025 — Issue #98 reported

- Test runs on XPU
- `StateDictStager.__init__` sees `torch.cuda.is_available() == False`, disables `pin_memory`
- Test asserts `is_pinned() == True`, gets `False`
- `AssertionError: Booleans mismatch: False is not True`

---

## The Test: `test_tensor_pinned_and_shared`

### What it does

1. Creates two tensors on the accelerator (XPU in our case)
2. Iterates over 4 configurations of `(pin_memory, share_memory)`:
   - `(False, False)` — no optimizations
   - `(True, False)` — pinned only
   - `(False, True)` — shared only
   - `(True, True)` — both pinned and shared
3. For each, creates a `StateDictStager` and calls `.stage(state_dict)`
4. Asserts the resulting CPU tensors have the expected `is_pinned()` and `is_shared()` status

### Which subtests fail on XPU

| pin_memory | share_memory | Expected is_pinned | Actual | Result |
|-----------|-------------|-------------------|--------|--------|
| False | False | False | False | PASS |
| True | False | True | **False** | **FAIL** |
| False | True | False | False | PASS |
| True | True | True | **False** | **FAIL** |

Both `pin_memory=True` cases fail.

---

## Root Cause Analysis

There are **two layers** of failure:

### Layer 1: The availability check (line 36 of `_state_dict_stager.py`)

```python
if pin_memory and not torch.cuda.is_available():
    warnings.warn("Ignoring pin_memory flag...")
    self.pin_memory = False
```

On XPU-only systems, `torch.cuda.is_available()` is `False`, so `self.pin_memory` is forced to `False`. Pinning is never even attempted.

### Layer 2: The pinning implementation (line 170)

Even if we bypass the availability check, the actual pinning uses CUDA-specific APIs:

```python
import torch.cuda._pin_memory_utils as pin_memory_utils
...
pin_memory_utils.pin_memory(new_storage.data_ptr(), new_storage.nbytes())
```

Which internally calls:
```python
def pin_memory(data_ptr, size):
    cudart = torch.cuda.cudart()
    cudart.cudaHostRegister(data_ptr, size, 1)
```

This would crash on XPU with `AssertionError: Torch not compiled with CUDA enabled`.

---

## Why XPU Can't Replicate CUDA's Approach

CUDA's `cudaHostRegister` is a driver-level feature that takes **any existing pointer** and page-locks it. This allows:
- Allocating shared memory first (`_new_shared()`)
- Then pinning that same memory in-place
- Result: one buffer that is both shared AND pinned

SYCL (which XPU uses) has **no equivalent**. The only way to get pinned memory is to allocate it as pinned from the start via `sycl::aligned_alloc_host()`. There is no "register existing memory" operation in the SYCL spec or Level Zero API.

### Consequence for the four test cases

| pin | share | CUDA approach | XPU feasibility |
|-----|-------|--------------|-----------------|
| F | F | Regular storage, just copy | Works |
| T | F | Regular storage -> `cudaHostRegister` in-place | Possible: allocate from pinned allocator instead |
| F | T | `_new_shared()` storage | Works |
| T | T | `_new_shared()` -> `cudaHostRegister` in-place (both!) | **Not possible**: can't pin shared memory without in-place registration |

---

## Available Fix Options

### Option A: Allocate pinned from the start (no in-place registration needed)

For the `pin_memory=True, share_memory=False` case, instead of allocating regular storage and trying to register it, allocate from the XPU pinned memory allocator upfront. This works for 3 out of 4 cases.

For the `pin_memory=True, share_memory=True` case, we must choose one or the other (shared OR pinned, not both) on XPU.

### Option B: Request Level Zero extension

Ask the Intel compute runtime team to implement something like `zeMemHostRegister` (analogous to `cudaHostRegister`). This would give full parity with CUDA.

### Option C: Adjust test expectations for non-CUDA backends

Update the test to accept that non-CUDA backends may not support pin+share simultaneously, while still testing that pinning works in isolation.

---

## SYCL Memory Model Reference

For context, SYCL provides three types of USM (Unified Shared Memory):

| SYCL API | Accessible from | PyTorch equivalent |
|----------|----------------|-------------------|
| `sycl::aligned_alloc_device` | Device only | `torch.empty(device='xpu')` |
| `sycl::aligned_alloc_host` | CPU, device can DMA | Pinned memory (`tensor.pin_memory()`) |
| `sycl::aligned_alloc_shared` | Both (auto-migrates) | Managed/unified memory (not used here) |

XPU's `CachingHostAllocator` uses `aligned_alloc_host` — this is the correct choice for pinned memory used in staging.
