# Copyright 2020-2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

skip_dict = {
    # failed on MTL windows, skip first for Preci
    # test_get_device_properties: memory_clock_rate is 0 on MTL Windows, the
    # ext_intel_memory_clock_rate SYCL aspect is unsupported for this
    # integrated GPU/driver combination (intel/torch-xpu-ops#5472)
    "test_xpu.py": ("test_mem_get_info_xpu", "test_get_device_properties"),
}
