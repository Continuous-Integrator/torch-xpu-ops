# Owner(s): ["module: dynamo"]
import copy
import functools
import io
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import contextmanager
from unittest import skipIf

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "..", "..", "test", "dynamo"
        )
    )
)

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._logging.structured
import torch.distributed as dist
import torch.fx as fx
from torch._inductor.test_case import TestCase
from torch._logging._internal import TorchLogsFormatter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_utils import find_free_port
from torch.testing._internal.inductor_utils import HAS_XPU_AND_TRITON
from torch.testing._internal.triton_utils import requires_gpu_and_triton


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


if torch.distributed.is_available():
    from torch.testing._internal.distributed.fake_pg import FakeStore

HAS_TLPARSE = shutil.which("tlparse") is not None
requires_tlparse = unittest.skipUnless(HAS_TLPARSE, "requires tlparse")
requires_distributed = functools.partial(
    unittest.skipIf, not dist.is_available(), "requires distributed"
)


def example_fn(a):
    output = a.mul(torch.ones(1000, 1000))
    output = output.add(torch.ones(1000, 1000))
    return output


def example_training_fn(a):
    output = a.mul(torch.ones(1000, 1000, requires_grad=True))
    output = output.add(torch.ones(1000, 1000))
    output.sum().backward()
    return output


def dynamo_error_fn(a):
    output = a.mul(torch.ones(1000, 1000))
    output = output.add(torch.ones(10, 10))
    return output


def inductor_error_fn(a):
    output = torch.round(a)
    return output


def inductor_schedule_fn(a):
    output = a.add(torch.ones(1000, 1000, device=device_type))
    return output


ARGS = (torch.ones(1000, 1000, requires_grad=True),)


def replace_dynamic(buffer, key):
    return re.sub(r'("' + key + r'":\s*)(\d+\.\d+)', r"\1<dynamic>", buffer)


class StructuredTraceTestingFilter(logging.Filter):
    def __init__(self, match_name=None):
        self.match_name = match_name

    def filter(self, record):
        if "str" in record.metadata:
            return False
        # torch_version is a global artifact emitted once per process the first
        # time the trace handler initializes. Its presence (and commit-hash
        # payload) depends on run context and test ordering, so drop it here to
        # keep each test's expected inline output deterministic.
        if (
            "artifact" in record.metadata
            and record.metadata["artifact"].get("name") == "torch_version"
        ):
            return False
        if self.match_name is not None:
            if "artifact" in record.metadata:
                if self.match_name != record.metadata["artifact"]["name"]:
                    return False
            elif self.match_name not in record.metadata:
                return False
        return True


class ChromiumEventFilter(logging.Filter):
    def filter(self, record):
        return "chromium_event" not in record.metadata


class StructuredTracePayloadFormatter(logging.Formatter):
    def format(self, record):
        return record.payload.strip()


class _DescribeIdNormalizer:
    def __init__(self):
        self._tensor_id_remap = {}
        self._storage_id_remap = {}
        self._next_tensor_id = 0
        self._next_storage_id = 0

    def normalize(self, metadata):
        if "describe_storage" in metadata:
            storage_meta = metadata["describe_storage"]
            if (storage_id := storage_meta.get("id")) is not None:
                storage_meta["id"] = self._normalize_storage_id(storage_id)
            storage_meta["describer_id"] = "ID"
        if "describe_tensor" in metadata:
            tensor_meta = metadata["describe_tensor"]
            if (tensor_id := tensor_meta.get("id")) is not None:
                tensor_meta["id"] = self._normalize_tensor_id(tensor_id)
            if (storage_id := tensor_meta.get("storage")) is not None:
                tensor_meta["storage"] = self._normalize_storage_id(storage_id)
            tensor_meta["describer_id"] = "ID"
            if "view_func" in tensor_meta:
                tensor_meta["view_func"] = "VIEW_FUNC"
        if "describe_source" in metadata:
            source_meta = metadata["describe_source"]
            if (source_id := source_meta.get("id")) is not None:
                source_meta["id"] = self._normalize_tensor_id(source_id)
            source_meta["describer_id"] = "ID"
        return metadata

    def _normalize_tensor_id(self, original_id):
        if original_id not in self._tensor_id_remap:
            self._tensor_id_remap[original_id] = self._next_tensor_id
            self._next_tensor_id += 1
        return self._tensor_id_remap[original_id]

    def _normalize_storage_id(self, original_id):
        if original_id not in self._storage_id_remap:
            self._storage_id_remap[original_id] = self._next_storage_id
            self._next_storage_id += 1
        return self._storage_id_remap[original_id]


class StructuredTraceTestingFormatter(logging.Formatter):
    def __init__(self):
        super().__init__()
        self._id_normalizer = _DescribeIdNormalizer()

    def format(self, record):
        metadata = copy.deepcopy(record.metadata)

        # Stub out values that are not stable across runs
        # TODO: Check that these match schema
        if "has_payload" in metadata:
            metadata["has_payload"] = "HASH"
        if "dynamo_start" in metadata:
            metadata["dynamo_start"]["stack"] = "STACK"
        if "inductor_output_code" in metadata:
            metadata["inductor_output_code"]["filename"] = "FILENAME"
            if "file_path" in metadata["inductor_output_code"]:
                metadata["inductor_output_code"]["file_path"] = "FILENAME"
        if "stack" in metadata:
            metadata["stack"] = "STACK"
        if "compilation_metrics" in metadata:
            metadata["compilation_metrics"] = "METRICS"
        if "bwd_compilation_metrics" in metadata:
            metadata["bwd_compilation_metrics"] = "METRICS"
        if "compilation_metrics_runtime" in metadata:
            metadata["compilation_metrics_runtime"] = "METRICS"
        if "bwd_compilation_metrics_runtime" in metadata:
            metadata["bwd_compilation_metrics_runtime"] = "METRICS"
        metadata = self._id_normalizer.normalize(metadata)
        if (
            (k := "create_symbol") in metadata
            or (k := "guard_added_fast") in metadata
            or (k := "create_unbacked_symbol") in metadata
        ):
            metadata[k]["user_stack"] = "STACK"
            metadata[k]["stack"] = "STACK"

        if "dump_file" in metadata:
            # Don't include the actually key number, that's sensitive to other
            # test runs
            metadata["dump_file"]["name"] = "<eval_with_key>"
            return (
                json.dumps(metadata)
                + "\n"
                + "\n".join(l.rstrip() for l in record.payload.splitlines())
            )

        return json.dumps(metadata)


trace_log = logging.getLogger("torch.__trace")

chrome_event_filter = ChromiumEventFilter()


def show_chrome_events(fn):
    """
    Don't hide chrome events for this test
    """

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        self.handler.removeFilter(chrome_event_filter)
        return fn(self, *args, **kwargs)

    return wrapper


class StructuredTraceTest(TestCase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        torch._logging.structured.INTERN_TABLE.clear()
        self.buffer = io.StringIO()
        self.old_level = trace_log.level
        trace_log.setLevel(logging.DEBUG)

        self.handler = logging.StreamHandler(self.buffer)
        self.handler.setFormatter(StructuredTraceTestingFormatter())
        self.handler.addFilter(StructuredTraceTestingFilter())
        self.handler.addFilter(chrome_event_filter)
        trace_log.addHandler(self.handler)

        self.raw_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
            mode="w", delete=True
        )  # set this to False to keep temporary files
        self.raw_handler = logging.StreamHandler(self.raw_file)
        self.raw_handler.setFormatter(TorchLogsFormatter(trace=True))
        trace_log.addHandler(self.raw_handler)

    def tearDown(self):
        trace_log.removeHandler(self.handler)
        trace_log.removeHandler(self.raw_handler)
        self.raw_file.close()
        trace_log.setLevel(self.old_level)
        super().tearDown()

    def assertExpectedInline(self, actual, expected, skip=0):
        super().assertExpectedInline(
            self._normalize_rank_field(self._normalize_describe_ids(actual)),
            self._normalize_rank_field(self._normalize_describe_ids(expected)),
            skip=skip + 1,
        )

    @staticmethod
    def _normalize_rank_field(text):
        if not isinstance(text, str):
            return text
        text = text.replace(', "rank": 0', "")
        text = text.replace('"rank": 0, ', "")
        text = text.replace('"rank": 0', "")
        return text

    @staticmethod
    def _normalize_describe_ids(text):
        if not isinstance(text, str):
            return text
        normalizer = _DescribeIdNormalizer()
        trailing_newline = text.endswith("\n")
        normalized_lines = []
        for line in text.splitlines():
            if not line:
                normalized_lines.append(line)
                continue
            try:
                metadata = json.loads(line)
            except json.JSONDecodeError:
                normalized_lines.append(line)
                continue
            normalized_lines.append(json.dumps(normalizer.normalize(metadata)))
        result = "\n".join(normalized_lines)
        if trailing_newline:
            result += "\n"
        return result

    def assertParses(self):
        if not HAS_TLPARSE:
            self.skipTest("requires tlparse")
        out = tempfile.mkdtemp()
        try:
            subprocess.check_call(
                [
                    "tlparse",
                    "-o",
                    out,
                    "--overwrite",
                    "--no-browser",
                    "--strict",
                    self.raw_file.name,
                ]
            )
        finally:
            shutil.rmtree(out, ignore_errors=True)

    @unittest.skip(
        "XPU: Missing chromium/structured trace events for compiled_autograd_id on XPU - infrastructure issue"
    )
    @requires_tlparse
    @torch._dynamo.config.patch("compiled_autograd", True)
    @torch._inductor.config.patch("fx_graph_cache", True)
    @show_chrome_events
    def test_compiled_autograd_id(self):
        def fn(a):
            return a.sin().sum().backward()

        x = torch.tensor([1.0], requires_grad=True)
        fn_opt = torch._dynamo.optimize("inductor")(fn)
        fn_opt(x)
        torch._dynamo.reset()
        # Reset raw log so assertParses only validates the cache-hit compilation.
        # Without this, the raw log contains two compilations with identical
        # compiled_autograd_id=0 (reset() restarts COMPILE_COUNTER from zero),
        # which tlparse --strict rejects as duplicate compiled_autograd_id.
        trace_log.removeHandler(self.raw_handler)
        self.raw_file.close()
        self.raw_file = tempfile.NamedTemporaryFile(mode="w", delete=True)  # noqa: SIM115
        self.raw_handler = logging.StreamHandler(self.raw_file)
        self.raw_handler.setFormatter(TorchLogsFormatter(trace=True))
        trace_log.addHandler(self.raw_handler)
        # Trigger a cache hit
        fn_opt(x)
        # Verify the cache-hit trace (including inductor_output_code) parses cleanly
        self.assertParses()
        logs = self.buffer.getvalue()
        self.assertRegex(
            logs,
            r'\{"chromium_event": \{\}, "frame_id": \d+, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"\}',
        )
        graph_event = re.search(
            r'\{"compiled_autograd_graph": \{\}, "compiled_autograd_id": (\d+), "attempt": 0, "has_payload": "HASH"\}',
            logs,
        )
        self.assertIsNotNone(graph_event)
        self.assertRegex(
            logs,
            r'\{"chromium_event": \{\}, "compiled_autograd_id": '
            + graph_event.group(1)
            + r', "frame_id": \d+, "frame_compile_id": 0, "attempt": 0, "has_payload": "HASH"\}',
        )

    @unittest.skip(
        "XPU: Missing chromium/structured trace events for compiled_autograd_attribution on XPU - infrastructure issue"
    )
    @requires_tlparse
    @torch._dynamo.config.patch("compiled_autograd", True)
    def test_compiled_autograd_attribution(self):
        # multiple dynamo recompiles should still be attributed to the parent compiled autograd id
        def fn():
            class MySin(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x):
                    ctx.save_for_backward(x)
                    return torch.sin(x)

                @staticmethod
                def backward(ctx, gO):
                    print("graph break")
                    (x,) = ctx.saved_tensors
                    print("graph break")
                    return gO * torch.cos(x)

            grads = []
            for i in [10, 100, 10, 15, 20, 25]:
                x = torch.arange(0.0, i, requires_grad=True)
                out = MySin.apply(x)
                loss = out.sum()
                loss.backward()
                grads.append(x.grad)

            return grads

        fn_opt = torch.compile(fn)  # noqa: UNSPECIFIED_BACKEND
        fn_opt()
        self.assertParses()
        logs = self.buffer.getvalue()
        pattern = (
            r'\{"dynamo_start": \{"stack": "STACK"\}(?:, "compiled_autograd_id": (\d+))?, '
            r'"frame_id": (\d+), "frame_compile_id": 0, "attempt": 0\}'
        )
        starts = re.findall(pattern, logs)
        ca_frames = {}
        for ca_id, frame in starts:
            if ca_id:
                ca_frames.setdefault(ca_id, set()).add(frame)
        self.assertEqual(len(ca_frames), 2)
        first_id, second_id = sorted(ca_frames, key=int)
        self.assertGreaterEqual(len(ca_frames[first_id]), 3)
        self.assertGreaterEqual(len(ca_frames[second_id]), 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
