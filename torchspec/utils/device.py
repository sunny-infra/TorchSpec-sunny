# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Accelerator device helpers (CUDA / Ascend NPU)."""

from __future__ import annotations

import torch


def get_accelerator_device_type() -> str:
    """Return ``\"cuda\"``, ``\"npu\"``, or ``\"cpu\"`` for the current process."""
    if torch.cuda.is_available():
        return "cuda"
    npu = getattr(torch, "npu", None)
    if npu is not None and npu.is_available():
        return "npu"
    return "cpu"


def get_accelerator_module():
    """Return ``torch.cuda`` or ``torch.npu``.

    Raises:
        RuntimeError: If neither CUDA nor NPU is available.
    """
    device_type = get_accelerator_device_type()
    if device_type == "cuda":
        return torch.cuda
    if device_type == "npu":
        return torch.npu
    raise RuntimeError("No CUDA or NPU accelerator is available")


def current_accelerator_device() -> torch.device:
    """Return a ``torch.device`` for the current accelerator ordinal."""
    device_type = get_accelerator_device_type()
    if device_type == "cpu":
        return torch.device("cpu")
    return torch.device(device_type, get_accelerator_module().current_device())


def set_accelerator_device(local_device_id: int) -> None:
    """Bind the current process to a local accelerator ordinal."""
    get_accelerator_module().set_device(local_device_id)


def accelerator_event(*args, **kwargs):
    """Construct a timing Event on the current accelerator backend."""
    return get_accelerator_module().Event(*args, **kwargs)


def accelerator_current_stream(device=None):
    """Return the current stream for *device* on the active accelerator."""
    module = get_accelerator_module()
    if device is None:
        return module.current_stream()
    return module.current_stream(device)


def default_distributed_backend() -> str:
    """Pick a process-group backend matching the available accelerator."""
    device_type = get_accelerator_device_type()
    if device_type == "npu":
        return "hccl"
    if device_type == "cuda":
        return "nccl"
    return "gloo"
