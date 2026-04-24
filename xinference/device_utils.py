# Copyright 2022-2026 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional, Union

import torch

DeviceType = Literal["cuda", "mps", "xpu", "vacc", "npu", "mlu", "musa", "cpu"]


@dataclass
class DeviceSpec:
    name: str
    is_available: Callable[[], bool]
    env_name: Optional[str]
    device_count_fn: Callable[[], int]
    empty_cache_fn: Optional[Callable] = None
    preferred_dtype: Optional[torch.dtype] = None
    hf_accelerate: bool = False


def _mps_empty_cache():
    try:
        torch.mps.empty_cache()
    except RuntimeError as e:
        if "invalid low watermark ratio" not in str(e):
            raise


def _count_with_visible_devices(env_name: str, device_count: int) -> int:
    env_val = os.getenv(env_name, None)
    if env_val is None:
        return device_count
    visible = env_val.split(",") if env_val else []
    return min(device_count, len(visible))


# ---------------------------------------------------------------------------
# Device availability checks
# ---------------------------------------------------------------------------


def is_vacc_available() -> bool:
    try:
        import torch_vacc  # noqa: F401

        return torch.vacc.is_available()
    except ImportError:
        return False


def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def is_npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except ImportError:
        return False


def is_mlu_available() -> bool:
    try:
        import torch_mlu  # noqa: F401

        return torch.mlu.is_available()
    except ImportError:
        return False


def is_musa_available() -> bool:
    try:
        import torch_musa  # noqa: F401
        import torchada  # noqa: F401

        return torch.musa.is_available()
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Device registry (ordered by detection priority)
# ---------------------------------------------------------------------------

DEVICE_REGISTRY: list[DeviceSpec] = [
    DeviceSpec(
        name="cuda",
        is_available=lambda: torch.cuda.is_available(),
        env_name="CUDA_VISIBLE_DEVICES",
        device_count_fn=lambda: torch.cuda.device_count(),
        empty_cache_fn=lambda: torch.cuda.empty_cache(),
        preferred_dtype=torch.float16,
        hf_accelerate=True,
    ),
    DeviceSpec(
        name="mps",
        is_available=lambda: torch.backends.mps.is_available(),
        env_name=None,
        device_count_fn=lambda: 0,
        empty_cache_fn=_mps_empty_cache,
        preferred_dtype=torch.float16,
    ),
    DeviceSpec(
        name="xpu",
        is_available=is_xpu_available,
        env_name=None,
        device_count_fn=lambda: torch.xpu.device_count(),
        empty_cache_fn=lambda: torch.xpu.empty_cache(),
        preferred_dtype=torch.bfloat16,
        hf_accelerate=True,
    ),
    DeviceSpec(
        name="npu",
        is_available=is_npu_available,
        env_name="ASCEND_RT_VISIBLE_DEVICES",
        device_count_fn=lambda: torch.npu.device_count(),
        empty_cache_fn=lambda: torch.npu.empty_cache(),
        preferred_dtype=torch.float16,
        hf_accelerate=True,
    ),
    DeviceSpec(
        name="mlu",
        is_available=is_mlu_available,
        env_name="MLU_VISIBLE_DEVICES",
        device_count_fn=lambda: torch.mlu.device_count(),
        empty_cache_fn=lambda: torch.mlu.empty_cache(),
        preferred_dtype=torch.float16,
        hf_accelerate=True,
    ),
    DeviceSpec(
        name="vacc",
        is_available=is_vacc_available,
        env_name="VACC_VISIBLE_DEVICES",
        device_count_fn=lambda: torch.vacc.device_count(),
        empty_cache_fn=lambda: torch.vacc.empty_cache(),
        preferred_dtype=torch.float16,
    ),
    DeviceSpec(
        name="musa",
        is_available=is_musa_available,
        env_name="MUSA_VISIBLE_DEVICES",
        device_count_fn=lambda: torch.musa.device_count(),
        empty_cache_fn=lambda: torch.musa.empty_cache(),
        preferred_dtype=torch.float16,
        hf_accelerate=True,
    ),
]


# ---------------------------------------------------------------------------
# Public API (data-driven, no if/elif chains)
# ---------------------------------------------------------------------------


def _find_device() -> Optional[DeviceSpec]:
    for spec in DEVICE_REGISTRY:
        if spec.is_available():
            return spec
    return None


def get_available_device() -> DeviceType:
    spec = _find_device()
    if spec is None:
        return "cpu"
    return spec.name  # type: ignore


def is_device_available(device: str) -> bool:
    if device == "cpu":
        return True
    return any(s.name == device and s.is_available() for s in DEVICE_REGISTRY)


def move_model_to_available_device(model):
    device = get_available_device()
    if device == "cpu":
        return model
    return model.to(device)


def get_device_preferred_dtype(device: str) -> Union[torch.dtype, None]:
    if device == "cpu":
        return torch.float32
    for spec in DEVICE_REGISTRY:
        if spec.name == device:
            return spec.preferred_dtype
    return None


def is_hf_accelerate_supported(device: str) -> bool:
    return any(s.name == device and s.hf_accelerate for s in DEVICE_REGISTRY)


def empty_cache():
    for spec in DEVICE_REGISTRY:
        if spec.is_available() and spec.empty_cache_fn:
            spec.empty_cache_fn()


def get_available_device_env_name():
    spec = _find_device()
    if spec is None:
        return None
    return spec.env_name


def gpu_count():
    spec = _find_device()
    if spec is None:
        return 0
    count = spec.device_count_fn()
    if spec.env_name is None:
        return count
    return _count_with_visible_devices(spec.env_name, count)


# ---------------------------------------------------------------------------
# GPU info collection
# ---------------------------------------------------------------------------


def _get_nvidia_gpu_mem_info(gpu_id: int) -> Dict[str, float]:
    from pynvml import (
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetName,
        nvmlDeviceGetUtilizationRates,
    )

    handler = nvmlDeviceGetHandleByIndex(gpu_id)
    gpu_name = nvmlDeviceGetName(handler)
    mem_info = nvmlDeviceGetMemoryInfo(handler)
    utilization = nvmlDeviceGetUtilizationRates(handler)
    return {
        "name": gpu_name,
        "total": mem_info.total,
        "used": mem_info.used,
        "free": mem_info.free,
        "util": utilization.gpu,
    }


def get_nvidia_gpu_info() -> Dict:
    from pynvml import nvmlDeviceGetCount, nvmlInit, nvmlShutdown

    try:
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        res = {}
        for i in range(device_count):
            res[f"gpu-{i}"] = _get_nvidia_gpu_mem_info(i)
        return res
    except Exception:
        # Fall back to torch-based detection when NVML lacks support.
        try:
            if torch.cuda.is_available():
                res = {}
                for i in range(torch.cuda.device_count()):
                    res[f"gpu-{i}"] = {
                        "name": torch.cuda.get_device_name(i),
                        "total": 0,
                        "used": 0,
                        "free": 0,
                        "util": 0,
                    }
                return res
        except Exception:
            pass
        return {}
    finally:
        try:
            nvmlShutdown()
        except Exception:
            pass
