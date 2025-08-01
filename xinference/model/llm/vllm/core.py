# Copyright 2022-2023 XProbe Inc.
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

import asyncio
import importlib.util
import itertools
import json
import logging
import multiprocessing
import os
import sys
import threading
import time
import uuid
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypedDict,
    Union,
)

import xoscar as xo
from packaging import version
from typing_extensions import NotRequired

from ....constants import XINFERENCE_MAX_TOKENS
from ....types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Completion,
    CompletionChoice,
    CompletionChunk,
    CompletionUsage,
    LoRA,
)
from .. import BUILTIN_LLM_FAMILIES, LLM, LLMFamilyV2, LLMSpecV1
from ..core import chat_context_var
from ..llm_family import CustomLLMFamilyV2, cache_model_tokenizer_and_config
from ..utils import (
    DEEPSEEK_TOOL_CALL_FAMILY,
    QWEN_TOOL_CALL_FAMILY,
    QWEN_TOOL_CALL_SYMBOLS,
    ChatModelMixin,
    generate_completion_chunk,
)
from .utils import vllm_check

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from vllm.outputs import RequestOutput


class VLLMModelConfig(TypedDict, total=False):
    tokenizer_mode: Optional[str]
    trust_remote_code: bool
    tensor_parallel_size: int
    block_size: int
    swap_space: int  # GiB
    gpu_memory_utilization: float
    max_num_batched_tokens: int
    max_num_seqs: int
    quantization: Optional[str]
    max_model_len: Optional[int]
    limit_mm_per_prompt: Optional[Dict[str, int]]
    guided_decoding_backend: Optional[str]
    scheduling_policy: Optional[str]
    reasoning_content: bool
    model_quantization: Optional[str]
    mm_processor_kwargs: NotRequired[dict[str, Any]]
    min_pixels: NotRequired[int]
    max_pixels: NotRequired[int]


class VLLMGenerateConfig(TypedDict, total=False):
    lora_name: Optional[str]
    n: int
    best_of: Optional[int]
    presence_penalty: float
    frequency_penalty: float
    temperature: float
    top_p: float
    top_k: int
    max_tokens: int
    stop_token_ids: Optional[List[int]]
    stop: Optional[Union[str, List[str]]]
    stream: bool  # non-sampling param, should not be passed to the engine.
    stream_options: Optional[Union[dict, None]]
    skip_special_tokens: Optional[bool]
    response_format: Optional[dict]
    guided_json: Optional[Union[str, dict]]
    guided_regex: Optional[str]
    guided_choice: Optional[List[str]]
    guided_grammar: Optional[str]
    guided_json_object: Optional[bool]
    guided_decoding_backend: Optional[str]
    guided_whitespace_pattern: Optional[str]


try:
    import vllm  # noqa: F401

    if not getattr(vllm, "__version__", None):
        raise ImportError(
            "vllm not installed properly, or wrongly be found in sys.path"
        )

    VLLM_INSTALLED = True
    VLLM_VERSION = version.parse(vllm.__version__)
except ImportError:
    VLLM_INSTALLED = False
    VLLM_VERSION = None

VLLM_SUPPORTED_VISION_MODEL_LIST: List[str] = []
VLLM_SUPPORTED_MODELS = [
    "llama-2",
    "llama-3",
    "mistral-v0.1",
    "codestral-v0.1",
    "Yi",
    "Yi-1.5",
    "code-llama",
    "code-llama-python",
    "deepseek",
    "deepseek-coder",
    "yi-coder",
]
VLLM_SUPPORTED_CHAT_MODELS = [
    "llama-2-chat",
    "llama-3-instruct",
    "baichuan-2-chat",
    "internlm2-chat",
    "internlm2.5-chat",
    "internlm2.5-chat-1m",
    "qwen-chat",
    "Yi-chat",
    "Yi-1.5-chat",
    "Yi-1.5-chat-16k",
    "code-llama-instruct",
    "mistral-instruct-v0.1",
    "mistral-instruct-v0.2",
    "mistral-instruct-v0.3",
    "mixtral-instruct-v0.1",
    "mixtral-8x22B-instruct-v0.1",
    "chatglm3",
    "chatglm3-32k",
    "chatglm3-128k",
    "glm4-chat",
    "glm4-chat-1m",
    "codegeex4",
    "deepseek-chat",
    "deepseek-coder-instruct",
    "yi-coder-chat",
]
if VLLM_INSTALLED and VLLM_VERSION >= version.parse("0.3.0"):
    VLLM_SUPPORTED_CHAT_MODELS.append("qwen1.5-chat")
    VLLM_SUPPORTED_MODELS.append("codeqwen1.5")
    VLLM_SUPPORTED_CHAT_MODELS.append("codeqwen1.5-chat")
    VLLM_SUPPORTED_CHAT_MODELS.append("qwen2-instruct")
    VLLM_SUPPORTED_MODELS.append("qwen2.5")
    VLLM_SUPPORTED_CHAT_MODELS.append("qwen2.5-instruct")
    VLLM_SUPPORTED_MODELS.append("qwen2.5-coder")
    VLLM_SUPPORTED_CHAT_MODELS.append("qwen2.5-coder-instruct")
    VLLM_SUPPORTED_CHAT_MODELS.append("XiYanSQL-QwenCoder-2504")
    VLLM_SUPPORTED_CHAT_MODELS.append("QwQ-32B-Preview")
    VLLM_SUPPORTED_CHAT_MODELS.append("QwQ-32B")
    VLLM_SUPPORTED_CHAT_MODELS.append("marco-o1")
    VLLM_SUPPORTED_CHAT_MODELS.append("deepseek-r1-distill-qwen")
    VLLM_SUPPORTED_CHAT_MODELS.append("fin-r1")
    VLLM_SUPPORTED_CHAT_MODELS.append("seallms-v3")
    VLLM_SUPPORTED_CHAT_MODELS.append("skywork-or1-preview")
    VLLM_SUPPORTED_CHAT_MODELS.append("skywork-or1")
    VLLM_SUPPORTED_CHAT_MODELS.append("HuatuoGPT-o1-Qwen2.5")
    VLLM_SUPPORTED_CHAT_MODELS.append("DianJin-R1")

if VLLM_INSTALLED and VLLM_VERSION >= version.parse("0.3.2"):
    VLLM_SUPPORTED_CHAT_MODELS.append("gemma-it")

if VLLM_INSTALLED and VLLM_VERSION >= version.parse("0.3.3"):
    VLLM_SUPPORTED_CHAT_MODELS.append("orion-chat")
    VLLM_SUPPORTED_CHAT_MODELS.append("orion-chat-rag")

if VLLM_INSTALLED and VLLM_VERSION >= version.parse("0.4.0"):
    VLLM_SUPPORTED_CHAT_MODELS.append("qwen1.5-moe-chat")
    VLLM_SUPPORTED_CHAT_MODELS.append("qwen2-moe-instruct")
    VLLM_SUPPORTED_CHAT_MODELS.append("c4ai-command-r-v01")

if VLLM_INSTALLED and VLLM_VERSION >= version.parse("0.5.1"):
    VLLM_SUPPORTED_CHAT_MODELS.append("deepseek-v2-chat")
    VLLM_SUPPORTED_CHAT_MODELS.append("deepseek-v2-chat-0628")
    VLLM_SUPPORTED_CHAT_MODELS.append("deepseek-v2.5")
    VLLM_SUPPORTED_CHAT_MODELS.append("deepseek-v3")
    VLLM_SUPPORTED_CHAT_MODELS.append("deepseek-v3-0324")
    VLLM_SUPPORTED_CHAT_MODELS.append("deepseek-r1")
    VLLM_SUPPORTED_CHAT_MODELS.append("deepseek-r1-0528")
    VLLM_SUPPORTED_CHAT_MODELS.append("deepseek-prover-v2")
    VLLM_SUPPORTED_CHAT_MODELS.append("deepseek-r1-0528-qwen3")

if VLLM_INSTALLED and VLLM_VERSION >= version.parse("0.5.3"):
    VLLM_SUPPORTED_CHAT_MODELS.append("gemma-2-it")
    VLLM_SUPPORTED_CHAT_MODELS.append("mistral-nemo-instruct")
    VLLM_SUPPORTED_CHAT_MODELS.append("mistral-large-instruct")

if VLLM_INSTALLED and VLLM_VERSION > version.parse("0.5.3"):
    VLLM_SUPPORTED_MODELS.append("llama-3.1")
    VLLM_SUPPORTED_CHAT_MODELS.append("llama-3.1-instruct")
    VLLM_SUPPORTED_CHAT_MODELS.append("llama-3.3-instruct")
    VLLM_SUPPORTED_CHAT_MODELS.append("deepseek-r1-distill-llama")
    VLLM_SUPPORTED_CHAT_MODELS.append("HuatuoGPT-o1-LLaMA-3.1")

if VLLM_INSTALLED and VLLM_VERSION >= version.parse("0.6.1"):
    VLLM_SUPPORTED_VISION_MODEL_LIST.append("internvl2")
    VLLM_SUPPORTED_VISION_MODEL_LIST.append("InternVL2.5")
    VLLM_SUPPORTED_VISION_MODEL_LIST.append("InternVL2.5-MPO")
    VLLM_SUPPORTED_VISION_MODEL_LIST.append("InternVL3")

if VLLM_INSTALLED and VLLM_VERSION >= version.parse("0.6.2"):
    VLLM_SUPPORTED_CHAT_MODELS.append("minicpm3-4b")

if VLLM_INSTALLED and VLLM_VERSION >= version.parse("0.6.3"):
    VLLM_SUPPORTED_MODELS.append("llama-3.2-vision")
    VLLM_SUPPORTED_VISION_MODEL_LIST.append("llama-3.2-vision-instruct")
    VLLM_SUPPORTED_VISION_MODEL_LIST.append("qwen2-vl-instruct")
    VLLM_SUPPORTED_VISION_MODEL_LIST.append("QvQ-72B-Preview")

if VLLM_INSTALLED and VLLM_VERSION >= version.parse("0.7.0"):
    VLLM_SUPPORTED_CHAT_MODELS.append("internlm3-instruct")

if VLLM_INSTALLED and VLLM_VERSION >= version.parse("0.7.2"):
    VLLM_SUPPORTED_VISION_MODEL_LIST.append("qwen2.5-vl-instruct")
    VLLM_SUPPORTED_CHAT_MODELS.append("moonlight-16b-a3b-instruct")

if VLLM_INSTALLED and VLLM_VERSION >= version.parse("0.7.3"):
    VLLM_SUPPORTED_CHAT_MODELS.append("qwen2.5-instruct-1m")
    VLLM_SUPPORTED_CHAT_MODELS.append("qwenLong-l1")

if VLLM_INSTALLED and VLLM_VERSION >= version.parse("0.8.0"):
    VLLM_SUPPORTED_CHAT_MODELS.append("gemma-3-1b-it")
    VLLM_SUPPORTED_VISION_MODEL_LIST.append("gemma-3-it")

if VLLM_INSTALLED and VLLM_VERSION >= version.parse("0.8.4"):
    VLLM_SUPPORTED_CHAT_MODELS.append("glm4-0414")

if VLLM_INSTALLED and VLLM_VERSION >= version.parse("0.8.5"):
    VLLM_SUPPORTED_CHAT_MODELS.append("qwen3")

if VLLM_INSTALLED and VLLM_VERSION >= version.parse("0.9.1"):
    VLLM_SUPPORTED_CHAT_MODELS.append("minicpm4")

if VLLM_INSTALLED and VLLM_VERSION >= version.parse("0.9.2"):
    VLLM_SUPPORTED_CHAT_MODELS.append("Ernie4.5")
    VLLM_SUPPORTED_VISION_MODEL_LIST.append("glm-4.1v-thinking")
    VLLM_SUPPORTED_CHAT_MODELS.append("Qwen3-Instruct")


class VLLMModel(LLM):
    def __init__(
        self,
        model_uid: str,
        model_family: "LLMFamilyV2",
        model_path: str,
        model_config: Optional[VLLMModelConfig],
        peft_model: Optional[List[LoRA]] = None,
    ):
        try:
            from vllm.lora.request import LoRARequest
        except ImportError:
            error_message = "Failed to import module 'vllm'"
            installation_guide = [
                "Please make sure 'vllm' is installed. ",
                "You can install it by `pip install vllm`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")
        super().__init__(model_uid, model_family, model_path)
        self._model_config = model_config
        self._engine = None
        self.lora_modules = peft_model
        self.lora_requests: List[LoRARequest] = []
        self._xavier_config = None
        self._context_length: Optional[int] = None
        # distributed inference
        self._device_count = None
        self._address = model_config.pop("address", None)  # type: ignore
        self._n_worker = model_config.pop("n_worker", 1)  # type: ignore
        self._shard = model_config.pop("shard", 0)  # type: ignore
        self._driver_info = model_config.pop("driver_info", None)  # type: ignore
        self._loading_thread: Optional[threading.Thread] = None
        self._loading_error = None
        # variables used for distributed inference and multiple GPUs
        self._pool_addresses = None
        self._worker_addresses: Optional[Dict[int, List[str]]] = None
        self._all_worker_ready: Optional[threading.Event] = None
        # used to call async
        self._loop = None

    def set_xavier_config(self, value: Optional[Dict]):
        self._xavier_config = value  # type: ignore

    def set_worker_addresses(self, shard: int, worker_addresses: List[str]):
        assert self._worker_addresses is not None
        self._worker_addresses[shard] = worker_addresses
        if (
            self._all_worker_ready is not None
            and len(self._worker_addresses) == self._n_worker
        ):
            self._all_worker_ready.set()

    @property
    def driver_info(self) -> Optional[dict]:
        return self._driver_info

    @property
    def need_create_pools(self):
        return True

    def set_pool_addresses(self, pool_addresses: List[str]):
        self._pool_addresses = pool_addresses  # type: ignore

    def get_pool_addresses(self) -> Optional[List[str]]:
        return self._pool_addresses

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        # loop will be passed into XinferenceDistributedExecutor,
        # to call aynsc method with asyncio.run_coroutine_threadsafe
        self._loop = loop  # type: ignore

    def load(self):
        try:
            import vllm
            from vllm import envs
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.engine.async_llm_engine import AsyncLLMEngine
            from vllm.executor.executor_base import ExecutorBase
            from vllm.lora.request import LoRARequest
        except ImportError:
            error_message = "Failed to import module 'vllm'"
            installation_guide = [
                "Please make sure 'vllm' is installed. ",
                "You can install it by `pip install vllm`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        from ..llm_family import LlamaCppLLMSpecV2

        if "0.3.1" <= vllm.__version__ <= "0.3.3":
            # from vllm v0.3.1 to v0.3.3, it uses cupy as NCCL backend
            # in which cupy will fork a process
            # only for xoscar >= 0.3.0, new process is allowed in subpool
            # besides, xinference set start method as forkserver for unix
            # we need to set it to fork to make cupy NCCL work
            multiprocessing.set_start_method("fork", force=True)

        self._device_count = self._get_cuda_count()
        self._model_config = self._sanitize_model_config(self._model_config)
        reasoning_content = self._model_config.pop("reasoning_content")
        enable_thinking = self._model_config.pop("enable_thinking", False)
        self.prepare_parse_reasoning_content(
            reasoning_content, enable_thinking=enable_thinking
        )

        if (
            isinstance(self.model_spec, LlamaCppLLMSpecV2)
            and self.model_spec.model_format == "ggufv2"
        ):
            # gguf
            self._preprocess_load_gguf()

        if self.lora_modules is None:
            self.lora_requests = []
        else:
            self.lora_requests = [
                LoRARequest(
                    lora_name=lora.lora_name,
                    lora_int_id=i,
                    lora_local_path=lora.local_path,
                )
                for i, lora in enumerate(self.lora_modules, start=1)
            ]

        enable_lora = len(self.lora_requests) > 0
        max_loras = len(self.lora_requests)

        logger.info(
            f"Loading {self.model_uid} with following model config: {self._model_config}"
            f"Enable lora: {enable_lora}. Lora count: {max_loras}."
        )

        if self._xavier_config is not None:
            from .xavier.engine import XavierEngine

            # Enabling Xavier means that `enable_prefix_caching` is enabled by default.
            self._model_config.setdefault("enable_prefix_caching", True)
            xavier_transfer_block_num = self._model_config.pop(
                "xavier_transfer_block_num", 512
            )
            self._xavier_config["transfer_block_num"] = xavier_transfer_block_num
            engine_args = AsyncEngineArgs(
                model=self.model_path,
                enable_lora=enable_lora,
                max_loras=max_loras,
                **self._model_config,
            )

            logger.debug(f"Start xavier for vllm with config: {self._xavier_config}")
            self._engine = XavierEngine.from_engine_args(
                engine_args, xavier_config=self._xavier_config
            )
        elif self._n_worker > 1 or (
            self._device_count > 1 and vllm.__version__ >= "0.7.0"
        ):
            from vllm.config import VllmConfig

            # model across multiple workers or GPUs
            engine_args = AsyncEngineArgs(
                model=self.model_path,
                enable_lora=enable_lora,
                max_loras=max_loras,
                **self._model_config,
            )
            self._enable_v1_if_supported(engine_args)

            assert self._loop is not None
            self._worker_addresses = {}

            def _load():
                try:
                    assert self._pool_addresses

                    if self._shard > 0:
                        assert self._driver_info
                        address = self._driver_info["address"]

                        coro = xo.actor_ref(address, self.raw_model_uid)
                        model_ref = asyncio.run_coroutine_threadsafe(
                            coro, self._loop
                        ).result()
                        coro = model_ref.set_worker_addresses(
                            self._shard, self._pool_addresses
                        )
                        asyncio.run_coroutine_threadsafe(coro, self._loop).result()
                    else:
                        self.set_worker_addresses(0, self._pool_addresses)
                        self._driver_info = {"address": self._address}

                        if self._n_worker > 1:
                            self._all_worker_ready = threading.Event()
                            # if model across workers, wait for other workers ready
                            self._all_worker_ready.wait()

                        # gather all worker addresses
                        worker_addresses = list(
                            itertools.chain(
                                *[
                                    self._worker_addresses[shard]
                                    for shard in range(self._n_worker)
                                ]
                            )
                        )
                        assert worker_addresses
                        loop = self._loop

                        if not (envs.is_set("VLLM_USE_V1") and envs.VLLM_USE_V1):
                            # vLLM v0
                            from .distributed_executor import (
                                XinferenceDistributedExecutor,
                            )

                            class XinferenceAsyncLLMEngine(AsyncLLMEngine):
                                @classmethod
                                def _get_executor_cls(
                                    cls, engine_config: VllmConfig
                                ) -> Type[ExecutorBase]:
                                    return partial(  # type: ignore
                                        XinferenceDistributedExecutor,
                                        pool_addresses=worker_addresses,
                                        n_worker=self._n_worker,
                                        loop=loop,
                                    )

                            self._engine = XinferenceAsyncLLMEngine.from_engine_args(
                                engine_args
                            )
                        else:
                            from vllm.v1.executor.abstract import Executor

                            from .distributed_executor import (
                                XinferenceDistributedExecutorV1,
                            )

                            # vLLM V1
                            # NOTE: loop has to be None for vLLM v1
                            # in v1, a new process called EngineCore will be created via fork by default
                            # in which executor is initialized, we cannot pass loop, or it will be stuck,
                            # instead, a new loop will be created inside executor
                            executor_cls = partial(  # type: ignore
                                XinferenceDistributedExecutorV1,
                                pool_addresses=worker_addresses,
                                n_worker=self._n_worker,
                            )
                            # patch vllm Executor.get_class
                            Executor.get_class = lambda vllm_config: executor_cls
                            self._engine = AsyncLLMEngine.from_engine_args(engine_args)
                except:
                    logger.exception("Creating vllm engine failed")
                    self._loading_error = sys.exc_info()

            self._loading_thread = threading.Thread(target=_load)
            self._loading_thread.start()
            # wait some time for init finish
            if self._shard == 0:
                self._loading_thread.join(1)
        else:
            engine_args = AsyncEngineArgs(
                model=self.model_path,
                enable_lora=enable_lora,
                max_loras=max_loras,
                **self._model_config,
            )
            self._enable_v1_if_supported(engine_args)
            self._engine = AsyncLLMEngine.from_engine_args(engine_args)

        self._check_health_task = None
        if hasattr(self._engine, "check_health"):
            # vLLM introduced `check_health` since v0.4.1
            self._check_health_task = self._loop.create_task(self._check_healthy())

    def wait_for_load(self):
        if self._loading_thread:
            self._loading_thread.join()
            if self._loading_error:
                _, err, tb = self._loading_error
                raise err.with_traceback(tb)

        # set context length after engine inited
        self._set_context_length()

    def _set_context_length(self):
        from vllm import envs

        if not (envs.is_set("VLLM_USE_V1") and envs.VLLM_USE_V1):
            # v0
            self._context_length = (
                self._engine.engine.vllm_config.model_config.max_model_len
            )
        else:
            # v1
            self._context_length = self._engine.model_config.max_model_len
        assert self._context_length is not None
        logger.debug("Model context length: %s", self._context_length)

    def _enable_v1_if_supported(self, engine_args: "vllm.AsyncEngineArgs"):
        from vllm import __version__ as vllm_version

        if os.getenv("VLLM_USE_V1") is not None:
            logger.debug(
                "Setting vLLM v1 via environment variable already, skip checking"
            )
            return

        try:
            supported_func = engine_args._is_v1_supported_oracle
        except AttributeError:
            logger.debug(
                "Cannot get `EngineArgs._is_v1_supported_oracle` "
                "to decide enabling vLLM v1, perhaps vllm version is too old, "
                "version: %s",
                vllm_version,
            )
            return

        model_config = engine_args.create_model_config()
        old_main_thread = threading.main_thread()
        try:
            # HACK: patch main thread to let vllm pass check
            # vllm do some signal handling when on main thread
            # but they will skip registering signal if not on main thread,
            # however, the _is_v1_supported_oracle will return False
            # when not on main thread, we patched the main thread temporially,
            # It's OK because Xinference will take care of all processes
            threading.main_thread = lambda: threading.current_thread()

            if supported_func(model_config):
                logger.debug("Setting vLLM v1 by checking model config")
                os.environ["VLLM_USE_V1"] = "1"
            else:
                logger.debug("Use vLLM v0 due to not supported config")
        finally:
            # patch back
            threading.main_thread = lambda: old_main_thread

    def _preprocess_load_gguf(self):
        # check if it is multi gguf files
        if (
            not os.path.isfile(self.model_path)
            and self.model_spec.quantization_parts
            and self.quantization in self.model_spec.quantization_parts
        ):
            raise RuntimeError(
                "vllm does not support multiple gguf files, please merge them first and "
                "provide `model_path` with merged file"
            )

        if "tokenizer" not in self._model_config:
            # find pytorch format without quantization
            family = next(
                family
                for family in BUILTIN_LLM_FAMILIES
                if family.model_name == self.model_family.model_name
            ).copy()
            non_quant_spec = next(
                spec
                for spec in family.model_specs
                if spec.quantization == "none"
                and spec.model_size_in_billions
                == self.model_spec.model_size_in_billions
                and spec.model_hub == self.model_spec.model_hub
            )
            family.model_specs = [non_quant_spec]
            path = cache_model_tokenizer_and_config(family)
            # other than gguf file, vllm requires to provide tokenizer and hf_config_path
            self._model_config["tokenizer"] = self._model_config["hf_config_path"] = (
                path
            )

        if not os.path.isfile(self.model_path):
            self.model_path = os.path.realpath(
                os.path.join(
                    self.model_path,
                    self.model_spec.model_file_name_template.format(
                        quantization=self.quantization
                    ),
                )
            )

    def stop(self):
        from vllm import envs

        # though the vLLM engine will shutdown when deleted,
        # but some issue e.g. GH#1682 reported
        # when deleting, the engine exists still
        logger.info("Stopping vLLM engine")
        if self._check_health_task:
            self._check_health_task.cancel()
        if self._engine:
            if not (envs.is_set("VLLM_USE_V1") and envs.VLLM_USE_V1):
                # v0
                if model_executor := getattr(
                    self._engine.engine, "model_executor", None
                ):
                    model_executor.shutdown()
                self._engine = None
            else:
                # v1
                self._engine.shutdown()
                self._engine = None

    async def init_xavier(self):
        await self._engine.init_xavier()

    async def _check_healthy(self, interval: int = 30):
        from vllm.engine.async_llm_engine import AsyncEngineDeadError

        logger.debug("Begin to check health of vLLM")

        while self._engine is not None:
            try:
                await self._engine.check_health()
            except (AsyncEngineDeadError, RuntimeError):
                logger.info("Detecting vLLM is not health, prepare to quit the process")
                try:
                    self.stop()
                except:
                    # ignore error when stop
                    pass
                # Just kill the process and let xinference auto-recover the model
                os._exit(1)
            else:
                await asyncio.sleep(interval)

    def _sanitize_model_config(
        self, model_config: Optional[VLLMModelConfig]
    ) -> VLLMModelConfig:
        if model_config is None:
            model_config = VLLMModelConfig()

        model_config.setdefault("tokenizer_mode", "auto")
        model_config.setdefault("trust_remote_code", True)
        model_config.setdefault("tensor_parallel_size", self._device_count)  # type: ignore
        model_config.setdefault("pipeline_parallel_size", self._n_worker)  # type: ignore
        model_config.setdefault("block_size", 16)
        model_config.setdefault("swap_space", 4)
        model_config.setdefault("gpu_memory_utilization", 0.90)
        model_config.setdefault("max_num_seqs", 256)
        if "model_quantization" in model_config:
            model_config["quantization"] = model_config.pop("model_quantization")
        else:
            model_config.setdefault("quantization", None)
        model_config.setdefault("max_model_len", None)
        model_config.setdefault("reasoning_content", False)
        # Add scheduling policy if vLLM version is 0.6.3 or higher
        if vllm.__version__ >= "0.6.3":
            model_config.setdefault("scheduling_policy", "fcfs")
            # init mm_processor_kwargs params
            mm_processor_kwargs = model_config.get("mm_processor_kwargs", {})
            if isinstance(mm_processor_kwargs, str):
                try:
                    mm_processor_kwargs = json.loads(mm_processor_kwargs)
                except json.JSONDecodeError:
                    logger.warning(
                        "Failed to parse mm_processor_kwargs as JSON, using default empty dict"
                    )
                    mm_processor_kwargs = {}
                except Exception as e:
                    logger.warning(
                        f"Unexpected error parsing mm_processor_kwargs: {e}, using default empty dict"
                    )
                    mm_processor_kwargs = {}
            pixel_params: Dict[str, int] = {}
            if "min_pixels" in model_config:
                pixel_params["min_pixels"] = model_config.pop("min_pixels")
            if "max_pixels" in model_config:
                pixel_params["max_pixels"] = model_config.pop("max_pixels")
            if pixel_params or mm_processor_kwargs:
                model_config["mm_processor_kwargs"] = {
                    **mm_processor_kwargs,
                    **pixel_params,
                }
        return model_config

    @staticmethod
    def _sanitize_generate_config(
        generate_config: Optional[Dict] = None,
    ) -> VLLMGenerateConfig:
        if not generate_config:
            generate_config = {}

        sanitized = VLLMGenerateConfig()

        response_format = generate_config.pop("response_format", None)
        guided_decoding_backend = generate_config.get("guided_decoding_backend", None)
        guided_json_object = None
        guided_json = None

        if response_format is not None:
            if response_format.get("type") == "json_object":
                guided_json_object = True
            elif response_format.get("type") == "json_schema":
                json_schema = response_format.get("json_schema")
                assert json_schema is not None
                guided_json = json_schema.get("json_schema")
                if guided_decoding_backend is None:
                    guided_decoding_backend = "outlines"

        sanitized.setdefault("lora_name", generate_config.get("lora_name", None))
        sanitized.setdefault("n", generate_config.get("n", 1))
        sanitized.setdefault("best_of", generate_config.get("best_of", None))
        sanitized.setdefault(
            "presence_penalty", generate_config.get("presence_penalty", 0.0)
        )
        sanitized.setdefault(
            "frequency_penalty", generate_config.get("frequency_penalty", 0.0)
        )
        sanitized.setdefault("temperature", generate_config.get("temperature", 1.0))
        sanitized.setdefault("top_p", generate_config.get("top_p", 1.0))
        sanitized.setdefault("top_k", generate_config.get("top_k", -1))
        sanitized.setdefault(  # type: ignore
            "max_tokens",
            generate_config.get("max_tokens", XINFERENCE_MAX_TOKENS)  # type: ignore
            or XINFERENCE_MAX_TOKENS,
        )
        sanitized.setdefault("stop", generate_config.get("stop", None))
        sanitized.setdefault(
            "stop_token_ids", generate_config.get("stop_token_ids", None)
        )
        sanitized.setdefault("stream", generate_config.get("stream", False))
        sanitized.setdefault(
            "stream_options", generate_config.get("stream_options", None)
        )
        sanitized.setdefault(
            "skip_special_tokens", generate_config.get("skip_special_tokens", True)
        )
        sanitized.setdefault(
            "guided_json", generate_config.get("guided_json", guided_json)
        )
        sanitized.setdefault("guided_regex", generate_config.get("guided_regex", None))
        sanitized.setdefault(
            "guided_choice", generate_config.get("guided_choice", None)
        )
        sanitized.setdefault(
            "guided_grammar", generate_config.get("guided_grammar", None)
        )
        sanitized.setdefault(
            "guided_whitespace_pattern",
            generate_config.get("guided_whitespace_pattern", None),
        )
        sanitized.setdefault(
            "guided_json_object",
            generate_config.get("guided_json_object", guided_json_object),
        )
        sanitized.setdefault(
            "guided_decoding_backend",
            generate_config.get("guided_decoding_backend", guided_decoding_backend),
        )

        return sanitized

    @classmethod
    def check_lib(cls) -> bool:
        return importlib.util.find_spec("vllm") is not None

    @classmethod
    def match_json(
        cls, llm_family: "LLMFamilyV2", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if not cls._has_cuda_device() and not cls._has_mlu_device():
            return False
        if not cls._is_linux():
            return False
        if llm_spec.model_format not in ["pytorch", "gptq", "awq", "fp8"]:
            return False
        if llm_spec.model_format == "pytorch":
            if quantization != "none" and not (quantization is None):
                return False
        if llm_spec.model_format == "awq":
            # Currently, only 4-bit weight quantization is supported for AWQ, but got 8 bits.
            if "4" not in quantization:
                return False
        if llm_spec.model_format == "gptq":
            if VLLM_INSTALLED and vllm.__version__ >= "0.3.3":
                if not any(q in quantization for q in ("3", "4", "8")):
                    return False
            else:
                if "4" not in quantization:
                    return False
        if isinstance(llm_family, CustomLLMFamilyV2):
            if llm_family.model_family not in VLLM_SUPPORTED_MODELS:
                return False
        else:
            if llm_family.model_name not in VLLM_SUPPORTED_MODELS:
                return False
        if "generate" not in llm_family.model_ability:
            return False
        return VLLM_INSTALLED

    @staticmethod
    def _convert_request_output_to_completion_chunk(
        request_id: str, model: str, request_output: "RequestOutput"
    ) -> Tuple[CompletionChunk, Optional[str]]:
        choices: List[CompletionChoice] = []
        finish_reason = None
        for output in request_output.outputs:
            choices.append(
                CompletionChoice(
                    text=output.text,
                    index=output.index,
                    logprobs=None,  # TODO: support logprobs.
                    finish_reason=None,
                )
            )
            finish_reason = output.finish_reason
        return (
            CompletionChunk(
                id=request_id,
                object="text_completion",
                created=int(time.time()),
                model=model,
                choices=choices,
            ),
            finish_reason,
        )

    @staticmethod
    def _convert_request_output_to_completion(
        request_id: str, model: str, request_output: "RequestOutput"
    ) -> Completion:
        choices = []
        for output in request_output.outputs:
            choices.append(
                CompletionChoice(
                    text=output.text,
                    index=output.index,
                    logprobs=None,  # TODO: support logprobs.
                    finish_reason=output.finish_reason,
                )
            )

        prompt_tokens = len(request_output.prompt_token_ids)
        completion_tokens = sum(
            len(output.token_ids) for output in request_output.outputs
        )
        usage = CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        return Completion(
            id=request_id,
            object="text_completion",
            created=int(time.time()),
            model=model,
            choices=choices,
            usage=usage,
        )

    async def _get_tokenizer(self, lora_request: Any) -> Any:
        try:
            return await self._engine.get_tokenizer(lora_request)  # type: ignore
        except AttributeError:
            return await self._engine.get_tokenizer_async(lora_request)  # type: ignore

    def _tokenize(self, tokenizer: Any, prompt: str, config: dict) -> List[int]:
        truncate_prompt_tokens = config.get("truncate_prompt_tokens")
        add_special_tokens = config.get("add_special_tokens", True)

        if truncate_prompt_tokens is None:
            encoded = tokenizer(prompt, add_special_tokens=add_special_tokens)
        elif truncate_prompt_tokens < 0:
            # Negative means we cap at the model's max length
            encoded = tokenizer(
                prompt,
                add_special_tokens=add_special_tokens,
                truncation=True,
                max_length=self._context_length,
            )
        else:
            encoded = tokenizer(
                prompt,
                add_special_tokens=add_special_tokens,
                truncation=True,
                max_length=truncate_prompt_tokens,
            )

        return encoded.input_ids

    async def _gen_tokens_prompt(
        self, tokenizer, prompt: Union[str, dict], config: dict
    ):
        from vllm import TokensPrompt

        token_ids = await asyncio.to_thread(
            self._tokenize, tokenizer, prompt, config  # type: ignore
        )
        return TokensPrompt(prompt_token_ids=token_ids)

    @vllm_check
    async def async_generate(
        self,
        prompt: Union[str, Dict[str, Any]],
        generate_config: Optional[Dict] = None,
        tools: object = False,
        request_id: Optional[str] = None,
    ) -> Union[Completion, AsyncGenerator[CompletionChunk, None]]:
        try:
            from vllm.sampling_params import SamplingParams
        except ImportError:
            error_message = "Failed to import module 'vllm'"
            installation_guide = [
                "Please make sure 'vllm' is installed. ",
                "You can install it by `pip install vllm`\n",
            ]

            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        sanitized_generate_config = self._sanitize_generate_config(generate_config)
        logger.debug(
            "Enter generate, prompt: %s, generate config: %s", prompt, generate_config
        )

        lora_model = sanitized_generate_config.pop("lora_name")

        lora_request = None
        if lora_model is not None:
            for lora in self.lora_requests:
                if lora_model == lora.lora_name:
                    lora_request = lora
                    break

        stream = sanitized_generate_config.pop("stream")
        stream_options = sanitized_generate_config.pop("stream_options", None)
        include_usage = (
            stream_options["include_usage"]
            if isinstance(stream_options, dict)
            else False
        )

        if VLLM_INSTALLED and vllm.__version__ >= "0.6.3":
            # guided decoding only available for vllm >= 0.6.3
            from vllm.sampling_params import GuidedDecodingParams

            guided_options = GuidedDecodingParams.from_optional(
                json=sanitized_generate_config.pop("guided_json", None),
                regex=sanitized_generate_config.pop("guided_regex", None),
                choice=sanitized_generate_config.pop("guided_choice", None),
                grammar=sanitized_generate_config.pop("guided_grammar", None),
                json_object=sanitized_generate_config.pop("guided_json_object", None),
                backend=sanitized_generate_config.pop("guided_decoding_backend", None),
                whitespace_pattern=sanitized_generate_config.pop(
                    "guided_whitespace_pattern", None
                ),
            )

            sampling_params = SamplingParams(
                guided_decoding=guided_options, **sanitized_generate_config
            )
        else:
            # ignore generate configs
            sanitized_generate_config.pop("guided_json", None)
            sanitized_generate_config.pop("guided_regex", None)
            sanitized_generate_config.pop("guided_choice", None)
            sanitized_generate_config.pop("guided_grammar", None)
            sanitized_generate_config.pop("guided_json_object", None)
            sanitized_generate_config.pop("guided_decoding_backend", None)
            sanitized_generate_config.pop("guided_whitespace_pattern", None)
            sampling_params = SamplingParams(**sanitized_generate_config)

        prompt_or_token_ids: Union[str, Dict[str, Any], List[int]] = prompt
        if sampling_params.max_tokens is None:
            # no max_tokens set, try to get the max tokens
            # this requires tokenizing
            tokenizer = await self._get_tokenizer(lora_request)
            prompt_or_token_ids = await self._gen_tokens_prompt(
                tokenizer, prompt, sanitized_generate_config  # type: ignore
            )
            sampling_params.max_tokens = max_tokens = self._context_length - len(  # type: ignore
                prompt_or_token_ids["prompt_token_ids"]  # type: ignore
            )
            logger.debug("No max_tokens set, setting to: %s", max_tokens)

        if not request_id:
            request_id = str(uuid.uuid1())

        assert self._engine is not None
        results_generator = self._engine.generate(
            prompt_or_token_ids,
            sampling_params,
            request_id,
            lora_request,
        )

        async def stream_results() -> AsyncGenerator[CompletionChunk, None]:
            previous_texts = [""] * sanitized_generate_config["n"]
            prompt_tokens, completion_tokens, total_tokens = 0, 0, 0
            complete_response = ""
            match_tool_call_tmp_results = []
            is_match_tool_call = False
            chunk = None
            finish_reason = None
            async for _request_output in results_generator:
                chunk, finish_reason = self._convert_request_output_to_completion_chunk(
                    request_id=request_id,
                    model=self.model_uid,
                    request_output=_request_output,
                )

                for i, choice in enumerate(chunk["choices"]):
                    delta = choice["text"][len(previous_texts[i]) :]
                    previous_texts[i] = choice["text"]
                    choice["text"] = delta
                    complete_response += delta

                prompt_tokens = len(_request_output.prompt_token_ids)
                completion_tokens = sum(
                    len(output.token_ids) for output in _request_output.outputs
                )
                total_tokens = prompt_tokens + completion_tokens
                chunk["usage"] = CompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )

                if tools:
                    """
                    The qwen2 tool call returns format like this:
                    <tool_call>
                    {...}
                    </tool_call>
                    Here is to match this.
                    """
                    if (len(QWEN_TOOL_CALL_SYMBOLS[0]) > len(complete_response)) and (
                        not QWEN_TOOL_CALL_SYMBOLS[0].startswith(complete_response)
                    ):
                        for c in match_tool_call_tmp_results:
                            yield c
                        match_tool_call_tmp_results.clear()
                        yield chunk
                    elif (len(QWEN_TOOL_CALL_SYMBOLS[0]) > len(complete_response)) and (
                        QWEN_TOOL_CALL_SYMBOLS[0].startswith(complete_response)
                    ):
                        match_tool_call_tmp_results.append(chunk)
                    else:
                        assert len(QWEN_TOOL_CALL_SYMBOLS[0]) <= len(complete_response)
                        if not is_match_tool_call and complete_response.startswith(
                            QWEN_TOOL_CALL_SYMBOLS[0]
                        ):
                            is_match_tool_call = True
                            match_tool_call_tmp_results.clear()

                        if not is_match_tool_call:
                            for c in match_tool_call_tmp_results:
                                yield c
                            match_tool_call_tmp_results.clear()
                            yield chunk
                        else:
                            chunk["choices"][0]["text"] = complete_response
                else:
                    yield chunk

            if is_match_tool_call:
                assert chunk is not None
                yield chunk

            logger.info(
                "Generate finished, request_id: %s, stop reason: %s, prompt tokens: %s, "
                "completion tokens: %s, all tokens: %s",
                request_id,
                finish_reason,
                prompt_tokens,
                completion_tokens,
                total_tokens,
            )

            # match OpenAI API stream
            yield generate_completion_chunk(
                chunk_text="",
                finish_reason=finish_reason,
                chunk_id=request_id,
                model_uid=self.model_uid,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

            if include_usage:
                chunk = CompletionChunk(
                    id=request_id,
                    object="text_completion",
                    created=int(time.time()),
                    model=self.model_uid,
                    choices=[],
                )
                chunk["usage"] = CompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
                yield chunk

        if stream:
            return stream_results()
        else:
            final_output = None
            async for request_output in results_generator:
                final_output = request_output

            assert final_output is not None
            return self._convert_request_output_to_completion(
                request_id, model=self.model_uid, request_output=final_output
            )


class VLLMChatModel(VLLMModel, ChatModelMixin):
    @classmethod
    def match_json(
        cls, llm_family: "LLMFamilyV2", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if llm_spec.model_format not in ["pytorch", "gptq", "awq", "fp8", "ggufv2"]:
            return False
        if llm_spec.model_format == "pytorch":
            if quantization != "none" and not (quantization is None):
                return False
        if llm_spec.model_format == "awq":
            # Currently, only 4-bit weight quantization is supported for AWQ, but got 8 bits.
            if "4" not in quantization:
                return False
        if llm_spec.model_format == "gptq":
            if VLLM_INSTALLED and vllm.__version__ >= "0.3.3":
                if not any(q in quantization for q in ("3", "4", "8")):
                    return False
            else:
                if "4" not in quantization:
                    return False
        if llm_spec.model_format == "ggufv2":
            if not (VLLM_INSTALLED and vllm.__version__ >= "0.8.2"):
                return False
        if isinstance(llm_family, CustomLLMFamilyV2):
            if llm_family.model_family not in VLLM_SUPPORTED_CHAT_MODELS:
                return False
        else:
            if llm_family.model_name not in VLLM_SUPPORTED_CHAT_MODELS:
                return False
        if "chat" not in llm_family.model_ability:
            return False
        return VLLM_INSTALLED

    def _sanitize_chat_config(
        self,
        generate_config: Optional[Dict] = None,
    ) -> Dict:
        if not generate_config:
            generate_config = {}
        if "reasoning" in getattr(self.model_family, "model_ability", []):
            generate_config.pop("stop", None)
            generate_config.pop("stop_token_ids", None)
        else:
            if not generate_config.get("stop") and self.model_family.stop:
                generate_config["stop"] = self.model_family.stop.copy()
            if (
                not generate_config.get("stop_token_ids")
                and self.model_family.stop_token_ids
            ):
                generate_config["stop_token_ids"] = (
                    self.model_family.stop_token_ids.copy()
                )
        return generate_config

    @staticmethod
    def is_tool_call_chunk_start(chunk):
        return chunk["choices"][0]["text"].startswith(QWEN_TOOL_CALL_SYMBOLS[0])

    @staticmethod
    def is_tool_call_chunk_end(chunk):
        return chunk["choices"][0]["text"].endswith(QWEN_TOOL_CALL_SYMBOLS[1])

    @staticmethod
    def prefill_messages(messages: List[Dict]) -> List[Dict]:
        """
        Preprocess messages to ensure content is not None

        Args:
            messages: Original message list

        Returns:
            Processed message list, where content is not None
        """
        processed_messages = []

        for msg in messages:
            if isinstance(msg, dict):
                if msg.get("content") is None:
                    msg_copy = msg.copy()
                    msg_copy["content"] = ""  # Replace None with empty string
                    processed_messages.append(msg_copy)
                else:
                    processed_messages.append(msg)
            else:
                processed_messages.append(msg)

        return processed_messages

    async def _async_to_tool_completion_chunks(
        self,
        chunks: AsyncGenerator[CompletionChunk, None],
        ctx: Optional[Dict[str, Any]] = {},
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        def set_context():
            if ctx:
                chat_context_var.set(ctx)

        i = 0
        previous_texts = [""]
        tool_call = False
        tool_call_texts = [""]
        if self.reasoning_parser:
            set_context()
            chunks = self.reasoning_parser.prepare_reasoning_content_streaming(chunks)
        async for chunk in chunks:
            set_context()
            if i == 0:
                for first_chunk in self._get_first_chat_completion_chunk(
                    chunk, self.reasoning_parser
                ):
                    yield first_chunk
            # usage
            choices = chunk.get("choices")
            if not choices:
                yield self._get_final_chat_completion_chunk(chunk)
            else:
                if self.is_tool_call_chunk_start(chunk):
                    tool_call = True
                if tool_call:
                    tool_call_text = tool_call_texts[-1]
                    tool_call_text += chunk["choices"][0]["text"]
                    tool_call_texts.append(tool_call_text)
                    if self.is_tool_call_chunk_end(chunk):
                        yield self._post_process_completion_chunk(
                            self.model_family,
                            self.model_uid,
                            chunk,
                            reasoning_parser=self.reasoning_parser,
                            tool_call_text=tool_call_text,
                        )
                        tool_call = False
                        tool_call_texts = [""]
                else:
                    yield self._to_chat_completion_chunk(
                        chunk, self.reasoning_parser, previous_texts
                    )
            i += 1

    @vllm_check
    async def async_chat(
        self,
        messages: List[Dict],
        generate_config: Optional[Dict] = None,
        request_id: Optional[str] = None,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        # Preprocess messages to ensure content is not None
        messages = self.prefill_messages(messages)

        tools = generate_config.pop("tools", []) if generate_config else None
        model_family = self.model_family.model_family or self.model_family.model_name
        chat_template_kwargs = (
            self._get_chat_template_kwargs_from_generate_config(
                generate_config, self.reasoning_parser
            )
            or {}
        )
        chat_context_var.set(chat_template_kwargs)
        full_context_kwargs = chat_template_kwargs.copy()
        if tools:
            if (
                model_family in QWEN_TOOL_CALL_FAMILY
                or model_family in DEEPSEEK_TOOL_CALL_FAMILY
            ):
                full_context_kwargs["tools"] = tools
        assert self.model_family.chat_template is not None
        full_prompt = self.get_full_context(
            messages, self.model_family.chat_template, **full_context_kwargs
        )

        generate_config = self._sanitize_chat_config(generate_config)
        stream = generate_config.get("stream", None)

        if stream:
            agen = await self.async_generate(
                full_prompt, generate_config, tools, request_id=request_id
            )
            assert isinstance(agen, AsyncGenerator)
            if tools:
                return self._async_to_tool_completion_chunks(agen, chat_template_kwargs)
            return self._async_to_chat_completion_chunks(
                agen, self.reasoning_parser, chat_template_kwargs
            )
        else:
            c = await self.async_generate(
                full_prompt, generate_config, request_id=request_id
            )
            assert not isinstance(c, AsyncGenerator)
            if tools:
                return self._post_process_completion(
                    self.model_family, self.model_uid, c, self.reasoning_parser
                )
            return self._to_chat_completion(c, self.reasoning_parser)


class VLLMVisionModel(VLLMModel, ChatModelMixin):
    @classmethod
    def match_json(
        cls, llm_family: "LLMFamilyV2", llm_spec: "LLMSpecV1", quantization: str
    ) -> bool:
        if not cls._has_cuda_device() and not cls._has_mlu_device():
            return False
        if not cls._is_linux():
            return False
        if llm_spec.model_format not in ["pytorch", "gptq", "awq", "fp8"]:
            return False
        if llm_spec.model_format == "pytorch":
            if quantization != "none" and not (quantization is None):
                return False
        if llm_spec.model_format == "awq":
            # Currently, only 4-bit weight quantization is supported for AWQ, but got 8 bits.
            if "4" not in quantization:
                return False
        if llm_spec.model_format == "gptq":
            if VLLM_INSTALLED and vllm.__version__ >= "0.3.3":
                if not any(q in quantization for q in ("3", "4", "8")):
                    return False
            else:
                if "4" not in quantization:
                    return False
        if isinstance(llm_family, CustomLLMFamilyV2):
            if llm_family.model_family not in VLLM_SUPPORTED_VISION_MODEL_LIST:
                return False
        else:
            if llm_family.model_name not in VLLM_SUPPORTED_VISION_MODEL_LIST:
                return False
        if "vision" not in llm_family.model_ability:
            return False
        return VLLM_INSTALLED

    def _sanitize_model_config(
        self, model_config: Optional[VLLMModelConfig]
    ) -> VLLMModelConfig:
        model_config = super()._sanitize_model_config(model_config)
        if vllm.__version__ >= "0.5.5":
            model_config["limit_mm_per_prompt"] = (
                json.loads(model_config.get("limit_mm_per_prompt"))  # type: ignore
                if model_config.get("limit_mm_per_prompt")
                else {
                    "image": 2,  # default 2 images all chat
                }
            )
        return model_config

    def _sanitize_chat_config(
        self,
        generate_config: Optional[Dict] = None,
    ) -> Dict:
        from ..utils import get_stop_token_ids_from_config_file

        if not generate_config:
            generate_config = {}
        if generate_config.get("stop_token_ids", None) is None:
            stop_token_ids = get_stop_token_ids_from_config_file(self.model_path)
            if stop_token_ids is not None:
                generate_config.setdefault("stop_token_ids", stop_token_ids)
            else:
                if self.model_family.stop_token_ids:
                    generate_config.setdefault(
                        "stop_token_ids", self.model_family.stop_token_ids.copy()
                    )
        return generate_config

    async def _gen_tokens_prompt(
        self, tokenizer, prompt: Union[str, dict], config: dict
    ):
        from vllm import TokensPrompt

        if isinstance(prompt, str):
            return super()._gen_tokens_prompt(tokenizer, prompt, config)

        prompt_str = prompt["prompt"]
        multi_modal_data = prompt.get("multi_modal_data")

        token_ids = await asyncio.to_thread(
            self._tokenize, tokenizer, prompt_str, config  # type: ignore
        )
        return TokensPrompt(
            prompt_token_ids=token_ids, multi_modal_data=multi_modal_data
        )

    @vllm_check
    async def async_chat(
        self,
        messages: List[ChatCompletionMessage],  # type: ignore
        generate_config: Optional[Dict] = None,
        request_id: Optional[str] = None,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        tools = generate_config.pop("tools", []) if generate_config else None

        model_family = self.model_family.model_family or self.model_family.model_name

        if "internvl" not in model_family.lower():
            from qwen_vl_utils import process_vision_info

            messages = self._transform_messages(messages)

            chat_template_kwargs = (
                self._get_chat_template_kwargs_from_generate_config(
                    generate_config, self.reasoning_parser
                )
                or {}
            )
            chat_context_var.set(chat_template_kwargs)
            full_context_kwargs = chat_template_kwargs.copy()
            if tools and model_family in QWEN_TOOL_CALL_FAMILY:
                full_context_kwargs["tools"] = tools
            assert self.model_family.chat_template is not None
            prompt = self.get_full_context(
                messages, self.model_family.chat_template, **full_context_kwargs
            )
            images, video_inputs = process_vision_info(messages)
            if video_inputs:
                raise ValueError("Not support video input now.")
        else:
            prompt, images = self.get_specific_prompt(model_family, messages)

        if not images:
            inputs = {
                "prompt": prompt,
            }
        elif len(images) == 1:
            inputs = {
                "prompt": prompt,
                "multi_modal_data": {"image": images[-1]},  # type: ignore
            }
        else:
            inputs = {
                "prompt": prompt,
                "multi_modal_data": {"image": images},  # type: ignore
            }
        generate_config = self._sanitize_chat_config(generate_config)

        stream = generate_config.get("stream", None)

        if stream:
            agen = await self.async_generate(
                inputs, generate_config, request_id=request_id
            )
            assert isinstance(agen, AsyncGenerator)
            return self._async_to_chat_completion_chunks(agen)
        else:
            c = await self.async_generate(
                inputs, generate_config, request_id=request_id
            )
            assert not isinstance(c, AsyncGenerator)
            return self._to_chat_completion(c)
