
import logging
import time
import warnings
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Iterable, List
from lib.model.model import Model
from lib.model.ai_model_python.python_model import PythonModel
from lib.utils.torch_device_selector import get_device_string

# Suppress the one-time "buffer is not writable" warning emitted by
# torch.export.load → torch.frombuffer when loading .pt2 archives.
# The warning is harmless (PyTorch makes the buffer writable internally)
# and cannot be caught with a context manager because it originates in C++.
warnings.filterwarnings("ignore", message=".*buffer is not writable.*", category=UserWarning)


DEFAULT_MODEL_CAPABILITY = "tagging"
DEFAULT_SUPPORTED_TARGET_SCOPES = ["asset", "frame", "region"]
VALID_MODEL_CAPABILITIES = {"tagging", "detection", "embedding", "classification", "temporal_segmentation"}
VALID_TARGET_SCOPES = {"asset", "frame", "region"}

class AIModel(Model):
    """Base class for all AI models (tagging, detection, embedding).

    Owns shared lifecycle: config parsing, device selection, batch sizing,
    model loading (encrypted via ModelRunner or unencrypted via PythonModel),
    and inference dispatch.

    Subclasses implement worker_function() with domain-specific logic.
    """

    def __init__(self, configValues, keep_on_device=False):
        Model.__init__(self, configValues)
        self.max_model_batch_size = configValues.get("max_model_batch_size", 12)
        # Immutable copy of the config cap — stored before update_batch_with_mutli_models
        # can overwrite max_model_batch_size via the legacy bspv path.
        # compute_batch_sizes reads this field for per-model caps.
        self._config_max_batch_cap = configValues.get("max_model_batch_size", None)
        self.batch_size_per_VRAM_GB = configValues.get("batch_size_per_VRAM_GB", None)
        self.activation_per_item_mb = configValues.get("activation_per_item_mb", None)
        self.model_file_name = configValues.get("model_file_name", None)
        self.model_license_name = configValues.get("model_license_name", None)
        self.device = configValues.get("device", None)
        self.fill_to_batch = configValues.get("fill_to_batch_size", True)
        self.keep_on_device = keep_on_device
        self.model_image_size = configValues.get("model_image_size", None)
        self.model_category = _normalize_string_list(configValues.get("model_category", None))
        self.model_type = configValues.get("model_type", "ImClass")
        self.model_version = configValues.get("model_version", None)
        self.model_identifier = configValues.get("model_identifier", None)
        self.normalization_config = configValues.get("normalization_config", 1)
        self.model_precision = configValues.get("model_precision", None)  # "float32", "float16", "bfloat16", or None
        self.model_capabilities = _resolve_model_capabilities(configValues, self.model_type)
        self.supported_target_scopes = _resolve_supported_target_scopes(configValues)
        # Explicit opt-in/opt-out for full_image_models: ALL resolution.
        # None = use default heuristic (tagging/embedding → included).
        self.full_image_model = configValues.get("full_image_model", None)
        # Optional override for VRAM budget weight estimation (MB).
        # If not set, weight is estimated from the model file size on disk.
        self._model_weight_mb_override = configValues.get("model_weight_mb", None)
        if self.model_file_name is None:
            raise ValueError("model_file_name is required for AI models")
        self.model = None
        if self.device is None:
            self.device = get_device_string()
        self.localdevice = torch.device(self.device)
        self.logger.debug(f"Using device: {self.device}")
    
        if self.device in ["cpu", "mps"]:
            self.batch_size_per_VRAM_GB = None
            self.max_queue_size = None
            self.max_batch_size = 1
            self.max_model_batch_size = 1

        self.update_batch_with_mutli_models(1)
    
    @property
    def inference_dtype(self):
        """Explicit inference dtype from model_precision config, or None."""
        _map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        return _map.get(self.model_precision)

    def update_batch_with_mutli_models(self, model_count):
        batch_multipliers = [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        if self.batch_size_per_VRAM_GB is not None and (torch.cuda.is_available() or torch.xpu.is_available()):
            multiplier_index = min(max(model_count, 1), len(batch_multipliers)) - 1
            batch_size_temp = self.batch_size_per_VRAM_GB * batch_multipliers[multiplier_index]
            if self.device == "cuda":
                gpuMemory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            elif self.device == "xpu":
                gpuMemory = torch.xpu.get_device_properties(0).total_memory / (1024 ** 3)
            scaledBatchSize = custom_round(batch_size_temp * gpuMemory)
            self.max_model_batch_size = scaledBatchSize
            self.max_batch_size = scaledBatchSize
            self.max_queue_size = scaledBatchSize * 4
            self.logger.debug(f"Setting batch size to {scaledBatchSize} based on VRAM size of {gpuMemory} GB for model {self.model_file_name} ({self.model_category})")

    async def load(self):
        if self.model is not None:
            return
        self.logger.info(f"Loading model {self.model_file_name} with batch size {self.max_model_batch_size}, {self.max_queue_size}, {self.max_batch_size}")
        if self.model_license_name is not None:
            from ai_processing import ModelRunner
            # Determine extension from license version minor (X.Y or X.Y.Z):
            #   Y=0 -> .pt.enc, Y=1 -> .pt2.enc, Y=2 -> .ep.enc
            _lic_parts = self.model_license_name.split(".")
            if len(_lic_parts) == 2:
                _minor_part = _lic_parts[-1]
            elif len(_lic_parts) >= 3:
                _minor_part = _lic_parts[-2]
            else:
                _minor_part = None
            try:
                _minor = int(_minor_part) if _minor_part is not None else 1
            except ValueError:
                _minor = 1
            _ext_map = {0: ".pt.enc", 1: ".pt2.enc", 2: ".ep.enc"}
            model_file_path = f"./models/{self.model_file_name}{_ext_map.get(_minor, '.pt2.enc')}"
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*buffer is not writable.*", category=UserWarning)
                self.model = ModelRunner(model_file_path, f"./models/{self.model_license_name}.lic",
                                         self.max_model_batch_size, self.device,
                                         self.fill_to_batch, self.keep_on_device)
        else:
            pt2 = Path(f"./models/{self.model_file_name}.pt2")
            pt = Path(f"./models/{self.model_file_name}.pt")
            if pt2.exists():
                model_path = str(pt2)
            elif pt.exists():
                model_path = str(pt)
            else:
                raise FileNotFoundError(f"No model file found for {self.model_file_name} (.pt2 or .pt)")
            self.model = PythonModel(model_path, self.max_model_batch_size,
                                      self.device, self.fill_to_batch,
                                      self.keep_on_device)

    async def unload(self):
        if self.model is None:
            return
        unload_fn = getattr(self.model, "unload_model", None)
        if callable(unload_fn):
            unload_fn()
        self.model = None

    def run_inference(self, input_tensor, apply_sigmoid=False):
        """Run model inference on input tensor. Returns output tensor."""
        if apply_sigmoid:
            return self.model.process_images(input_tensor)
        return self.model.run_raw(input_tensor)

    async def worker_function(self, data):
        raise NotImplementedError("Subclasses must implement worker_function")

def _normalize_string_list(raw_value) -> List[str] | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        return [raw_value]
    if isinstance(raw_value, Iterable):
        normalized = []
        for item in raw_value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized or None
    return [str(raw_value)]


def _resolve_model_capabilities(config_values, model_type: str) -> List[str]:
    configured = config_values.get("model_capabilities", config_values.get("capabilities", None))
    normalized = _normalize_string_list(configured)
    if normalized:
        valid = [capability for capability in normalized if capability in VALID_MODEL_CAPABILITIES]
        if valid:
            return valid

    model_type_normalized = (model_type or "").lower()
    if "embed" in model_type_normalized:
        return ["embedding"]
    if "detect" in model_type_normalized:
        return ["detection"]
    if "classif" in model_type_normalized:
        return ["classification"]
    return [DEFAULT_MODEL_CAPABILITY]


def _resolve_supported_target_scopes(config_values) -> List[str]:
    configured = config_values.get("supported_target_scopes", config_values.get("target_scopes", None))
    normalized = _normalize_string_list(configured)
    if normalized:
        valid = [scope for scope in normalized if scope in VALID_TARGET_SCOPES]
        if valid:
            return valid
    return DEFAULT_SUPPORTED_TARGET_SCOPES.copy()

def custom_round(value):
    if value < 8:
        return int(value)
    # Calculate the remainder when the value is divided by 8
    remainder = int(value) % 8
    # If the remainder is less than or equal to 4, round down
    if remainder <= 5:
        return int(value) - remainder
    # Otherwise, round up
    else:
        return int(value) + (8 - remainder)
