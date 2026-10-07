import asyncio
from contextlib import asynccontextmanager
import logging
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import torch
from lib.config.config_utils import load_config, save_config
from lib.configurator.configure_model_capabilities import (
    load_model_capabilities_config,
    save_model_capabilities_config,
)
from lib.configurator.configure_active_ai import (
    choose_active_models,
    load_active_ai_config,
    load_active_ai_models,
    save_active_ai_config,
)
from lib.logging.logger import setup_logger
from lib.pipeline.pipeline_manager import PipelineManager
from lib.server.exceptions import NoActiveModelsException, ServerStopException
from lib.utils.memory_utils import clear_gpu_cache
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import requests

def get_latest_release_version(logger):
    api_url = "https://api.github.com/repos/skier233/nsfw_ai_model_server/releases/latest"
    try:
        response = requests.get(api_url)
        latest_release = response.json()
        return latest_release['tag_name']
    except:
        logger.warning("Failed to get the latest release version from GitHub to see if an update is needed. Are you offline?")

class ServerManager:
    def __init__(self):
        config_path = "./config/config.yaml"
        if os.path.exists(config_path):
            config = load_config(config_path, default_config={})
        else:
            raise ServerStopException(f"Main config file does not exist: {config_path}")
        loglevel = config.get("loglevel", "INFO")
        setup_logger("logger", loglevel)
        self.logger = logging.getLogger("logger")
        self.port = config.get("port", 8000)

        self.config = config
        self.pipeline_manager = PipelineManager()
        self.reload_lock = asyncio.Lock()
        self.reloading = False
        self.default_image_pipeline = config.get("default_image_pipeline", None)
        if self.default_image_pipeline is None:
            self.logger.error("No default image pipeline found in the configuration file.")
            raise ServerStopException("No default image pipeline found in the configuration file.")
        
        self.default_video_pipeline = config.get("default_video_pipeline", None)
        if self.default_video_pipeline is None:
            self.logger.error("No default video pipeline found in the configuration file.")
            raise ServerStopException("No default video pipeline found in the configuration file.")

        self.default_audio_pipeline = config.get("default_audio_pipeline", "audio_pipeline_v4")

    def _check_version(self):
        version_path = "./config/version.yaml"
        versionconfig = {}
        if os.path.exists(version_path):
            versionconfig = load_config(version_path, default_config={})
        version = versionconfig.get("VERSION", "1.3.4")

        latest_version = get_latest_release_version(self.logger)
        self.logger.debug(f"Current version: {version}, Latest version: {latest_version}")
        if version != latest_version:
            self.logger.warning("There is a new version available! Please update the server using install/update.sh or install/update.ps1")

    async def startup(self):
        # Run the GitHub version check once at server startup (main process only)
        # rather than in __init__, which would re-run it in every spawned worker.
        self._check_version()
        pipelines = self.config["active_pipelines"]
        if not pipelines:
            self.logger.error("No pipelines found in the configuration file.")
            raise ServerStopException("No pipelines found in the configuration file.")

        if not load_active_ai_models():
            self.logger.warning("No active AI models configured; starting model management endpoints without loaded AI pipelines.")
            self.background_task = asyncio.create_task(check_inactivity())
            return

        try:
            await self.pipeline_manager.load_pipelines(pipelines)
        except NoActiveModelsException as e:
            self.logger.error(f"Error: No active AI models found in active_ai.yaml")
            try:
                choose_active_models()
            except Exception as e:
                self.logger.debug(f"Error: {e}")
            raise ServerStopException("No active AI models. Choose models in select_ai_models.ps1/sh and start the server again.")
        self.logger.info("Pipelines loaded successfully")
        self.background_task = asyncio.create_task(check_inactivity())

    async def get_request_future(self, data, pipeline_name):
        if self.reloading:
            raise RuntimeError("Model pipelines are reloading")
        if not self.pipeline_manager.has_pipeline(pipeline_name):
            await self._ensure_pipelines_loaded(pipeline_name)
        if self.reloading:
            raise RuntimeError("Model pipelines are reloading")
        return await self.pipeline_manager.get_request_future(data, pipeline_name)

    async def _ensure_pipelines_loaded(self, pipeline_name=None):
        if pipeline_name is not None and self.pipeline_manager.has_pipeline(pipeline_name):
            return
        if not load_active_ai_models():
            return

        async with self.reload_lock:
            if pipeline_name is not None and self.pipeline_manager.has_pipeline(pipeline_name):
                return
            if not load_active_ai_models():
                return

            self.reloading = True
            try:
                await self._reload_pipelines(self.config["active_pipelines"], wait_for_idle=False)
            finally:
                self.reloading = False

    async def update_active_models(self, active_ai_models):
        async with self.reload_lock:
            previous_config = load_active_ai_config()
            next_config = {"active_ai_models": list(active_ai_models or [])}

            pipelines = self.config["active_pipelines"]
            self.reloading = True
            try:
                save_active_ai_config(next_config)
                await self._wait_for_idle_requests(max_outstanding=1)
                if _has_active_ai_models(next_config):
                    if self.pipeline_manager.pipelines:
                        await self.pipeline_manager.reconfigure_pipelines(pipelines)
                    else:
                        await self.pipeline_manager.load_pipelines(pipelines)
                    clear_gpu_cache()
                else:
                    await self._unload_pipelines(wait_for_idle=False)
                self.logger.info("Updated active AI models successfully")
                return next_config
            except Exception:
                self.logger.error("Reload failed; attempting to restore previous active AI configuration")
                self.logger.debug("Exception details:", exc_info=True)
                save_active_ai_config({"active_ai_models": list(previous_config.get("active_ai_models", []) or [])})

                if _has_active_ai_models(previous_config):
                    await self._reload_pipelines(pipelines)
                else:
                    await self._unload_pipelines()
                raise
            finally:
                self.reloading = False

    async def reload_active_models(self, active_ai_models=None):
        async with self.reload_lock:
            previous_config = load_active_ai_config()
            next_config = {"active_ai_models": list(previous_config.get("active_ai_models", []) or [])}
            if active_ai_models is not None:
                next_config["active_ai_models"] = list(active_ai_models)

            pipelines = self.config["active_pipelines"]
            self.reloading = True
            try:
                save_active_ai_config(next_config)
                await self._wait_for_idle_requests(max_outstanding=1)
                if _has_active_ai_models(next_config):
                    await self._reload_pipelines(pipelines, wait_for_idle=False)
                else:
                    await self._unload_pipelines(wait_for_idle=False)
                self.logger.info("Reloaded active AI models successfully")
                return next_config
            except Exception:
                self.logger.error("Reload failed; attempting to restore previous active AI configuration")
                self.logger.debug("Exception details:", exc_info=True)
                save_active_ai_config({"active_ai_models": list(previous_config.get("active_ai_models", []) or [])})

                if _has_active_ai_models(previous_config):
                    await self._reload_pipelines(pipelines)
                else:
                    await self._unload_pipelines()
                raise
            finally:
                self.reloading = False

    async def register_custom_pipeline(self, definition):
        pipeline_name = _validate_pipeline_name(definition.pipeline_name)
        media_kind = _normalize_media_kind(definition.media_kind)
        pipeline_config = _build_dynamic_pipeline_config(pipeline_name, media_kind)
        capability_entry = _build_model_capability_entry(definition, media_kind)
        pipeline_path = f"./config/pipelines/{pipeline_name}.yaml"
        config_path = "./config/config.yaml"

        async with self.reload_lock:
            previous_main_config = dict(self.config)
            previous_capabilities = load_model_capabilities_config()
            previous_pipeline_config = load_config(pipeline_path, default_config={}) if os.path.exists(pipeline_path) else None
            pipeline_previously_existed = os.path.exists(pipeline_path)

            next_main_config = dict(previous_main_config)
            active_pipelines = list(next_main_config.get("active_pipelines", []) or [])
            if pipeline_name not in active_pipelines:
                active_pipelines.append(pipeline_name)
            next_main_config["active_pipelines"] = active_pipelines

            next_capabilities = dict(previous_capabilities or {})
            next_capabilities[pipeline_name] = capability_entry

            self.reloading = True
            try:
                save_config(pipeline_path, pipeline_config)
                save_config(config_path, next_main_config)
                save_model_capabilities_config(next_capabilities)

                await self._reload_pipelines(active_pipelines)
                self.config = next_main_config
                self.logger.info(f"Registered custom pipeline {pipeline_name}")
                return {
                    "pipeline_name": pipeline_name,
                    "media_kind": media_kind,
                    "reloaded": True,
                    "loaded_pipelines": sorted(self.pipeline_manager.pipelines.keys()),
                }
            except Exception:
                self.logger.error(f"Custom pipeline registration failed for {pipeline_name}; rolling back")
                self.logger.debug("Exception details:", exc_info=True)
                save_config(config_path, previous_main_config)
                save_model_capabilities_config(previous_capabilities)
                if pipeline_previously_existed:
                    save_config(pipeline_path, previous_pipeline_config)
                elif os.path.exists(pipeline_path):
                    os.remove(pipeline_path)
                await self._reload_pipelines(previous_main_config.get("active_pipelines", []) or [])
                self.config = previous_main_config
                raise
            finally:
                self.reloading = False

    async def delete_custom_pipeline(self, pipeline_name):
        pipeline_name = _validate_pipeline_name(pipeline_name)
        pipeline_path = f"./config/pipelines/{pipeline_name}.yaml"
        config_path = "./config/config.yaml"

        async with self.reload_lock:
            previous_main_config = dict(self.config)
            previous_capabilities = load_model_capabilities_config()
            previous_pipeline_config = load_config(pipeline_path, default_config={}) if os.path.exists(pipeline_path) else None
            pipeline_previously_existed = os.path.exists(pipeline_path)

            next_main_config = dict(previous_main_config)
            next_main_config["active_pipelines"] = [
                item for item in list(next_main_config.get("active_pipelines", []) or []) if item != pipeline_name
            ]
            next_capabilities = dict(previous_capabilities or {})
            next_capabilities.pop(pipeline_name, None)

            self.reloading = True
            try:
                save_config(config_path, next_main_config)
                save_model_capabilities_config(next_capabilities)
                if os.path.exists(pipeline_path):
                    os.remove(pipeline_path)

                await self._reload_pipelines(next_main_config.get("active_pipelines", []) or [])
                self.config = next_main_config
                self.logger.info(f"Deleted custom pipeline {pipeline_name}")
                return {
                    "pipeline_name": pipeline_name,
                    "media_kind": "",
                    "reloaded": True,
                    "loaded_pipelines": sorted(self.pipeline_manager.pipelines.keys()),
                }
            except Exception:
                self.logger.error(f"Custom pipeline delete failed for {pipeline_name}; rolling back")
                self.logger.debug("Exception details:", exc_info=True)
                save_config(config_path, previous_main_config)
                save_model_capabilities_config(previous_capabilities)
                if pipeline_previously_existed:
                    save_config(pipeline_path, previous_pipeline_config)
                await self._reload_pipelines(previous_main_config.get("active_pipelines", []) or [])
                self.config = previous_main_config
                raise
            finally:
                self.reloading = False

    async def _reload_pipelines(self, pipelines, wait_for_idle=True):
        if wait_for_idle:
            await self._wait_for_idle_requests(max_outstanding=1)
        await self.pipeline_manager.stop_pipelines()
        clear_gpu_cache()
        replacement_manager = PipelineManager()
        await replacement_manager.load_pipelines(pipelines)
        self.pipeline_manager = replacement_manager
        clear_gpu_cache()

    async def _unload_pipelines(self, wait_for_idle=True):
        if wait_for_idle:
            await self._wait_for_idle_requests(max_outstanding=1)
        await self.pipeline_manager.stop_pipelines()
        self.pipeline_manager = PipelineManager()
        clear_gpu_cache()

    async def _wait_for_idle_requests(self, timeout_seconds=30.0, max_outstanding=0):
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        while outstanding_requests_middleware.outstanding_requests > max_outstanding:
            if loop.time() - started_at >= timeout_seconds:
                raise RuntimeError("Timed out waiting for in-flight requests to complete before reload")
            await asyncio.sleep(0.05)


def _validate_pipeline_name(pipeline_name):
    normalized = str(pipeline_name or "").strip()
    if not normalized:
        raise ValueError("pipeline_name is required")
    if any(not (ch.isalnum() or ch in "_.-") for ch in normalized):
        raise ValueError("pipeline_name may only contain letters, digits, '.', '_' and '-'")
    return normalized


def _normalize_media_kind(media_kind):
    normalized = str(media_kind or "").strip().lower()
    if normalized not in {"image", "video", "audio"}:
        raise ValueError("media_kind must be one of: image, video, audio")
    return normalized


def _build_dynamic_pipeline_config(pipeline_name, media_kind):
    if media_kind == "image":
        return {
            "inputs": ["image_path", "threshold", "return_confidence", "skipped_categories", "requested_model_names"],
            "output": "result",
            "short_name": pipeline_name,
            "version": 4.0,
            "models": [
                {
                    "name": "dynamic_image_ai",
                    "inputs": ["image_path", "threshold", "return_confidence", "skipped_categories", "requested_model_names"],
                    "outputs": ["result_early"],
                },
                {"name": "image_result_postprocessor_v4", "inputs": ["result_early"], "outputs": ["result"]},
            ],
        }
    if media_kind == "video":
        return {
            "inputs": [
                "video_path",
                "return_timestamps",
                "time_interval",
                "threshold",
                "return_confidence",
                "vr_video",
                "skipped_categories",
                "requested_model_names",
            ],
            "output": "results",
            "short_name": pipeline_name,
            "version": 4.1,
            "models": [
                {
                    "name": "dynamic_video_ai",
                    "inputs": [
                        "video_path",
                        "return_timestamps",
                        "time_interval",
                        "threshold",
                        "return_confidence",
                        "vr_video",
                        "skipped_categories",
                        "requested_model_names",
                    ],
                    "outputs": ["childrenResults"],
                },
                {
                    "name": "dynamic_asset_ai",
                    "inputs": ["video_path", "vr_video", "skipped_categories", "requested_model_names"],
                    "outputs": ["assetResults"],
                },
                {
                    "name": "video_result_postprocessor_v4",
                    "inputs": ["childrenResults", "video_path", "time_interval", "assetResults"],
                    "outputs": ["results"],
                },
            ],
        }
    return {
        "inputs": ["audio_path", "threshold", "requested_model_names"],
        "output": "result",
        "short_name": pipeline_name,
        "version": 4.0,
        "models": [
            {"name": "dynamic_audio_ai", "inputs": ["audio_path", "threshold", "requested_model_names"], "outputs": ["childrenResults"]},
            {"name": "audio_result_postprocessor_v4", "inputs": ["childrenResults", "audio_path"], "outputs": ["result"]},
        ],
    }


def _build_model_capability_entry(definition, media_kind):
    if media_kind == "audio":
        audio_models = _normalize_model_list(definition.audio_models)
        return {"audio_models": "ALL" if definition.use_all_audio_models else audio_models}

    full_image_models = _normalize_model_list(definition.full_image_models)
    detector_models = _normalize_model_list(definition.detector_models)
    region_models = {}
    for detector_name, model_names in (definition.region_models or {}).items():
        detector_name = str(detector_name or "").strip()
        normalized_models = _normalize_model_list(model_names)
        if detector_name and normalized_models:
            region_models[detector_name] = normalized_models

    entry = {
        "full_image_models": "ALL" if definition.use_all_full_image_models else full_image_models,
        "detector_models": detector_models,
        "region_models": region_models,
    }
    if media_kind == "video":
        asset_models = _normalize_model_list(definition.asset_models)
        entry["asset_models"] = "ALL" if definition.use_all_asset_models else asset_models
    return entry


def _normalize_model_list(value):
    normalized = []
    for item in value or []:
        text = str(item or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized
    

@asynccontextmanager
async def lifespan(app: FastAPI):
    await server_manager.startup()
    yield
    pass

class OutstandingRequestsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.outstanding_requests = 0
        self.last_request_timestamp = asyncio.get_event_loop().time()

    async def dispatch(self, request: Request, call_next):
        self.outstanding_requests += 1
        self.last_request_timestamp = asyncio.get_event_loop().time()
        try:
            response = await call_next(request)
        except Exception as e:
            response = JSONResponse({"detail": str(e)}, status_code=500)
        self.outstanding_requests -= 1
        return response
    
server_manager = ServerManager()
port = server_manager.port
app = FastAPI(lifespan=lifespan)

outstanding_requests_middleware = OutstandingRequestsMiddleware(app)
app.add_middleware(BaseHTTPMiddleware, dispatch=outstanding_requests_middleware.dispatch)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    try:
        response = await call_next(request)
    except Exception as e:
        response = JSONResponse({"detail": str(e)}, status_code=500)
    response.headers["X-Outstanding-Requests"] = str(outstanding_requests_middleware.outstanding_requests)
    return response

last_request_timestamp = 0

async def check_inactivity():
    global last_request_timestamp
    while True:
        await asyncio.sleep(300)  # check every 5 minutes
        middleware = outstanding_requests_middleware
        if middleware.outstanding_requests == 0 and asyncio.get_event_loop().time() - middleware.last_request_timestamp > 300 and middleware.last_request_timestamp > last_request_timestamp:
            last_request_timestamp = middleware.last_request_timestamp
            if server_manager.reloading:
                continue

            print("No requests in the last 5 minutes, Clearing cached memory")
            clear_gpu_cache()


def _has_active_ai_models(config):
    return bool(config.get("active_ai_models", []) or [])

origins = [
    "*",  # Replace with the actual origins you need
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods, including OPTIONS
    allow_headers=["*"],
)

@app.exception_handler(ServerStopException)
async def server_stop_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": f"Server is stopping due to error: {exc.message}"},
    )
