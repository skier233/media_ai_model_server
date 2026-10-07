from pathlib import Path
from typing import Iterable

from lib.configurator.configure_active_ai import (
    load_active_ai_models,
    load_available_ai_models,
)
from lib.model.whole_asset import is_whole_asset_model


VALID_LOAD_POLICIES = {"use_loaded", "load_if_cheap", "load_or_fail"}


def get_model_catalog(server_manager):
    active_model_names = set(load_active_ai_models())
    available_models = load_available_ai_models()
    available_by_name = {model.get("yaml_file_name"): model for model in available_models}
    active_categories = _get_active_categories(active_model_names, available_by_name)
    loaded_model_names = set()
    loaded_model_file_names = set()

    for pipeline in server_manager.pipeline_manager.pipelines.values():
        for model_info in pipeline.get_ai_models_info():
            if model_info.config_name:
                loaded_model_names.add(model_info.config_name)
            loaded_model_file_names.add(model_info.name)

    catalog = []
    for model_config in available_models:
        config_name = model_config.get("yaml_file_name")
        file_name = model_config.get("model_file_name") or config_name
        categories = _normalize_string_list(model_config.get("model_category")) or []
        category_conflicts = _get_category_conflicts(config_name, categories, active_model_names, active_categories)
        artifact_available, artifact_path = _model_artifact_status(model_config)
        incompatibility_reasons = []
        if category_conflicts:
            incompatibility_reasons.append(f"Category already active: {', '.join(category_conflicts)}")
        if not artifact_available:
            incompatibility_reasons.append(f"Model file missing: {artifact_path}")
        config_problem = _model_config_problem(model_config)
        if config_problem:
            incompatibility_reasons.append(config_problem)
        entry = {
            "config_name": config_name,
            "name": file_name,
            "identifier": model_config.get("model_identifier"),
            "version": model_config.get("model_version"),
            "categories": categories,
            "type": model_config.get("model_type", model_config.get("type")),
            "info": model_config.get("model_info"),
            "image_size": model_config.get("model_image_size"),
            "artifact_available": artifact_available,
            "artifact_path": artifact_path,
            "incompatible": bool(incompatibility_reasons) and config_name not in active_model_names,
            "incompatibility_reason": "; ".join(incompatibility_reasons),
            "capabilities": _resolve_capabilities(model_config),
            "supported_scopes": _resolve_supported_scopes(model_config),
            # Analyses a whole video and drives its own decode (e.g. shot boundaries).
            "whole_asset": is_whole_asset_model(model_config),
            "active": config_name in active_model_names,
            "loaded": config_name in loaded_model_names or file_name in loaded_model_file_names,
        }
        catalog.append(entry)

    catalog.sort(key=lambda item: (item["active"] is False, (item.get("categories") or [""])[0], item["config_name"]))
    return catalog


def get_loaded_models(server_manager):
    catalog = get_model_catalog(server_manager)
    return [entry for entry in catalog if entry["active"] or entry["loaded"]]


def get_capability_catalog(server_manager):
    capability_index = {}

    for entry in get_model_catalog(server_manager):
        for capability in entry.get("capabilities") or []:
            capability_entry = capability_index.setdefault(
                capability,
                {
                    "capability": capability,
                    "supported_scopes": set(),
                    "models": [],
                },
            )
            capability_entry["supported_scopes"].update(entry.get("supported_scopes") or [])
            capability_entry["models"].append(
                {
                    "config_name": entry["config_name"],
                    "name": entry["name"],
                    "type": entry.get("type"),
                    "loaded": entry.get("loaded", False),
                    "active": entry.get("active", False),
                }
            )

    capabilities = []
    for capability_name in sorted(capability_index):
        capability_entry = capability_index[capability_name]
        capability_entry["supported_scopes"] = sorted(capability_entry["supported_scopes"])
        capability_entry["models"].sort(key=lambda item: (item["loaded"] is False, item["config_name"]))
        capabilities.append(capability_entry)

    return capabilities


def filter_pipeline_models(pipeline, requested_model_names=None):
    ai_models = pipeline.get_ai_models_info()
    requested = set(_normalize_string_list(requested_model_names) or [])
    if not requested:
        return ai_models
    filtered = []
    for model_info in ai_models:
        if model_info.config_name in requested or model_info.name in requested:
            filtered.append(model_info)
    return filtered


async def resolve_request_models(server_manager, want, default_scope, load_policy):
    if load_policy not in VALID_LOAD_POLICIES:
        raise ValueError(f"Unsupported load_policy '{load_policy}'. Expected one of {sorted(VALID_LOAD_POLICIES)}")

    requested_model_names = resolve_want_model_names(server_manager, want, default_scope)
    if not requested_model_names:
        return [], False

    active_model_names = set(load_active_ai_models())
    missing_models = [name for name in requested_model_names if name not in active_model_names]
    if not missing_models:
        return requested_model_names, False

    if load_policy == "use_loaded":
        raise RuntimeError(f"Requested models are not active: {missing_models}")

    next_active = list(active_model_names)
    next_active.extend(missing_models)
    next_active = _dedupe(next_active)
    _validate_models_can_activate(missing_models, active_model_names)
    await server_manager.update_active_models(active_ai_models=next_active)
    return requested_model_names, True


async def load_models(server_manager, model_names):
    requested = _validate_model_names(model_names, server_manager)
    current_active = load_active_ai_models()
    _validate_models_can_activate(requested, current_active)
    active_model_names = _dedupe(current_active + requested)
    await server_manager.update_active_models(active_ai_models=active_model_names)
    return get_loaded_models(server_manager)


async def unload_models(server_manager, model_names):
    requested = _validate_model_names(model_names, server_manager)
    active_model_names = [name for name in load_active_ai_models() if name not in requested]
    await server_manager.update_active_models(active_ai_models=_dedupe(active_model_names))
    return get_loaded_models(server_manager)


def resolve_want_model_names(server_manager, want, default_scope):
    want = want or []
    if not want:
        return []

    catalog = get_model_catalog(server_manager)
    catalog_by_name = {entry["config_name"]: entry for entry in catalog}
    requested_model_names = []

    for item in want:
        explicit_models = _normalize_string_list(getattr(item, "models", None)) or []
        if explicit_models:
            for model_name in explicit_models:
                if model_name not in catalog_by_name:
                    raise ValueError(f"Unknown model '{model_name}'")
            requested_model_names.extend(explicit_models)

            # When a want item names explicit models, honor that exact set
            # instead of broadening the request by capability/scope.
            continue

        requested_capabilities = set(_normalize_string_list(getattr(item, "capability", None)) or [])
        requested_capabilities.update(_normalize_string_list(getattr(item, "capabilities", None)) or [])
        requested_categories = set(_normalize_string_list(getattr(item, "category", None)) or [])
        requested_categories.update(_normalize_string_list(getattr(item, "categories", None)) or [])
        requested_categories.update(_normalize_string_list(getattr(item, "model_category", None)) or [])
        requested_scopes = set(_normalize_string_list(getattr(item, "scope", None)) or [])
        requested_scopes.update(_normalize_string_list(getattr(item, "scopes", None)) or [])
        # On a video, an explicit capability or category ask must reach the
        # models that analyse the whole video as well as the per-frame ones;
        # the frame default alone would silently exclude them. Every other
        # request keeps its default scope, and only a video request can match
        # a whole-video model at all.
        video_request = default_scope == "frame"
        video_ask = (
            video_request
            and not requested_scopes
            and bool(requested_capabilities or requested_categories)
        )
        if not requested_scopes and default_scope and not video_ask:
            requested_scopes.add(default_scope)

        matched = []
        if requested_capabilities or requested_categories or requested_scopes:
            for entry in catalog:
                if requested_capabilities and requested_capabilities.isdisjoint(entry["capabilities"]):
                    continue
                if requested_categories and requested_categories.isdisjoint(entry["categories"]):
                    continue
                if requested_scopes and requested_scopes.isdisjoint(entry["supported_scopes"]):
                    continue
                if entry.get("whole_asset") and not video_request:
                    continue
                if video_ask and not ("frame" in entry["supported_scopes"] or entry.get("whole_asset")):
                    continue
                matched.append(entry["config_name"])
            if not matched and not explicit_models:
                raise ValueError(
                    f"No models matched want item capabilities={sorted(requested_capabilities)} "
                    f"categories={sorted(requested_categories)} scopes={sorted(requested_scopes)}"
                )
            requested_model_names.extend(matched)

    return _dedupe(requested_model_names)


def expand_requested_models_with_dependencies(server_manager, pipeline_name, requested_model_names):
    """Ensure a requested region/embedding model also runs the detector that feeds it.

    Region-scoped models (e.g. a face embedder) consume a detector's per-frame
    output within the same analysis pass — the server is stateless and does not
    reuse previously stored detections as input. If a request names only the
    region model, the per-model skip gate (async_processing.batch_data_append_with_skips)
    skips the upstream detector, so the region model runs with no detections to
    process and produces nothing.

    Expand the requested set to include the upstream detector for every requested
    region model, using the pipeline's actual loaded-model wiring (config_name
    based, via model_capabilities). This works regardless of whether the caller
    knew the loaded detector's model name.
    """
    if not requested_model_names:
        return requested_model_names

    try:
        dynamic_ai_manager = server_manager.pipeline_manager.dynamic_ai_manager
        region_rules = dynamic_ai_manager.model_capabilities.get_region_model_rules(
            pipeline_name, available_models=dynamic_ai_manager.models
        ) or []
    except Exception:
        return requested_model_names

    if not region_rules:
        return requested_model_names

    requested_set = set(requested_model_names)
    additions = []
    for rule in region_rules:
        detector_name = rule.get("key")
        region_models = rule.get("models") or []
        if not detector_name or detector_name in requested_set:
            continue
        if any(region_model in requested_set for region_model in region_models):
            additions.append(detector_name)
            requested_set.add(detector_name)

    if not additions:
        return requested_model_names

    return _dedupe(list(requested_model_names) + additions)


def _get_available_models_by_name():
    return {model.get("yaml_file_name"): model for model in load_available_ai_models()}


def _get_active_categories(active_model_names, available_by_name):
    categories = set()
    for model_name in active_model_names:
        model_config = available_by_name.get(model_name)
        if not model_config:
            continue
        categories.update(_normalize_string_list(model_config.get("model_category")) or [])
    return categories


def _get_category_conflicts(config_name, categories, active_model_names, active_categories):
    if config_name in active_model_names:
        return []
    return sorted(category for category in categories if category in active_categories)


def _validate_models_can_activate(model_names, current_active_model_names):
    available_by_name = _get_available_models_by_name()
    active_model_names = set(current_active_model_names or [])
    active_categories = _get_active_categories(active_model_names, available_by_name)

    for model_name in model_names:
        if model_name in active_model_names:
            continue
        model_config = available_by_name.get(model_name)
        if not model_config:
            raise ValueError(f"Unknown model '{model_name}'")

        categories = _normalize_string_list(model_config.get("model_category")) or []
        conflicts = _get_category_conflicts(model_name, categories, active_model_names, active_categories)
        if conflicts:
            raise ValueError(f"Cannot activate model '{model_name}': category already active: {', '.join(conflicts)}")

        artifact_available, artifact_path = _model_artifact_status(model_config)
        if not artifact_available:
            raise ValueError(f"Cannot activate model '{model_name}': model file missing: {artifact_path}")
        config_problem = _model_config_problem(model_config)
        if config_problem:
            raise ValueError(f"Cannot activate model '{model_name}': {config_problem}")

        active_model_names.add(model_name)
        active_categories.update(categories)


def _model_artifact_status(model_config):
    model_file_name = model_config.get("model_file_name") or model_config.get("yaml_file_name")
    if not model_file_name:
        return False, "model file name is not configured"

    license_name = model_config.get("model_license_name")
    if license_name:
        model_path = Path("./models") / f"{model_file_name}{_encrypted_model_extension(license_name)}"
        return model_path.exists(), str(model_path)

    candidates = [Path("./models") / f"{model_file_name}.pt2", Path("./models") / f"{model_file_name}.pt"]
    for candidate in candidates:
        if candidate.exists():
            if is_whole_asset_model(model_config):
                # A shot-boundary artifact is unusable without its label map.
                sidecar = Path("./models") / f"{model_file_name}.labels.json"
                if not sidecar.exists():
                    return False, str(sidecar)
            return True, str(candidate)
    return False, str(candidates[0])


def _model_config_problem(model_config):
    """Why a model's configuration cannot work, or None."""
    if is_whole_asset_model(model_config) and model_config.get("model_license_name"):
        # The licensed runner cannot be asked for float32, which this model needs.
        return "Shot-boundary artifacts must be unencrypted"
    return None


def _encrypted_model_extension(license_name):
    parts = str(license_name or "").split(".")
    if len(parts) == 2:
        minor_part = parts[-1]
    elif len(parts) >= 3:
        minor_part = parts[-2]
    else:
        minor_part = None
    try:
        minor = int(minor_part) if minor_part is not None else 1
    except ValueError:
        minor = 1
    return {0: ".pt.enc", 1: ".pt2.enc", 2: ".ep.enc"}.get(minor, ".pt2.enc")


def _validate_model_names(model_names, server_manager):
    normalized = _normalize_string_list(model_names) or []
    catalog_names = {entry["config_name"] for entry in get_model_catalog(server_manager)}
    unknown = [name for name in normalized if name not in catalog_names]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}")
    return normalized


def _normalize_string_list(raw_value):
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


def _resolve_capabilities(model_config):
    capabilities = _normalize_string_list(model_config.get("model_capabilities", model_config.get("capabilities")))
    if capabilities:
        return capabilities
    model_type = str(model_config.get("model_type", "")).lower()
    if "embed" in model_type:
        return ["embedding"]
    if "detect" in model_type:
        return ["detection"]
    if "classif" in model_type:
        return ["classification"]
    return ["tagging"]


def _resolve_supported_scopes(model_config):
    scopes = _normalize_string_list(model_config.get("supported_target_scopes", model_config.get("target_scopes")))
    return scopes or ["asset", "frame", "region"]


def _dedupe(values):
    deduped = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped