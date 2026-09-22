from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

from lib.model.whole_asset import is_whole_asset_model


_SELECTOR_KEYS = {
    "capability",
    "capabilities",
    "category",
    "categories",
    "model_category",
    "scope",
    "scopes",
    "target_scope",
    "target_scopes",
    "supported_target_scopes",
    "model_type",
    "model_types",
    "type",
    "types",
    "name",
    "names",
    "model",
    "models",
    "config_name",
    "config_names",
    "exclude",
    "full_image_model",
}


class ModelCapabilitiesConfig:
    def __init__(
        self,
        config: Dict[str, Any],
        active_model_library: Sequence[str],
        logger,
        known_pipelines: Optional[Iterable[str]] = None,
    ):
        self.config = config or {}
        self.active_model_library = set(_normalize_string_list(active_model_library))
        self.known_pipelines = set(_normalize_string_list(known_pipelines))
        self.logger = logger

    def set_known_pipelines(self, pipelines: Optional[Iterable[str]]):
        self.known_pipelines = set(_normalize_string_list(pipelines))

    def prune_unavailable_models(self):
        """Prune unavailable legacy explicit model-name entries in-place.

        Selector-based entries are intentionally left alone here. They are resolved later against
        the loaded active model metadata, which lets a source-controlled config say "models with
        category X/capability Y" without depending on local config names.
        """
        if not isinstance(self.config, dict):
            return

        for pipeline_name, raw_entry in list(self.config.items()):
            if not isinstance(raw_entry, dict):
                continue
            # Before the selector check: the shipped video pipeline selects its
            # detector and region models by capability, which must not leave an
            # explicit asset_models list naming an inactive model unpruned.
            self._prune_asset_models(pipeline_name, raw_entry)
            if _contains_selector(raw_entry):
                continue

            detector_models = _normalize_string_list(raw_entry.get("detector_models", []))
            pruned_detectors = []
            removed_detectors = set()
            for name in detector_models:
                if name in self.active_model_library:
                    pruned_detectors.append(name)
                else:
                    self.logger.warning(
                        f"model_capabilities: pipeline '{pipeline_name}' detector_models references "
                        f"'{name}' which is not in active_ai_models - removing from pipeline (model not downloaded or not active)."
                    )
                    removed_detectors.add(name)

            if removed_detectors:
                raw_entry["detector_models"] = pruned_detectors

            region_model_map = raw_entry.get("region_models", {}) or {}
            if isinstance(region_model_map, dict) and region_model_map:
                pruned_region_map = {}
                for detector_key, region_value in region_model_map.items():
                    detector_key_norm = str(detector_key).strip()

                    if detector_key_norm in removed_detectors:
                        self.logger.warning(
                            f"model_capabilities: pipeline '{pipeline_name}' region_models['{detector_key_norm}'] "
                            f"removed because its detector '{detector_key_norm}' was pruned."
                        )
                        continue

                    if isinstance(region_value, dict):
                        pruned_label_models = {}
                        for label, label_model_names in region_value.items():
                            available = []
                            for model_name in _normalize_string_list(label_model_names):
                                if model_name in self.active_model_library:
                                    available.append(model_name)
                                else:
                                    self.logger.warning(
                                        f"model_capabilities: pipeline '{pipeline_name}' "
                                        f"region_models['{detector_key_norm}']['{label}'] references "
                                        f"'{model_name}' which is not in active_ai_models - removing."
                                    )
                            if available:
                                pruned_label_models[label] = available
                            else:
                                self.logger.warning(
                                    f"model_capabilities: pipeline '{pipeline_name}' "
                                    f"region_models['{detector_key_norm}']['{label}'] has no remaining models - removing label rule."
                                )
                        if pruned_label_models:
                            pruned_region_map[detector_key_norm] = pruned_label_models
                        else:
                            self.logger.warning(
                                f"model_capabilities: pipeline '{pipeline_name}' region_models['{detector_key_norm}'] "
                                f"has no remaining label rules - removing region rule entirely."
                            )
                    else:
                        model_names = _normalize_string_list(region_value)
                        available = []
                        for model_name in model_names:
                            if model_name in self.active_model_library:
                                available.append(model_name)
                            else:
                                self.logger.warning(
                                    f"model_capabilities: pipeline '{pipeline_name}' "
                                    f"region_models['{detector_key_norm}'] references "
                                    f"'{model_name}' which is not in active_ai_models - removing."
                                )
                        if available:
                            pruned_region_map[detector_key_norm] = available
                        else:
                            self.logger.warning(
                                f"model_capabilities: pipeline '{pipeline_name}' region_models['{detector_key_norm}'] "
                                f"has no remaining models - removing region rule entirely."
                            )

                raw_entry["region_models"] = pruned_region_map

            full_image_raw = raw_entry.get("full_image_models", None)
            if isinstance(full_image_raw, list):
                available = []
                for name in _normalize_string_list(full_image_raw):
                    if name in self.active_model_library:
                        available.append(name)
                    else:
                        self.logger.warning(
                            f"model_capabilities: pipeline '{pipeline_name}' full_image_models references "
                            f"'{name}' which is not in active_ai_models - removing."
                        )
                if not available and full_image_raw:
                    self.logger.warning(
                        f"model_capabilities: pipeline '{pipeline_name}' full_image_models list is now empty "
                        f"after pruning - falling back to ALL."
                    )
                    raw_entry["full_image_models"] = "ALL"
                else:
                    raw_entry["full_image_models"] = available

    def _prune_asset_models(self, pipeline_name, raw_entry):
        """Drop the names in an explicit asset_models list that are not active.

        Selector items stay: they are resolved later against the loaded models.
        Unlike full_image_models, an emptied list stays empty: the listed
        whole-video models were chosen deliberately, and falling back to ALL
        would start running others.
        """
        asset_raw = raw_entry.get("asset_models", None)
        if not isinstance(asset_raw, list):
            return
        kept = []
        for item in asset_raw:
            if not isinstance(item, str):
                kept.append(item)
                continue
            name = item.strip()
            if not name:
                continue
            if name in self.active_model_library:
                kept.append(name)
            else:
                self.logger.warning(
                    f"model_capabilities: pipeline '{pipeline_name}' asset_models references "
                    f"'{name}' which is not in active_ai_models - removing."
                )
        raw_entry["asset_models"] = kept

    def validate(self):
        if not isinstance(self.config, dict):
            raise ValueError("model_capabilities config must be a mapping of pipeline_name -> capability settings")

        for pipeline_name, raw_entry in self.config.items():
            pipeline_name = str(pipeline_name).strip()
            if not pipeline_name:
                raise ValueError("model_capabilities contains an empty pipeline name key")
            if self.known_pipelines and pipeline_name not in self.known_pipelines:
                raise ValueError(
                    f"model_capabilities references unknown pipeline '{pipeline_name}'. "
                    f"Known pipelines: {sorted(self.known_pipelines)}"
                )

            if not isinstance(raw_entry, dict):
                raise ValueError(f"model_capabilities entry for pipeline '{pipeline_name}' must be a mapping")

            self._validate_reference_value(raw_entry.get("full_image_models", "ALL"), allow_all=True)
            self._validate_reference_value(raw_entry.get("detector_models", []), allow_all=False)
            if "audio_models" in raw_entry:
                self._validate_reference_value(raw_entry.get("audio_models"), allow_all=True)
            if "asset_models" in raw_entry:
                self._validate_reference_value(raw_entry.get("asset_models"), allow_all=True)
            self._validate_region_models(pipeline_name, raw_entry.get("region_models", {}) or {})

    def resolve_full_image_model_names_for_validation(self, pipeline_name: str) -> List[str]:
        entry = self.config.get(pipeline_name, {}) or {}
        full_image_models = entry.get("full_image_models", "ALL")
        return self._resolve_full_image_model_names(full_image_models, None)

    def get_pipeline_entry(self, pipeline_name: str) -> Dict[str, Any]:
        entry = self.config.get(pipeline_name, None)
        if entry is None:
            return {}
        if not isinstance(entry, dict):
            raise ValueError(f"model_capabilities entry for pipeline '{pipeline_name}' must be a mapping")
        return entry

    def resolve_model_names_for_stage(self, pipeline_name: str, stage: str, available_models: Sequence[Any]) -> Optional[List[str]]:
        entry = self.get_pipeline_entry(pipeline_name)
        if not entry:
            return None

        stage_name = str(stage).strip().lower()
        if stage_name == "detector":
            if "detector_models" not in entry:
                return None
            return self._resolve_reference_names(entry.get("detector_models", []), available_models)

        if stage_name == "region":
            if "region_models" not in entry:
                return None
            ordered = []
            seen = set()
            for rule in self.get_region_model_rules(pipeline_name, available_models=available_models):
                for model_name in list(rule.get("models", []) or []):
                    if model_name in seen:
                        continue
                    seen.add(model_name)
                    ordered.append(model_name)
            return ordered

        if stage_name == "full_image":
            full_image_models = entry.get("full_image_models", "ALL")
            if isinstance(full_image_models, list) and len(full_image_models) == 0:
                return []
            resolved = self._resolve_full_image_model_names(full_image_models, available_models)
            return resolved

        if stage_name == "audio":
            audio_models = entry.get("audio_models", None)
            if audio_models is None:
                return None
            resolved = self._resolve_audio_model_names(audio_models, available_models)
            return resolved

        if stage_name == "asset":
            # A missing key means every whole-asset model, never "no
            # constraint": entries written before the asset stage existed must
            # not wire frame models into it.
            return self._resolve_asset_model_names(entry.get("asset_models", "ALL"), available_models)

        return None

    def get_region_model_rules(self, pipeline_name: str, available_models: Optional[Sequence[Any]] = None) -> List[Dict[str, Any]]:
        entry = self.get_pipeline_entry(pipeline_name)
        if not entry:
            return []

        region_model_map = entry.get("region_models", {}) or {}
        if isinstance(region_model_map, list):
            return self._resolve_region_rule_list(pipeline_name, region_model_map, available_models)

        if not isinstance(region_model_map, dict):
            raise ValueError(
                f"model_capabilities pipeline '{pipeline_name}' region_models must be a mapping or list of rules"
            )

        rules = []
        for raw_key, raw_value in region_model_map.items():
            detector_model_names = self._resolve_reference_names(raw_key, available_models)
            if not detector_model_names:
                continue

            if _looks_like_model_selector(raw_value) or _is_all_config(raw_value) or not isinstance(raw_value, dict):
                model_names = self._resolve_reference_names(raw_value, available_models)
                if not model_names:
                    continue
                for detector_model_name in detector_model_names:
                    rules.append({
                        "key": detector_model_name,
                        "models": model_names,
                        "labels": [],
                    })
                continue

            for raw_label, raw_models in raw_value.items():
                label = str(raw_label).strip()
                if not label:
                    raise ValueError(
                        f"model_capabilities pipeline '{pipeline_name}' region_models contains empty label key"
                    )
                model_names = self._resolve_reference_names(raw_models, available_models)
                if not model_names:
                    continue
                for detector_model_name in detector_model_names:
                    rules.append({
                        "key": detector_model_name,
                        "models": model_names,
                        "labels": [label],
                    })

        return rules

    def _resolve_region_rule_list(
        self,
        pipeline_name: str,
        raw_rules: Sequence[Any],
        available_models: Optional[Sequence[Any]],
    ) -> List[Dict[str, Any]]:
        rules = []
        for index, raw_rule in enumerate(raw_rules):
            if not isinstance(raw_rule, dict):
                raise ValueError(
                    f"model_capabilities pipeline '{pipeline_name}' region_models[{index}] must be a mapping"
                )

            detector_ref = _first_present(raw_rule, "detector", "detectors", "detector_models", "key")
            if detector_ref is None:
                raise ValueError(
                    f"model_capabilities pipeline '{pipeline_name}' region_models[{index}] must declare detector"
                )

            model_ref = _first_present(raw_rule, "models", "region_models", "model")
            if model_ref is None:
                raise ValueError(
                    f"model_capabilities pipeline '{pipeline_name}' region_models[{index}] must declare models"
                )

            detector_model_names = self._resolve_reference_names(detector_ref, available_models)
            model_names = self._resolve_reference_names(model_ref, available_models)
            labels = _normalize_string_list(_first_present(raw_rule, "labels", "label", "classes", "class_names"))
            if not detector_model_names or not model_names:
                continue

            for detector_model_name in detector_model_names:
                rules.append({
                    "key": detector_model_name,
                    "models": model_names,
                    "labels": labels,
                })

        return rules

    def _resolve_full_image_model_names(self, raw_value: Any, available_models: Optional[Sequence[Any]]) -> List[str]:
        if _is_all_token(raw_value):
            return self._resolve_all_full_image_model_names(available_models, exclude_names=[])

        if _is_all_config(raw_value):
            exclude_names = _normalize_string_list(raw_value.get("exclude", []))
            missing_excludes = [name for name in exclude_names if name not in self.active_model_library]
            if missing_excludes:
                raise ValueError(
                    f"model_capabilities full_image_models.exclude references model(s) not in active_ai_models: {missing_excludes}"
                )
            return self._resolve_all_full_image_model_names(available_models, exclude_names=exclude_names)

        return self._resolve_reference_names(raw_value, available_models)

    _FULL_IMAGE_CAPABILITIES = {"tagging", "embedding"}

    _AUDIO_CAPABILITIES = {"embedding", "classification"}

    def _resolve_asset_model_names(self, raw_value: Any, available_models: Optional[Sequence[Any]]) -> List[str]:
        if _is_all_token(raw_value):
            return self._resolve_all_asset_model_names(available_models, exclude_names=[])
        if _is_all_config(raw_value):
            exclude_names = _normalize_string_list(raw_value.get("exclude", []))
            return self._resolve_all_asset_model_names(available_models, exclude_names=exclude_names)
        return self._resolve_reference_names(raw_value, available_models)

    def _resolve_all_asset_model_names(
        self, available_models: Optional[Sequence[Any]], exclude_names: Iterable[str]
    ) -> List[str]:
        if not available_models:
            return []

        excluded = set(exclude_names)
        resolved = []
        for model in available_models:
            model_name = _get_model_config_name(model)
            if not model_name or model_name in excluded:
                continue
            if is_whole_asset_model(model):
                resolved.append(model_name)
        return resolved

    def _resolve_audio_model_names(self, raw_value: Any, available_models: Optional[Sequence[Any]]) -> List[str]:
        if _is_all_token(raw_value):
            return self._resolve_all_audio_model_names(available_models)
        if _is_all_config(raw_value):
            return self._resolve_all_audio_model_names(available_models)
        return self._resolve_reference_names(raw_value, available_models)

    def _resolve_reference_names(self, raw_value: Any, available_models: Optional[Sequence[Any]]) -> List[str]:
        if raw_value is None:
            return []
        if isinstance(raw_value, str):
            text = raw_value.strip()
            return [text] if text else []
        if isinstance(raw_value, dict):
            return self._resolve_selector_names(raw_value, available_models)

        resolved = []
        seen = set()
        for item in raw_value:
            for name in self._resolve_reference_names(item, available_models):
                if name in seen:
                    continue
                seen.add(name)
                resolved.append(name)
        return resolved

    def _resolve_selector_names(self, selector: Dict[str, Any], available_models: Optional[Sequence[Any]]) -> List[str]:
        explicit_names = _selector_values(selector, "name", "names", "model", "models", "config_name", "config_names")
        exclude_names = set(_selector_values(selector, "exclude"))
        if available_models is None:
            return [name for name in explicit_names if name not in exclude_names]

        resolved = []
        seen = set()
        for model in available_models:
            model_name = _get_model_config_name(model)
            if not model_name or model_name in exclude_names or model_name not in self.active_model_library:
                continue
            if not _model_matches_selector(model, selector):
                continue
            if model_name in seen:
                continue
            seen.add(model_name)
            resolved.append(model_name)
        return resolved

    def _resolve_all_audio_model_names(self, available_models: Optional[Sequence[Any]]) -> List[str]:
        if not available_models:
            return []

        resolved = []
        for model in available_models:
            model_name = _get_model_config_name(model)
            if not model_name:
                continue

            model_type = str(getattr(_get_inner_model(model), "model_type", "") or "").lower()
            capabilities = set(_get_model_values(model, "model_capabilities"))

            if "audio" in model_type or (capabilities & self._AUDIO_CAPABILITIES and "audio" in model_type):
                resolved.append(model_name)
                continue

            config_type = str(getattr(_get_inner_model(model), "_config_type", "") or getattr(model, "_config_type", "")).lower()
            if config_type.startswith("audio"):
                resolved.append(model_name)

        return _dedupe_strings(resolved)

    def _resolve_all_full_image_model_names(self, available_models: Optional[Sequence[Any]], exclude_names: List[str]) -> List[str]:
        exclude_set = set(exclude_names)
        if not available_models:
            return [name for name in sorted(self.active_model_library) if name not in exclude_set]

        resolved = []
        for model in available_models:
            model_name = _get_model_config_name(model)
            if not model_name or model_name in exclude_set:
                continue

            explicit = getattr(_get_inner_model(model), "full_image_model", None)
            if explicit is not None:
                if explicit:
                    resolved.append(model_name)
                continue

            capabilities = set(_get_model_values(model, "model_capabilities"))
            if capabilities & self._FULL_IMAGE_CAPABILITIES:
                resolved.append(model_name)
        return _dedupe_strings(resolved)

    def _validate_region_models(self, pipeline_name: str, raw_value: Any):
        if isinstance(raw_value, list):
            for index, rule in enumerate(raw_value):
                if not isinstance(rule, dict):
                    raise ValueError(
                        f"model_capabilities pipeline '{pipeline_name}' region_models[{index}] must be a mapping"
                    )
                if _first_present(rule, "detector", "detectors", "detector_models", "key") is None:
                    raise ValueError(
                        f"model_capabilities pipeline '{pipeline_name}' region_models[{index}] must declare detector"
                    )
                if _first_present(rule, "models", "region_models", "model") is None:
                    raise ValueError(
                        f"model_capabilities pipeline '{pipeline_name}' region_models[{index}] must declare models"
                    )
                self._validate_reference_value(_first_present(rule, "detector", "detectors", "detector_models", "key"), allow_all=False)
                self._validate_reference_value(_first_present(rule, "models", "region_models", "model"), allow_all=False)
                _ = _normalize_string_list(_first_present(rule, "labels", "label", "classes", "class_names"))
            return

        if not isinstance(raw_value, dict):
            raise ValueError(
                f"model_capabilities pipeline '{pipeline_name}' region_models must be a mapping or list of rules"
            )

        detector_models = set(_normalize_string_list(self.config.get(pipeline_name, {}).get("detector_models", [])))
        for detector_model_name, region_value in raw_value.items():
            detector_model_name = str(detector_model_name).strip()
            if not detector_model_name:
                raise ValueError(
                    f"model_capabilities pipeline '{pipeline_name}' region_models contains an empty key"
                )

            if detector_models and detector_model_name not in detector_models and not _contains_selector(self.config.get(pipeline_name, {})):
                raise ValueError(
                    f"model_capabilities pipeline '{pipeline_name}' region_models key '{detector_model_name}' must also be listed in detector_models"
                )

            if _looks_like_model_selector(region_value) or _is_all_config(region_value) or not isinstance(region_value, dict):
                self._validate_reference_value(region_value, allow_all=False)
                continue

            for label_name, label_model_names in region_value.items():
                if not str(label_name).strip():
                    raise ValueError(
                        f"model_capabilities pipeline '{pipeline_name}' region_models['{detector_model_name}'] contains empty label key"
                    )
                self._validate_reference_value(label_model_names, allow_all=False)

    def _validate_reference_value(self, raw_value: Any, allow_all: bool):
        if raw_value is None:
            return
        if _is_all_token(raw_value) or _is_all_config(raw_value):
            if allow_all:
                return
            raise ValueError("model_capabilities selector value does not allow ALL")
        if isinstance(raw_value, str):
            if raw_value not in self.active_model_library:
                raise ValueError(
                    f"model_capabilities references model '{raw_value}' that is not in active_ai_models"
                )
            return
        if isinstance(raw_value, dict):
            explicit_names = _selector_values(raw_value, "name", "names", "model", "models", "config_name", "config_names")
            missing = [name for name in explicit_names if name not in self.active_model_library]
            if missing:
                raise ValueError(
                    f"model_capabilities selector references model(s) not in active_ai_models: {missing}"
                )
            return
        for item in raw_value:
            self._validate_reference_value(item, allow_all=allow_all)


def _looks_like_model_selector(value: Any) -> bool:
    return isinstance(value, dict) and bool(set(value.keys()) & _SELECTOR_KEYS)


def _contains_selector(value: Any) -> bool:
    if _looks_like_model_selector(value):
        return True
    if isinstance(value, dict):
        return any(_contains_selector(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_selector(item) for item in value)
    return False


def _is_all_token(value: Any) -> bool:
    return isinstance(value, str) and value.strip().upper() == "ALL"


def _is_all_config(value: Any) -> bool:
    return isinstance(value, dict) and str(value.get("mode", "")).strip().upper() == "ALL"


def _first_present(mapping: Dict[str, Any], *keys: str):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _selector_values(selector: Dict[str, Any], *keys: str) -> List[str]:
    values = []
    for key in keys:
        if key in selector:
            values.extend(_normalize_string_list(selector.get(key)))
    return _dedupe_strings(values)


def _model_matches_selector(model: Any, selector: Dict[str, Any]) -> bool:
    model_name = _get_model_config_name(model)
    candidate_names = {model_name, str(getattr(_get_inner_model(model), "model_file_name", "") or "").strip()}
    candidate_names.discard("")

    explicit_names = set(_selector_values(selector, "name", "names", "model", "models", "config_name", "config_names"))
    if explicit_names and candidate_names.isdisjoint(explicit_names):
        return False

    capabilities = _selector_values(selector, "capability", "capabilities")
    if capabilities and not _has_any(capabilities, _get_model_values(model, "model_capabilities")):
        return False

    categories = _selector_values(selector, "category", "categories", "model_category")
    if categories and not _has_any(categories, _get_model_values(model, "model_category")):
        return False

    scopes = _selector_values(selector, "scope", "scopes", "target_scope", "target_scopes", "supported_target_scopes")
    if scopes and not _has_any(scopes, _get_model_values(model, "supported_target_scopes")):
        return False

    model_types = _selector_values(selector, "model_type", "model_types", "type", "types")
    if model_types:
        actual_types = [str(getattr(_get_inner_model(model), "model_type", "") or "")]
        actual_types.append(str(getattr(_get_inner_model(model), "_config_type", "") or getattr(model, "_config_type", "")))
        if not _has_any(model_types, actual_types):
            return False

    if "full_image_model" in selector:
        expected = bool(selector.get("full_image_model"))
        if bool(getattr(_get_inner_model(model), "full_image_model", False)) != expected:
            return False

    return True


def _get_inner_model(model: Any) -> Any:
    return getattr(model, "model", model)


def _get_model_config_name(model: Any) -> str:
    inner = _get_inner_model(model)
    return str(
        getattr(model, "config_name", None)
        or getattr(inner, "config_name", None)
        or getattr(inner, "model_file_name", "")
        or ""
    ).strip()


def _get_model_values(model: Any, attribute_name: str) -> List[str]:
    return _normalize_string_list(getattr(_get_inner_model(model), attribute_name, None))


def _has_any(required: Iterable[str], actual: Iterable[str]) -> bool:
    actual_set = {str(value).strip().lower() for value in actual if str(value).strip()}
    return any(str(value).strip().lower() in actual_set for value in required if str(value).strip())


def _normalize_string_list(raw_value) -> List[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        text = raw_value.strip()
        return [text] if text else []
    if isinstance(raw_value, dict):
        return []
    normalized = []
    for item in raw_value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _dedupe_strings(values: Iterable[str]) -> List[str]:
    seen = set()
    deduped = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped
