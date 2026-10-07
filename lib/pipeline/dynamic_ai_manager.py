import logging
import os
import re
from typing import Iterable, List, Optional
from lib.config.config_utils import load_config
from lib.config.model_capabilities import ModelCapabilitiesConfig
from lib.configurator.configure_model_capabilities import load_model_capabilities_config
from lib.model.ai_model import AIModel
from lib.model.whole_asset import is_whole_asset_model
from lib.pipeline.pipeline import ModelWrapper
from lib.pipeline.preprocess_spec import PreprocessSpec
from lib.migrations.migration_v20 import migrate_to_2_0
from lib.server.exceptions import NoActiveModelsException
from lib.model.video_preprocessor import VideoPreprocessorModel


ai_active_directory = "./config/active_ai.yaml"



class DynamicAIManager:
    def __init__(self, model_manager):
        self.logger = logging.getLogger("logger")
        self.model_manager = model_manager
        self.configfile = load_config(ai_active_directory)
        self.ai_model_names = self.configfile.get("active_ai_models", [])
        self.active_capabilities = self.configfile.get("active_capabilities", None)
        self.capability_model_groups = self.configfile.get("capability_model_groups", None)
        model_capabilities_config = load_model_capabilities_config()
        self.model_capabilities = ModelCapabilitiesConfig(
            config=model_capabilities_config,
            active_model_library=self.ai_model_names,
            logger=self.logger,
        )
        self.loaded = False
        self.image_size = None
        self.normalization_config = None
        self.models = []
        self.models_by_config_name = {}

    def reload_active_config(self):
        self.configfile = load_config(ai_active_directory)
        self.ai_model_names = self.configfile.get("active_ai_models", [])
        self.active_capabilities = self.configfile.get("active_capabilities", None)
        self.capability_model_groups = self.configfile.get("capability_model_groups", None)
        self.model_capabilities = ModelCapabilitiesConfig(
            config=load_model_capabilities_config(),
            active_model_library=self.ai_model_names,
            logger=self.logger,
        )
        self.loaded = False
        self.models = []
        self.models_by_config_name = {}

    def set_known_pipelines(self, pipeline_names):
        self.model_capabilities.set_known_pipelines(pipeline_names)
        self.model_capabilities.prune_unavailable_models()
        self.model_capabilities.validate()

    def load(self):
        if self.loaded:
            return
        models = []
        self.models_by_config_name = {}
        self.ai_model_names = self._resolve_active_model_names()

        if self.ai_model_names is None or len(self.ai_model_names) == 0:
            self.logger.error("Error: No active AI models found in active_ai.yaml")
            raise NoActiveModelsException("Error: No active AI models found in active_ai.yaml")
        for model_name in self.ai_model_names:
            loaded_model = self.model_manager.get_or_create_model(model_name)
            if loaded_model is not None:
                loaded_model.config_name = model_name
                child_model = getattr(loaded_model, "model", None)
                if child_model is not None:
                    child_model.config_name = model_name
            self.models_by_config_name[model_name] = loaded_model
            models.append(loaded_model)
        models = [model for model in models if model is not None]
        if len(models) == 0:
            self.logger.error("Error: Active model names resolved, but none could be loaded successfully.")
            raise NoActiveModelsException("Error: Could not load any resolved active AI models")
        self.__verify_models(models)
        self.models = models
        self.loaded = True

    def get_dynamic_models(self, mode, inputs, outputs, required_capabilities=None, required_scope=None, pipeline_name=None, model_config=None):
        self.load()
        selected_models = self._select_models(
            required_capabilities=required_capabilities,
            required_scope=required_scope,
            mode=mode,
            pipeline_name=pipeline_name,
            model_config=model_config,
        )
        normalized_mode_early = (mode or "").lower()
        if len(selected_models) == 0 and normalized_mode_early == "asset":
            # An asset-scope stage is optional by construction: most deployments
            # install no asset model at all. Returning an empty-but-complete
            # stage keeps the rest of the pipeline working instead of failing
            # the whole pipeline load.
            return self._create_dynamic_asset_wrappers(inputs, outputs, [])
        if len(selected_models) == 0:
            # Face-only (or detector-only) pipelines have no full-image
            # tagging models but DO have detector + region models.  Allow
            # empty selection when region branches are configured.
            has_region_branches = False
            # A video pipeline whose only active models analyse whole videos
            # (e.g. a shot-boundary-only deployment) has nothing to run per
            # frame, which is a valid configuration rather than an error.
            has_asset_models = normalized_mode_early == "video" and any(
                is_whole_asset_model(model) for model in self.models
            )
            if pipeline_name:
                _det_names = self._frame_detector_names(pipeline_name)
                _region_rules = self.model_capabilities.get_region_model_rules(pipeline_name, available_models=self.models)
                has_region_branches = bool(_det_names) and bool(_region_rules)
            if not has_region_branches and not has_asset_models:
                requested_caps = _normalize_string_list(required_capabilities)
                requested_scope = required_scope or "any"
                raise ValueError(
                    f"Error: No active AI models matched dynamic expansion filters "
                    f"(capabilities={requested_caps or ['any']}, scope={requested_scope})"
                )

        normalized_mode = (mode or "").lower()
        if normalized_mode == "video":
            should_expand_region_branch = not bool(_normalize_string_list(required_capabilities)) and required_scope is None
            return self._create_dynamic_video_wrappers(
                inputs,
                outputs,
                selected_models,
                pipeline_name=pipeline_name,
                enable_region_branch=should_expand_region_branch,
            )
        if normalized_mode == "image":
            should_expand_region_branch = not bool(_normalize_string_list(required_capabilities)) and required_scope is None
            return self._create_dynamic_image_wrappers(
                inputs,
                outputs,
                selected_models,
                pipeline_name=pipeline_name,
                enable_region_branch=should_expand_region_branch,
            )
        if normalized_mode == "region":
            return self._create_dynamic_region_wrappers(inputs, outputs, selected_models)
        if normalized_mode == "audio":
            return self._create_dynamic_audio_wrappers(inputs, outputs, selected_models, pipeline_name=pipeline_name)
        if normalized_mode == "asset":
            return self._create_dynamic_asset_wrappers(inputs, outputs, selected_models, pipeline_name=pipeline_name)

        raise ValueError(f"Error: Unsupported dynamic mode '{mode}'. Expected 'image', 'video', 'region', 'audio', or 'asset'.")

    def get_dynamic_models_from_config(self, model_config, pipeline_name=None):
        mode = model_config.get("dynamic_mode", None)
        if mode is None:
            raise ValueError("Error: dynamic_ai model config requires dynamic_mode set to 'image', 'video', or 'region'.")

        return self.get_dynamic_models(
            mode=mode,
            inputs=model_config["inputs"],
            outputs=model_config["outputs"],
            required_capabilities=model_config.get("required_capabilities", model_config.get("capabilities", None)),
            required_scope=model_config.get("required_scope", model_config.get("target_scope", None)),
            pipeline_name=pipeline_name,
            model_config=model_config,
        )

    def get_dynamic_video_ai_models(self, inputs, outputs, pipeline_name=None):
        return self.get_dynamic_models(mode="video", inputs=inputs, outputs=outputs, pipeline_name=pipeline_name)

    def get_dynamic_audio_ai_models(self, inputs, outputs, pipeline_name=None):
        return self.get_dynamic_models(mode="audio", inputs=inputs, outputs=outputs, pipeline_name=pipeline_name)

    def get_dynamic_asset_ai_models(self, inputs, outputs, pipeline_name=None):
        return self.get_dynamic_models(
            mode="asset",
            inputs=inputs,
            outputs=outputs,
            required_scope="asset",
            pipeline_name=pipeline_name,
        )

    def get_dynamic_region_ai_models(self, inputs, outputs, required_capabilities=None, pipeline_name=None):
        return self.get_dynamic_models(
            mode="region",
            inputs=inputs,
            outputs=outputs,
            required_capabilities=required_capabilities,
            required_scope="region",
            pipeline_name=pipeline_name,
        )

    def _create_dynamic_video_wrappers(self, inputs, outputs, models, pipeline_name=None, enable_region_branch=False):
        model_wrappers = []

        # ── Collect preprocessing specs from all models ──
        all_specs, model_to_spec, detector_names, detector_models, detector_to_spec, region_source_spec, region_model_rules = (
            self._collect_pipeline_specs(models, pipeline_name, enable_region_branch)
        )

        # ── Create per-pipeline preprocessor ──
        if pipeline_name:
            _preproc_alias = f"video_preprocessor_dynamic__{pipeline_name}"
            video_preprocessor = self.model_manager.get_or_create_model_alias("video_preprocessor_dynamic", _preproc_alias)
        else:
            video_preprocessor = self.model_manager.get_or_create_model("video_preprocessor_dynamic")
        video_preprocessor.model.specs = all_specs
        # Every active model that does not analyse the whole video itself feeds
        # off this preprocessor (full-image, detector and region alike), so
        # deriving the set from the model type rather than from the wiring
        # cannot under-count and silently skip a decode that was needed.
        video_preprocessor.model.downstream_model_names = _downstream_model_names(
            [m for m in self.models if not is_whole_asset_model(m)]
        )
        # Which models read which spec, so a request that names its models is
        # only given the specs those models read. The detector's input and the
        # region source belong to the detector and to every model run on its
        # regions.
        spec_consumers = {}
        for model in models:
            spec_consumers.setdefault(model_to_spec[id(model)].key, set()).update(_downstream_model_names([model]))
        if region_source_spec is not None:
            region_models = self._select_models_by_name(_dedupe_strings([
                name for rule in region_model_rules for name in (rule.get("models", []) or [])
            ]))
            region_names = _downstream_model_names(detector_models, region_models)
            for spec in [region_source_spec] + list(detector_to_spec.values()):
                spec_consumers.setdefault(spec.key, set()).update(region_names)
        video_preprocessor.model.spec_consumers = spec_consumers

        # Build output names: fixed outputs + one per spec.
        fixed_outputs = [
            "dynamic_children",
            "frame_index",
            "dynamic_threshold",
            "dynamic_return_confidence",
            "dynamic_skipped_categories",
            "dynamic_requested_model_names",
        ]
        spec_keys = [s.key for s in all_specs]
        model_wrappers.append(ModelWrapper(video_preprocessor, inputs, fixed_outputs + spec_keys))

        # ── Wire tagging models ──
        for model in models:
            spec = model_to_spec[id(model)]
            model_wrappers.append(ModelWrapper(
                model,
                [spec.key, "dynamic_threshold", "dynamic_return_confidence", "dynamic_skipped_categories"],
                model.model.model_category,
            ))

        # ── Wire detectors and region branches ──
        additional_coalesce_inputs = self._build_region_branches(
            model_wrappers, detector_names, detector_models, detector_to_spec,
            region_source_spec, region_model_rules, pipeline_name,
            threshold_key="dynamic_threshold", return_confidence_key="dynamic_return_confidence",
            skipped_categories_key="dynamic_skipped_categories",
            region_targets_extra_inputs_fn=lambda det_key, rs_key: ["frame_index", rs_key, "frame_index"] + ([det_key] if det_key != rs_key else []),
        )

        # ── Coalesce ──
        coalesce_inputs = self._build_coalesce_inputs(models)
        coalesce_inputs.extend(additional_coalesce_inputs)
        coalesce_inputs = _dedupe_strings(coalesce_inputs)
        coalesce_inputs.insert(0, "frame_index")
        model_wrappers.append(ModelWrapper(self.model_manager.get_or_create_model("result_coalescer"), coalesce_inputs, ["dynamic_result"]))
        model_wrappers.append(ModelWrapper(self.model_manager.get_or_create_model("result_finisher"), ["dynamic_result"], []))
        model_wrappers.append(ModelWrapper(self.model_manager.get_or_create_model("batch_awaiter"), ["dynamic_children"], outputs))

        self.logger.debug("Finished creating dynamic Video AI models")
        return model_wrappers
    
    def get_dynamic_image_ai_models(self, inputs, outputs, pipeline_name=None):
        return self.get_dynamic_models(mode="image", inputs=inputs, outputs=outputs, pipeline_name=pipeline_name)

    def _create_dynamic_image_wrappers(self, inputs, outputs, models, pipeline_name=None, enable_region_branch=False):
        model_wrappers = []

        # ── Collect preprocessing specs from all models ──
        all_specs, model_to_spec, detector_names, detector_models, detector_to_spec, region_source_spec, region_model_rules = (
            self._collect_pipeline_specs(models, pipeline_name, enable_region_branch)
        )

        # ── Create per-pipeline preprocessor ──
        if pipeline_name:
            _preproc_alias = f"image_preprocessor_dynamic__{pipeline_name}"
            image_preprocessor = self.model_manager.get_or_create_model_alias("image_preprocessor_dynamic", _preproc_alias)
        else:
            image_preprocessor = self.model_manager.get_or_create_model("image_preprocessor_dynamic")
        image_preprocessor.model.specs = all_specs

        spec_keys = [s.key for s in all_specs]
        model_wrappers.append(ModelWrapper(image_preprocessor, [inputs[0]], spec_keys))

        # ── Wire tagging models ──
        for model in models:
            spec = model_to_spec[id(model)]
            model_wrappers.append(ModelWrapper(
                model,
                [spec.key, inputs[1], inputs[2], inputs[3]],
                model.model.model_category,
            ))

        # ── Wire detectors and region branches ──
        additional_coalesce_inputs = self._build_region_branches(
            model_wrappers, detector_names, detector_models, detector_to_spec,
            region_source_spec, region_model_rules, pipeline_name,
            threshold_key=inputs[1], return_confidence_key=inputs[2],
            skipped_categories_key=inputs[3],
            region_targets_extra_inputs_fn=lambda det_key, rs_key: [inputs[0], rs_key] + ([det_key] if det_key != rs_key else []),
        )

        # ── Coalesce ──
        coalesce_inputs = self._build_coalesce_inputs(models)
        coalesce_inputs.extend(additional_coalesce_inputs)
        coalesce_inputs = _dedupe_strings(coalesce_inputs)
        model_wrappers.append(ModelWrapper(self.model_manager.get_or_create_model("result_coalescer"), coalesce_inputs, outputs=outputs))

        self.logger.debug("Finished creating dynamic Image AI models")
        return model_wrappers

    def _create_dynamic_region_wrappers(self, inputs, outputs, models):
        model_wrappers = []

        # Build region children futures from a parent image/frame tensor and region targets.
        # inputs expected: [source_tensor, region_targets, threshold, return_confidence, skipped_categories]
        model_wrappers.append(
            ModelWrapper(
                self.model_manager.get_or_create_model("region_children_builder"),
                inputs,
                [
                    "dynamic_region_children",
                    "dynamic_region_image",
                    "dynamic_region_target",
                    "dynamic_threshold",
                    "dynamic_return_confidence",
                    "dynamic_skipped_categories",
                    "dynamic_region_source",
                    "dynamic_detection_index",
                ],
            )
        )

        # Run selected models on each region child image.
        for model in models:
            model_wrappers.append(
                ModelWrapper(
                    model,
                    ["dynamic_region_image", "dynamic_threshold", "dynamic_return_confidence", "dynamic_skipped_categories"],
                    model.model.model_category,
                )
            )

        # Coalesce per-region results: detection_index for correlation + model outputs.
        # The full region target stays in the child future data for models that
        # need it (e.g. face alignment reads kps), but is not included in the
        # coalesced output.
        coalesce_inputs = self._build_coalesce_inputs(models)
        coalesce_inputs.insert(0, "dynamic_detection_index")
        model_wrappers.append(
            ModelWrapper(
                self.model_manager.get_or_create_model("result_coalescer"),
                coalesce_inputs,
                ["dynamic_region_result"],
            )
        )

        model_wrappers.append(
            ModelWrapper(
                self.model_manager.get_or_create_model("result_finisher"),
                ["dynamic_region_result"],
                [],
            )
        )

        # Await all region children to return array of per-region outputs.
        model_wrappers.append(
            ModelWrapper(
                self.model_manager.get_or_create_model("batch_awaiter"),
                ["dynamic_region_children"],
                outputs,
            )
        )

        self.logger.debug("Finished creating dynamic Region AI models")
        return model_wrappers

    def _create_dynamic_audio_wrappers(self, inputs, outputs, models, pipeline_name=None):
        """Wire audio preprocessor → per-window AI models → coalescer → finisher → batch_awaiter.

        Follows the video pipeline one-to-many pattern:
        - The audio preprocessor extracts audio, separates vocals, detects speech
          regions, and spawns one child ItemFuture per overlapping window.
        - Each child flows independently through the selected audio models.
        - result_coalescer collects per-window model outputs.
        - result_finisher marks the child future as done.
        - batch_awaiter waits for all children and writes the aggregated list
          to the parent future (outputs[0] = "childrenResults").
        """
        from lib.pipeline.audio_preprocess_spec import AudioPreprocessSpec as APS

        model_wrappers = []

        # ── Collect audio preprocessing specs from all models ──
        spec_set = set()
        model_to_spec = {}
        for model in models:
            spec = APS.for_model(model.model)
            model_to_spec[id(model)] = spec
            spec_set.add(spec)
        all_specs = sorted(spec_set, key=lambda s: s.sample_rate, reverse=True)

        # ── Create per-pipeline audio preprocessor ──
        if pipeline_name:
            _preproc_alias = f"audio_preprocessor_dynamic__{pipeline_name}"
            audio_preprocessor = self.model_manager.get_or_create_model_alias(
                "audio_preprocessor_dynamic", _preproc_alias
            )
        else:
            audio_preprocessor = self.model_manager.get_or_create_model("audio_preprocessor_dynamic")
        audio_preprocessor.model.specs = all_specs

        fixed_outputs = [
            "dynamic_children",
            "window_index",
            "window_start",
            "window_end",
            "dynamic_requested_model_names",
        ]
        spec_keys = [s.key for s in all_specs]
        model_wrappers.append(ModelWrapper(audio_preprocessor, inputs, fixed_outputs + spec_keys))

        # ── Wire per-window AI models ──
        for model in models:
            spec = model_to_spec[id(model)]
            model_wrappers.append(ModelWrapper(
                model,
                [spec.key],
                model.model.model_category,
            ))

        # ── Coalesce per-window results ──
        coalesce_inputs = self._build_coalesce_inputs(models)
        coalesce_inputs = _dedupe_strings(coalesce_inputs)
        coalesce_inputs.insert(0, "window_index")
        coalesce_inputs.insert(1, "window_start")
        coalesce_inputs.insert(2, "window_end")
        model_wrappers.append(ModelWrapper(
            self.model_manager.get_or_create_model("result_coalescer"),
            coalesce_inputs,
            ["dynamic_result"],
        ))

        # ── Mark child future as done ──
        model_wrappers.append(ModelWrapper(
            self.model_manager.get_or_create_model("result_finisher"),
            ["dynamic_result"],
            [],
        ))

        # ── Await all children → aggregate into parent ──
        model_wrappers.append(ModelWrapper(
            self.model_manager.get_or_create_model("batch_awaiter"),
            ["dynamic_children"],
            outputs,
        ))

        self.logger.debug("Finished creating dynamic Audio AI models")
        return model_wrappers

    def _create_dynamic_asset_wrappers(self, inputs, outputs, models, pipeline_name=None):
        """Wire asset-scope models directly onto the parent future.

        Asset-scope models consume the asset itself (a video path) rather than
        preprocessed frames, and return a single result for the whole asset, so
        there is no preprocessor and no child fan-out here.  Each model reads
        the stage's declared inputs and writes its own categories; a collector
        gathers those into the stage output.

        ``models`` may legitimately be empty: the collector is still wired (on
        the asset input alone) so the stage completes and the pipeline's
        downstream join is satisfied.
        """
        model_wrappers = []
        for model in models:
            model_wrappers.append(ModelWrapper(model, list(inputs), model.model.model_category))

        # A model that needs every frame gets them from the pipeline's shared
        # video preprocessor, which has to know to decode in full for it.
        if pipeline_name:
            video_preprocessor = self.model_manager.get_or_create_model_alias(
                "video_preprocessor_dynamic", f"video_preprocessor_dynamic__{pipeline_name}")
        else:
            video_preprocessor = self.model_manager.get_or_create_model("video_preprocessor_dynamic")
        video_preprocessor.model.dense_consumers = [
            m.model for m in models if getattr(m.model, "requires_every_frame", False)
        ]

        collector_inputs = [inputs[0]] + _dedupe_strings(self._build_coalesce_inputs(models))
        model_wrappers.append(ModelWrapper(
            self.model_manager.get_or_create_model("asset_result_collector"),
            collector_inputs,
            outputs,
        ))

        self.logger.debug(f"Finished creating dynamic Asset AI models ({len(models)} model(s))")
        return model_wrappers

    def _select_models(self, required_capabilities=None, required_scope=None, mode=None, pipeline_name=None, model_config=None):
        normalized_capabilities = set(_normalize_string_list(required_capabilities) or [])
        normalized_scope = str(required_scope).strip() if required_scope is not None else None
        candidate_models = self.models

        dynamic_stage = self._infer_dynamic_stage(mode, normalized_capabilities)
        if pipeline_name:
            configured_model_names = self.model_capabilities.resolve_model_names_for_stage(
                pipeline_name=pipeline_name,
                stage=dynamic_stage,
                available_models=self.models,
            )
            if configured_model_names is not None:
                candidate_models = self._select_models_by_name(configured_model_names)

        selected = []

        for model in candidate_models:
            model_capabilities = set(getattr(model.model, "model_capabilities", []) or [])
            model_scopes = set(getattr(model.model, "supported_target_scopes", []) or [])

            if normalized_capabilities and not (normalized_capabilities & model_capabilities):
                continue
            if normalized_scope is not None and normalized_scope not in model_scopes:
                continue
            # A whole-asset model decodes the video itself: it belongs to the
            # asset stage and to no other, and nothing else belongs there. This
            # holds whatever model_capabilities.yaml says, including for
            # pipelines that declare no stage constraints (the v3 pipelines).
            if (dynamic_stage == "asset") != is_whole_asset_model(model):
                continue
            selected.append(model)

        return selected

    def _frame_detector_names(self, pipeline_name):
        """Detectors configured for the pipeline, never a whole-asset model:
        a detector runs on preprocessed frames."""
        names = self.model_capabilities.resolve_model_names_for_stage(
            pipeline_name=pipeline_name, stage="detector", available_models=self.models,
        ) or []
        return [name for name in names if not is_whole_asset_model(self.models_by_config_name.get(name))]

    def _infer_dynamic_stage(self, mode, normalized_capabilities):
        if str(mode or "").strip().lower() == "region":
            return "region"
        if str(mode or "").strip().lower() == "audio":
            return "audio"
        if str(mode or "").strip().lower() == "asset":
            return "asset"
        if "detection" in normalized_capabilities:
            return "detector"
        return "full_image"

    def _select_models_by_name(self, model_names):
        ordered_names = _normalize_string_list(model_names) or []
        if not ordered_names:
            return []

        selected = []
        missing = []
        for model_name in ordered_names:
            model = self.models_by_config_name.get(model_name, None)
            if model is None:
                missing.append(model_name)
                continue
            selected.append(model)

        if missing:
            raise ValueError(
                f"Error: model_capabilities selected model(s) that are not currently loadable from active_ai_models: {missing}"
            )

        return selected

    def _build_coalesce_inputs(self, models):
        coalesce_inputs = []
        for model in models:
            categories = model.model.model_category
            if isinstance(categories, list):
                coalesce_inputs.extend(categories)
            else:
                coalesce_inputs.append(categories)
        return coalesce_inputs

    # ── Spec collection and region branch helpers ──────────────────────

    def _collect_pipeline_specs(self, tagging_models, pipeline_name, enable_region_branch):
        """Collect and deduplicate PreprocessSpecs for every model in a pipeline.

        Returns:
            (all_specs, model_to_spec, detector_names, detector_models,
             detector_to_spec, region_source_spec, region_model_rules)
        """
        spec_set = set()
        model_to_spec = {}  # id(model_wrapper) → PreprocessSpec

        for model in tagging_models:
            spec = PreprocessSpec.for_model(model.model)
            model_to_spec[id(model)] = spec
            spec_set.add(spec)

        detector_names = []
        detector_models = []
        detector_to_spec = {}  # detector_name → PreprocessSpec
        region_source_spec = None
        region_model_rules = []

        if pipeline_name and enable_region_branch:
            detector_names = self._frame_detector_names(pipeline_name)
            region_model_rules = self.model_capabilities.get_region_model_rules(pipeline_name, available_models=self.models) or []

            if detector_names and region_model_rules:
                detector_models = self._select_models_by_name(detector_names)
                for det_name, det_model in zip(detector_names, detector_models):
                    det_spec = PreprocessSpec.for_detector(det_model.model)
                    detector_to_spec[det_name] = det_spec
                    spec_set.add(det_spec)

                region_source_spec = PreprocessSpec.for_region_source()
                spec_set.add(region_source_spec)

        # Sort: highest effective resolution first.  This means the
        # preprocessor publishes large tensors (region source) before
        # small ones, letting the detector chain start early.

        # Merge specs that differ only in precision (e.g. f16 + f32 for the
        # same geometry).  Keep the wider dtype (float32) so models that need
        # half can cast cheaply at inference time instead of storing two copies.
        from collections import defaultdict
        geom_groups = defaultdict(list)
        for spec in spec_set:
            geom_groups[spec._geometry_key].append(spec)

        merged_spec_set = set()
        spec_remap = {}
        for _geom, group in geom_groups.items():
            dtypes = {s.effective_dtype for s in group}
            if len(dtypes) > 1:
                canonical = group[0].as_float32()
                for s in group:
                    if s != canonical:
                        spec_remap[s] = canonical
                merged_spec_set.add(canonical)
            else:
                merged_spec_set.update(group)

        if spec_remap:
            for mid in model_to_spec:
                old = model_to_spec[mid]
                if old in spec_remap:
                    model_to_spec[mid] = spec_remap[old]
            for dname in detector_to_spec:
                old = detector_to_spec[dname]
                if old in spec_remap:
                    detector_to_spec[dname] = spec_remap[old]

        all_specs = sorted(merged_spec_set, key=lambda s: s.effective_resolution, reverse=True)

        return (all_specs, model_to_spec, detector_names, detector_models,
                detector_to_spec, region_source_spec, region_model_rules)

    def _build_region_branches(self, model_wrappers, detector_names, detector_models,
                               detector_to_spec, region_source_spec, region_model_rules,
                               pipeline_name, *, threshold_key, return_confidence_key,
                               skipped_categories_key, region_targets_extra_inputs_fn):
        """Wire detector → region_targets → region_children → region_models → coalesce.

        Appends ModelWrappers to *model_wrappers* in-place and returns the
        list of additional coalesce input keys.

        ``region_targets_extra_inputs_fn(det_input_key, region_source_key)``
        returns the extra inputs for ``detector_result_to_region_targets``
        (differs between image and video paths).
        """
        additional_coalesce_inputs = []
        if not detector_names or not region_model_rules or region_source_spec is None:
            return additional_coalesce_inputs

        region_source_key = region_source_spec.key
        region_rules_by_detector = {}
        for rule in region_model_rules:
            detector_key = str(rule.get("key", "")).strip()
            if not detector_key:
                continue
            region_rules_by_detector.setdefault(detector_key, []).append(rule)

        for detector_name, detector_model in zip(detector_names, detector_models):
            det_spec = detector_to_spec[detector_name]
            det_input_key = det_spec.key
            detector_outputs = _normalize_string_list(detector_model.model.model_category) or []
            additional_coalesce_inputs.extend(detector_outputs)

            model_wrappers.append(ModelWrapper(
                detector_model,
                [det_input_key, threshold_key, return_confidence_key, skipped_categories_key],
                detector_outputs,
            ))

            detector_rules = region_rules_by_detector.get(detector_name) or []
            if not detector_rules:
                continue
            if not detector_outputs:
                self.logger.warning(
                    f"Skipping region branch for detector '{detector_name}' in pipeline '{pipeline_name}' because it emits no output categories"
                )
                continue

            detector_output_key = detector_outputs[0]
            if len(detector_outputs) > 1:
                self.logger.warning(
                    f"Detector '{detector_name}' in pipeline '{pipeline_name}' emits multiple output keys {detector_outputs}. "
                    f"Using '{detector_output_key}' for region target extraction."
                )

            alias = _sanitize_key(detector_name)
            rt_key = f"region_targets__{alias}"
            re_key = f"region_errors__{alias}"
            ci_key = "detection_index"  # clean name for coalescer

            extra_inputs = region_targets_extra_inputs_fn(det_input_key, region_source_key)
            model_wrappers.append(ModelWrapper(
                self.model_manager.get_or_create_model("detector_result_to_region_targets"),
                [detector_output_key] + extra_inputs,
                [rt_key, re_key],
            ))

            for rule_index, detector_rule in enumerate(detector_rules):
                region_model_names = _dedupe_strings(list(detector_rule.get("models", []) or []))
                if not region_model_names:
                    continue

                labels = _normalize_string_list(detector_rule.get("labels", []) or []) or []
                label_suffix = _format_label_filter_suffix(labels)
                branch_alias = alias if len(detector_rules) == 1 and not label_suffix else f"{alias}__rule_{rule_index}{label_suffix}"
                ch_key = f"dynamic_region_children__{branch_alias}"
                ri_key = f"dynamic_region_image__{branch_alias}"
                rtg_key = f"dynamic_region_target__{branch_alias}"
                th_key = f"dynamic_threshold__{branch_alias}"
                rc_key = f"dynamic_return_confidence__{branch_alias}"
                sc_key = f"dynamic_skipped_categories__{branch_alias}"
                rr_key = f"dynamic_region_result__{branch_alias}"
                rrs_key = f"regions__{branch_alias}"

                model_wrappers.append(ModelWrapper(
                    self.model_manager.get_or_create_model("region_children_builder"),
                    [region_source_key, rt_key, threshold_key, return_confidence_key, skipped_categories_key],
                    [ch_key, ri_key, rtg_key, th_key, rc_key, sc_key, f"dynamic_region_source__{branch_alias}", ci_key],
                ))

                region_models = self._select_models_by_name(region_model_names)
                region_branch_outputs = []
                for region_model in region_models:
                    branch_outputs = _normalize_string_list(region_model.model.model_category) or []
                    region_branch_outputs.extend(branch_outputs)
                    model_wrappers.append(ModelWrapper(
                        region_model, [ri_key, th_key, rc_key, sc_key], branch_outputs,
                    ))

                model_wrappers.append(ModelWrapper(
                    self.model_manager.get_or_create_model("result_coalescer"),
                    [ci_key] + region_branch_outputs, [rr_key],
                ))
                model_wrappers.append(ModelWrapper(
                    self.model_manager.get_or_create_model("result_finisher"), [rr_key], [],
                ))
                model_wrappers.append(ModelWrapper(
                    self.model_manager.get_or_create_model("batch_awaiter"), [ch_key], [rrs_key],
                ))

                additional_coalesce_inputs.append(rrs_key)

        return additional_coalesce_inputs
    
    def __verify_models(self, models, second_pass=False):
        first_image_size = None
        first_norm_config = None
        for model in models:
            inner_model = model.model
            if not isinstance(inner_model, AIModel):
                raise ValueError(f"Error: Dynamic AI models must all be AI models! {inner_model} is not an AI model!")
            try:
                if inner_model.model_category is None:
                    raise ValueError(f"Error: Dynamic AI models must have model_category set! {inner_model} does not have model_category set and needs a migration")
                if inner_model.model_version is None:
                    raise ValueError(f"Error: Dynamic AI models must have model_version set! {inner_model} does not have model_version set and needs a migration")
                # model_image_size describes what the SHARED frame preprocessor must
                # produce. Models that never consume it have no size to declare:
                # audio models, and asset-scope models that drive their own decode
                # (e.g. temporal segmentation over a whole video).
                is_audio = bool(inner_model.model_type) and "audio" in inner_model.model_type.lower()
                uses_frame_preprocessor = not is_audio and not is_whole_asset_model(inner_model)
                if uses_frame_preprocessor and inner_model.model_image_size is None:
                    raise ValueError(f"Error: Dynamic AI models must have model_image_size set! {inner_model} does not have model_image_size set and needs a migration")
            except ValueError as e:
                if second_pass:
                    raise e
                self.logger.warning(f"Detected old model files. Attempting migration...")
                migrate_to_2_0()
                self.logger.info("Migration complete. Verifying models again...")
                models = []
                for model_name in self.ai_model_names:
                    models.append(self.model_manager.get_and_refresh_model(model_name))
                self.__verify_models(models, True)
                return

            if first_image_size is None and uses_frame_preprocessor:
                first_image_size = inner_model.model_image_size
            if first_norm_config is None and uses_frame_preprocessor:
                first_norm_config = inner_model.normalization_config
        if first_image_size is not None:
            self.image_size = first_image_size
        if first_norm_config is not None:
            self.normalization_config = first_norm_config
        self.logger.debug("Finished verifying dynamic AI models")

    def _resolve_active_model_names(self):
        explicit_names = _normalize_string_list(self.configfile.get("active_ai_models", [])) or []
        capability_flags = self.configfile.get("active_capabilities", None)
        capability_groups = self.configfile.get("capability_model_groups", None)

        if capability_flags is None:
            return _dedupe_strings(explicit_names)

        if not isinstance(capability_flags, dict):
            self.logger.warning("active_capabilities must be a mapping of capability->bool; ignoring malformed value.")
            return _dedupe_strings(explicit_names)

        enabled_capabilities = []
        for capability_name, enabled in capability_flags.items():
            if capability_name not in ["tagging", "detection", "embedding"]:
                self.logger.warning(f"Unknown active_capabilities key '{capability_name}' ignored.")
                continue
            if bool(enabled):
                enabled_capabilities.append(capability_name)

        selected_names = list(explicit_names)
        if enabled_capabilities:
            if isinstance(capability_groups, dict) and capability_groups:
                for capability_name in enabled_capabilities:
                    group_models = _normalize_string_list(capability_groups.get(capability_name, [])) or []
                    selected_names.extend(group_models)
            else:
                self.logger.info("active_capabilities enabled without capability_model_groups; using auto-discovered model capability index.")
                discovered_index = self._build_capability_model_index()
                for capability_name in enabled_capabilities:
                    selected_names.extend(discovered_index.get(capability_name, []))

        deduped = _dedupe_strings(selected_names)
        self.logger.info(
            f"Resolved active AI models: {len(deduped)} model(s) from explicit list + capability flags {enabled_capabilities}"
        )
        return deduped

    def _build_capability_model_index(self):
        capability_index = {"tagging": [], "detection": [], "embedding": []}
        models_directory = "./config/models"

        if not os.path.isdir(models_directory):
            self.logger.warning("Model config directory ./config/models not found while building capability index.")
            return capability_index

        for file_name in os.listdir(models_directory):
            if not file_name.endswith(".yaml"):
                continue

            model_name = file_name[:-5]
            model_config = load_config(os.path.join(models_directory, file_name), default_config={}) or {}
            model_type = model_config.get("type", "")
            if model_type not in ("model", "face_torch_export"):
                continue

            capabilities = self._infer_model_capabilities(model_config)
            for capability in capabilities:
                if capability in capability_index:
                    capability_index[capability].append(model_name)

        for capability, names in capability_index.items():
            capability_index[capability] = _dedupe_strings(names)

        return capability_index

    def _infer_model_capabilities(self, model_config):
        configured = model_config.get("model_capabilities", model_config.get("capabilities", None))
        normalized = _normalize_string_list(configured)
        if normalized:
            return [value for value in normalized if value in ["tagging", "detection", "embedding"]] or ["tagging"]

        model_type = str(model_config.get("model_type", "")).lower()
        if "detect" in model_type:
            return ["detection"]
        if "embed" in model_type:
            return ["embedding"]
        return ["tagging"]


def _normalize_string_list(value: Optional[Iterable[str]]) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else None
    normalized = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized or None


def _downstream_model_names(*model_groups) -> set:
    """Config/file names of every model fed by a preprocessor.

    Matches what the per-model skip gate compares `requested_model_names`
    against, so the preprocessor can make the same decision one step earlier.
    """
    names = set()
    for group in model_groups:
        for model in group or []:
            inner = getattr(model, "model", model)
            for attribute in ("config_name", "model_file_name"):
                value = str(getattr(inner, attribute, "") or "").strip()
                if value:
                    names.add(value)
    return names


def _dedupe_strings(values: List[str]) -> List[str]:
    seen = set()
    deduped = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _format_label_filter_suffix(labels: List[str]) -> str:
    normalized = [_sanitize_key(label) for label in labels if str(label).strip()]
    normalized = [label for label in normalized if label]
    if not normalized:
        return ""
    return "__labels__" + "__or__".join(normalized)


def _sanitize_key(value: str) -> str:
    text = str(value).strip()
    if not text:
        return "unknown"
    return re.sub(r"[^0-9a-zA-Z_]+", "_", text)

