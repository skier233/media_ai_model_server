
import asyncio
import logging
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from lib.async_lib.async_processing import ItemFuture
from lib.model.preprocessing_python.image_preprocessing import get_video_duration_av
from lib.model.postprocessing.AI_VideoResult import AIVideoResult, AIVideoResultV3
import lib.model.postprocessing.timeframe_processing as timeframe_processing
from lib.model.postprocessing.category_settings import category_config
from lib.model.skip_input import Skip
from lib.model.postprocessing.post_processing_settings import get_or_default, post_processing_config
from lib.pipeline.target_primitives import build_region_target, targets_to_dicts

logger = logging.getLogger("logger")

async def result_coalescer(data):
    for item in data:
        itemFuture = item.item_future
        result = {}
        structured_outputs = []
        structured_errors = []
        for input_name in item.input_names:
            ai_result = itemFuture[input_name]
            if isinstance(ai_result, Skip):
                structured_outputs.append(
                    {
                        "model_step": input_name,
                        "status": "skipped",
                    }
                )
                continue

            if isinstance(ai_result, Exception):
                error_text = str(ai_result)
                structured_outputs.append(
                    {
                        "model_step": input_name,
                        "status": "error",
                        "error": error_text,
                    }
                )
                structured_errors.append({"model_step": input_name, "error": error_text})
                continue

            result[input_name] = ai_result
            structured_outputs.append(
                {
                    "model_step": input_name,
                    "status": "ok",
                }
            )

        result["_outputs"] = structured_outputs
        if structured_errors:
            result["_errors"] = structured_errors
        await itemFuture.set_data(item.output_names[0], result)
        
async def asset_result_collector(data):
    """Collect asset-scope model outputs into one dict on the parent future.

    A sibling of :func:`result_coalescer` for the asset stage.  ``input_names[0]``
    is the asset path and acts purely as a trigger, so the stage still completes
    — with an empty result — on the common deployment that has no asset-scope
    model installed at all.  It is deliberately excluded from the result, which
    is why this cannot just reuse ``result_coalescer``.
    """
    for item in data:
        itemFuture = item.item_future
        result = {}
        structured_outputs = []
        structured_errors = []
        for input_name in item.input_names[1:]:
            ai_result = itemFuture[input_name]
            if isinstance(ai_result, Skip):
                skipped = {"model_step": input_name, "status": "skipped"}
                if getattr(ai_result, "reason", None):
                    skipped["reason"] = ai_result.reason
                structured_outputs.append(skipped)
                continue

            if isinstance(ai_result, Exception):
                error_text = str(ai_result) or type(ai_result).__name__
                structured_outputs.append(
                    {"model_step": input_name, "status": "error", "error": error_text}
                )
                structured_errors.append({"model_step": input_name, "error": error_text})
                continue

            result[input_name] = ai_result
            structured_outputs.append({"model_step": input_name, "status": "ok"})

        result["_outputs"] = structured_outputs
        if structured_errors:
            result["_errors"] = structured_errors
        await itemFuture.set_data(item.output_names[0], result)

async def result_finisher(data):
    for item in data:
        itemFuture = item.item_future
        future_results = itemFuture[item.input_names[0]]
        itemFuture.close_future(future_results)

async def batch_awaiter(data):
    """
    Non-blocking child-future joiner.

    Instead of blocking the worker while waiting for children to complete,
    this spawns a lightweight async task per item and returns immediately.
    The worker is freed to process other queued items. When all children
    finish, the task writes the aggregated result back via set_data,
    which re-enters the pipeline event handler normally.

    This means batch_awaiter can safely be nested (e.g. video frame children
    that themselves have region children) without deadlocking, because no
    worker is held while waiting.
    """
    for item in data:
        itemFuture = item.item_future
        child_futures = itemFuture[item.input_names[0]]
        if child_futures is None:
            child_futures = []
        if not isinstance(child_futures, list):
            child_futures = [child_futures]

        # Fire-and-forget: spawn a task that monitors children and writes
        # the result back to the parent future when they all complete.
        # The worker returns immediately after this loop.
        asyncio.create_task(
            _join_child_futures(itemFuture, child_futures, item.output_names[0])
        )


async def _join_child_futures(parent_future, child_futures, output_name):
    """Await all child futures and write aggregated results to the parent."""
    try:
        # Collect the underlying asyncio.Future from each ItemFuture
        raw_futures = []
        for child in child_futures:
            if hasattr(child, "future") and child.future is not None:
                raw_futures.append(child.future)
            elif isinstance(child, asyncio.Future):
                raw_futures.append(child)
            else:
                raw_futures.append(asyncio.ensure_future(child))

        if not raw_futures:
            await parent_future.set_data(output_name, [])
            return

        results = await asyncio.gather(*raw_futures, return_exceptions=True)

        normalized_results = []
        for result in results:
            if isinstance(result, Exception):
                normalized_results.append({"_error": str(result), "_status": "error"})
            elif isinstance(result, dict):
                normalized_results.append(dict(result))
            else:
                normalized_results.append(result)

        await parent_future.set_data(output_name, normalized_results)
    except Exception as e:
        logger.error(f"_join_child_futures failed: {e}", exc_info=True)
        try:
            parent_future.set_exception(e)
        except Exception:
            pass

async def video_result_postprocessor(data):
    for item in data:
        itemFuture = item.item_future
        duration = get_video_duration_av(itemFuture[item.input_names[1]])
        result = {"frames": itemFuture[item.input_names[0]], "video_duration": duration, "frame_interval": float(itemFuture[item.input_names[2]]), "threshold": float(itemFuture[item.input_names[3]]), "ai_models_info": itemFuture['pipeline'].get_ai_models_info()}
        del itemFuture.data["pipeline"]

        videoResult = itemFuture[item.input_names[4]]
        if videoResult is not None:
            videoResult.add_server_result(result)
        else:
            videoResult = AIVideoResult.from_server_result(result)

        toReturn = {"json_result": videoResult.to_json(), "video_tag_info": timeframe_processing.compute_video_tag_info(videoResult)}
        
        await itemFuture.set_data(item.output_names[0], toReturn)

async def video_result_postprocessor_v3(data):
    for item in data:
        itemFuture = item.item_future
        duration = get_video_duration_av(itemFuture[item.input_names[1]])
        pipeline = itemFuture['pipeline']
        currently_active_models = pipeline.get_ai_models_info()

        raw_frames = itemFuture[item.input_names[0]]
        clean_frames, _, _ = _extract_structured_frames(raw_frames)
        timespan_frames = _filter_timespan_compatible_frames(clean_frames)
        per_frame_data = _extract_per_frame_data(clean_frames)

        raw_frame_interval = itemFuture[item.input_names[2]]
        raw_threshold = itemFuture[item.input_names[3]]

        result = {
            "frames": timespan_frames,
            "per_frame_data": per_frame_data,
            "video_duration": duration,
            "frame_interval": float(raw_frame_interval) if raw_frame_interval is not None else 0.5,
            "threshold": float(raw_threshold) if raw_threshold is not None else 0.5,
            "ai_models_info": currently_active_models,
            "skipped_categories": itemFuture[item.input_names[4]],
        }

        if "pipeline" in itemFuture.data:
            del itemFuture.data["pipeline"]

        used_models = [
            model for model in currently_active_models
            if not all(
                category in (result.get("skipped_categories") or [])
                for category in model.categories
            )
        ]
        result["models"] = used_models
        max_merge_seconds = post_processing_config.get("max_timespan_merge_seconds", 2)

        try:
            videoResult = AIVideoResultV3.from_server_result(result, max_merge_seconds=max_merge_seconds)
        except Exception as e:
            logger.error(f"Error creating AIVideoResultV3: {e}")
            raise

        root_future = getattr(itemFuture, "root_future", itemFuture)
        metrics_source = getattr(root_future, "_pipeline_metrics", {}) or {}
        metrics = dict(metrics_source)
        metrics.setdefault("preprocess_seconds", 0.0)
        metrics.setdefault("ai_inference_seconds", 0.0)
        metrics["ai_model_count"] = len(currently_active_models)

        total_runtime = metrics.get("preprocess_seconds", 0.0) + metrics.get("ai_inference_seconds", 0.0)
        metrics["total_runtime_seconds"] = total_runtime

        payload = {"result": videoResult, "metrics": metrics}

        await itemFuture.set_data(item.output_names[0], payload)

async def image_result_postprocessor(data):
    toReturn = {}
    for item in data:
        itemFuture = item.item_future
        result = itemFuture[item.input_names[0]]
        for category, tags in result.items():
            if category not in category_config:
                continue
            toReturn[category] = []
            for tag in tags:
                if isinstance(tag, tuple):
                    tagname, confidence = tag
                    if tagname not in category_config[category]:
                        continue
                    
                    tag_threshold = float(get_or_default(category_config[category][tagname], 'TagThreshold', 0.5))
                    renamed_tag = category_config[category][tagname]['RenamedTag']

                    if not post_processing_config["use_category_image_thresholds"]:
                        toReturn[category].append((renamed_tag, confidence))
                    elif confidence >= tag_threshold:
                        toReturn[category].append((renamed_tag, confidence))
                else:
                    if tag not in category_config[category]:
                        continue
                    renamed_tag = category_config[category][tag]['RenamedTag']
                    toReturn[category].append(renamed_tag)


        await itemFuture.set_data(item.output_names[0], toReturn)

async def image_result_postprocessor_v3(data):
    for item in data:
        itemFuture = item.item_future
        pipeline = itemFuture['pipeline']
        result = itemFuture[item.input_names[0]]
        coalesced_outputs = []
        coalesced_errors = []
        if isinstance(result, dict):
            coalesced_outputs = list(result.get("_outputs") or [])
            coalesced_errors = list(result.get("_errors") or [])
            result = {
                key: value
                for key, value in result.items()
                if not str(key).startswith("_")
            }
        # v3 API: do not rely on category_config for filtering/renaming.
        # Preserve the model output shape (category -> list of tags/tuples) as-is.
        toReturn = {}
        if isinstance(result, dict):
            for category, tags in result.items():
                if tags is None:
                    toReturn[category] = []
                    continue
                if isinstance(tags, (list, tuple)):
                    toReturn[category] = list(tags)
                else:
                    # Defensive: if a backend returns a single tag, wrap it.
                    toReturn[category] = [tags]
        else:
            # Defensive: unexpected shape, return as-is.
            toReturn = result

        root_future = getattr(itemFuture, "root_future", itemFuture)
        metrics_source = getattr(root_future, "_pipeline_metrics", {}) or {}
        metrics = dict(metrics_source)
        metrics.setdefault("preprocess_seconds", 0.0)
        metrics.setdefault("ai_inference_seconds", 0.0)
        metrics["ai_model_count"] = len(pipeline.get_ai_models_info())

        total_runtime = metrics.get("preprocess_seconds", 0.0) + metrics.get("ai_inference_seconds", 0.0)
        metrics["total_runtime_seconds"] = total_runtime

        source_asset_id = itemFuture["image_path"]
        target = {
            "target_id": "asset:0",
            "scope": "asset",
            "source_asset_id": str(source_asset_id) if source_asset_id is not None else "unknown",
        }
        results_v2 = {
            "assets": [
                {
                    "asset_id": target["source_asset_id"],
                    "targets": [target],
                    "outputs": coalesced_outputs,
                    "errors": coalesced_errors,
                }
            ],
            "errors": coalesced_errors,
        }

        if "pipeline" in itemFuture.data:
            del itemFuture.data["pipeline"]

        payload = {"result": toReturn, "results_v2": results_v2, "metrics": metrics}

        await itemFuture.set_data(item.output_names[0], payload)


async def image_result_postprocessor_v4(data):
    for item in data:
        itemFuture = item.item_future
        pipeline = itemFuture['pipeline']
        result = itemFuture[item.input_names[0]]
        if not isinstance(result, dict):
            result = {"value": result}

        coalesced_outputs = list(result.get("_outputs") or [])
        coalesced_errors = list(result.get("_errors") or [])
        cleaned_result = {
            key: value
            for key, value in result.items()
            if not str(key).startswith("_")
        }

        current_models = pipeline.get_ai_models_info()
        requested_model_names = _normalize_requested_model_names(itemFuture["requested_model_names"])
        used_models = _select_used_models(current_models, requested_model_names)
        capability_index = _build_category_capability_index(used_models or current_models)
        analysis = _normalize_analysis_payload(cleaned_result, capability_index)

        metrics = _collect_pipeline_metrics(itemFuture, current_models)
        source_asset_id = itemFuture["image_path"]
        target = {
            "target_id": "asset:0",
            "scope": "asset",
            "source_asset_id": str(source_asset_id) if source_asset_id is not None else "unknown",
        }
        results_v2 = {
            "assets": [
                {
                    "asset_id": target["source_asset_id"],
                    "targets": [target],
                    "outputs": coalesced_outputs,
                    "errors": coalesced_errors,
                }
            ],
            "errors": coalesced_errors,
        }

        if "pipeline" in itemFuture.data:
            del itemFuture.data["pipeline"]

        payload = {
            "asset_id": target["source_asset_id"],
            "requested_model_names": requested_model_names,
            "models": used_models,
            "analysis": analysis,
            "results_v2": results_v2,
            "metrics": metrics,
        }
        await itemFuture.set_data(item.output_names[0], payload)


async def video_result_postprocessor_v4(data):
    for item in data:
        itemFuture = item.item_future
        duration = get_video_duration_av(itemFuture[item.input_names[1]])
        pipeline = itemFuture['pipeline']
        current_models = pipeline.get_ai_models_info()
        requested_model_names = _normalize_requested_model_names(itemFuture["requested_model_names"])
        used_models = _select_used_models(current_models, requested_model_names)
        capability_index = _build_category_capability_index(used_models or current_models)

        raw_frames = itemFuture[item.input_names[0]]
        normalized_frames = []
        if not isinstance(raw_frames, list):
            raw_frames = []
        for frame in raw_frames:
            if not isinstance(frame, dict):
                normalized_frames.append({"raw": frame})
                continue
            frame_index = frame.get("frame_index")
            frame_clean = {
                key: value
                for key, value in frame.items()
                if key != "frame_index" and not str(key).startswith("_")
            }
            frame_analysis = _normalize_analysis_payload(frame_clean, capability_index)
            normalized_frame = {
                "frame_index": frame_index,
                "time_seconds": frame_index,
                "analysis": frame_analysis,
            }
            frame_outputs = list(frame.get("_outputs") or [])
            frame_errors = list(frame.get("_errors") or [])
            if frame_outputs:
                normalized_frame["outputs"] = frame_outputs
            if frame_errors:
                normalized_frame["errors"] = frame_errors
            normalized_frames.append(normalized_frame)

        if "pipeline" in itemFuture.data:
            del itemFuture.data["pipeline"]

        metrics = _collect_pipeline_metrics(itemFuture, current_models)
        raw_frame_interval = itemFuture[item.input_names[2]]
        payload = {
            "asset_id": str(itemFuture[item.input_names[1]]),
            "requested_model_names": requested_model_names,
            "models": used_models,
            "duration_seconds": duration,
            "frame_interval_seconds": float(raw_frame_interval) if raw_frame_interval is not None else 0.5,
            "frame_count": len(normalized_frames),
            "frames": normalized_frames,
            "metrics": metrics,
        }
        _apply_asset_results(
            payload,
            itemFuture[item.input_names[3]] if len(item.input_names) > 3 else None,
            capability_index,
        )
        await itemFuture.set_data(item.output_names[0], payload)


def _apply_asset_results(payload, asset_results, capability_index):
    """Lift asset-scope stage output onto the response as a top-level ``analysis``.

    Whole-asset results are not frame data, so they do not belong under
    ``frames``.  They are emitted with exactly the shape the image pipeline
    already uses for its asset-level analysis, so a client needs one code path
    for both: known capabilities bucket under ``analysis.capabilities``, and
    anything else (temporal segmentation included) under ``analysis.other``.

    The key is absent when the stage did not run, so a caller can tell "not
    requested" from "ran and found nothing".
    """
    if not isinstance(asset_results, dict):
        return

    values = {
        key: value
        for key, value in asset_results.items()
        if not str(key).startswith("_") and not isinstance(value, Skip)
    }
    if values:
        payload["analysis"] = _normalize_analysis_payload(values, capability_index)

    outputs = list(asset_results.get("_outputs") or [])
    errors = list(asset_results.get("_errors") or [])
    if outputs:
        payload.setdefault("asset_outputs", outputs)
    if errors:
        payload.setdefault("asset_errors", errors)


async def audio_result_postprocessor_v4(data):
    for item in data:
        itemFuture = item.item_future
        children_results = itemFuture[item.input_names[0]]
        audio_path = itemFuture[item.input_names[1]] if len(item.input_names) > 1 else "unknown"
        pipeline = itemFuture['pipeline']
        current_models = pipeline.get_ai_models_info()
        requested_model_names = _normalize_requested_model_names(itemFuture["requested_model_names"])
        used_models = _select_used_models(current_models, requested_model_names)
        capability_index = _build_category_capability_index(used_models or current_models)

        if not isinstance(children_results, list):
            children_results = []

        windows = []
        for window in children_results:
            if not isinstance(window, dict):
                windows.append({"raw": window})
                continue
            window_clean = {
                key: value
                for key, value in window.items()
                if key not in {"window_index", "window_start", "window_end"} and not str(key).startswith("_")
            }
            normalized_window = {
                "index": window.get("window_index"),
                "start": window.get("window_start"),
                "end": window.get("window_end"),
                "analysis": _normalize_analysis_payload(window_clean, capability_index),
            }
            window_outputs = list(window.get("_outputs") or [])
            window_errors = list(window.get("_errors") or [])
            if window_outputs:
                normalized_window["outputs"] = window_outputs
            if window_errors:
                normalized_window["errors"] = window_errors
            windows.append(normalized_window)

        if "pipeline" in itemFuture.data:
            del itemFuture.data["pipeline"]

        metrics = _collect_pipeline_metrics(itemFuture, current_models)
        payload = {
            "asset_id": str(audio_path),
            "requested_model_names": requested_model_names,
            "models": used_models,
            "window_count": len(windows),
            "windows": windows,
            "metrics": metrics,
        }
        await itemFuture.set_data(item.output_names[0], payload)


async def audio_result_postprocessor(data):
    """Post-processor for the audio embedding pipeline.

    Receives per-window collated results from batch_awaiter, then:
            1. Classification filtering - reject windows dominated by music
            2. Type-binning - classify kept windows as moan/speech/breath
            3. Per-type centroid embedding - mean of audio embedding vectors per type, L2-normalised
            4. Overall centroid - mean of all kept audio embedding vectors
    """
    for item in data:
        itemFuture = item.item_future
        children_results = itemFuture[item.input_names[0]]  # list of per-window dicts
        audio_path = itemFuture[item.input_names[1]] if len(item.input_names) > 1 else "unknown"

        root_future = getattr(itemFuture, "root_future", itemFuture)
        metrics_source = getattr(root_future, "_pipeline_metrics", {}) or {}
        metrics = dict(metrics_source)

        if not isinstance(children_results, list):
            children_results = []

        # Classification category indices
        # Whitelist: human non-speech vocalizations we want to keep
        IDX_WHITELIST = [8, 9, 14, 22, 24, 25, 38, 39, 44, 45, 46]
        # Speech
        IDX_SPEECH = [0, 1, 2, 3, 4, 5, 15]
        # Music (reject if dominant)
        IDX_MUSIC = [
            27, 28, 32, 33, 34, 137, 138, 140, 141, 142, 143, 144, 145, 146,
            153, 154, 162, 163, 164, 165, 166, 167, 168, 184, 266, 270,
        ]
        # Type bins for kept windows
        IDX_MOAN = [25, 38, 39, 22, 24, 44, 45, 46, 14, 8, 9]
        IDX_BREATH = [41, 26, 43]

        threshold = 0.01  # Classifier sigmoid scores are typically low for specific classes

        kept_windows = []
        rejected_music = 0
        rejected_quiet = 0

        for win in children_results:
            if not isinstance(win, dict):
                continue
            if "_error" in win:
                continue

            # Extract classifier probabilities
            classifier_list = win.get("audio_classification_audioclass", [])
            probs = None
            if classifier_list and isinstance(classifier_list, list) and isinstance(classifier_list[0], dict):
                probs = classifier_list[0].get("probabilities")

            if probs is None:
                # No classification available — keep by default
                kept_windows.append(win)
                continue

            # Compute category scores
            music_score = max((probs[i] for i in IDX_MUSIC if i < len(probs)), default=0.0)
            whitelist_score = max((probs[i] for i in IDX_WHITELIST if i < len(probs)), default=0.0)
            speech_score = max((probs[i] for i in IDX_SPEECH if i < len(probs)), default=0.0)

            # Reject music-dominated windows
            if music_score > 0.05 and music_score > max(whitelist_score, speech_score):
                rejected_music += 1
                continue

            # Keep if any relevant vocal activity
            keep_score = max(whitelist_score, speech_score)
            if keep_score < threshold:
                rejected_quiet += 1
                continue

            # Annotate window with scores for downstream use
            win["_scores"] = {
                "whitelist": round(whitelist_score, 4),
                "speech": round(speech_score, 4),
                "music": round(music_score, 4),
            }
            kept_windows.append(win)

        # ── Type-binning ──
        type_bins = {"moan": [], "speech": [], "breath": []}

        for win in kept_windows:
            probs = None
            classifier_list = win.get("audio_classification_audioclass", [])
            if classifier_list and isinstance(classifier_list, list) and isinstance(classifier_list[0], dict):
                probs = classifier_list[0].get("probabilities")

            # Extract embedding vector
            emb_list = win.get("audio_embeddings_audioembed", [])
            vector = None
            if emb_list and isinstance(emb_list, list) and isinstance(emb_list[0], dict):
                vector = emb_list[0].get("vector")

            if vector is None:
                continue

            if probs is not None:
                moan_score = max((probs[i] for i in IDX_MOAN if i < len(probs)), default=0.0)
                speech_score = max((probs[i] for i in IDX_SPEECH if i < len(probs)), default=0.0)
                breath_score = max((probs[i] for i in IDX_BREATH if i < len(probs)), default=0.0)

                scores = {"moan": moan_score, "speech": speech_score, "breath": breath_score}
                dominant = max(scores, key=scores.get)
                win["_dominant_type"] = dominant
            else:
                dominant = "moan"  # default bin
                win["_dominant_type"] = dominant

            type_bins[dominant].append(vector)

        # ── Per-type centroid embeddings ──
        embeddings = {}
        for type_name, vectors in type_bins.items():
            if not vectors:
                continue
            arr = np.array(vectors, dtype=np.float32)
            centroid = arr.mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm > 1e-8:
                centroid = centroid / norm
            embeddings[type_name] = {
                "centroid": centroid.tolist(),
                "norm": round(float(np.linalg.norm(centroid)), 6),
                "dim": int(centroid.shape[0]),
                "window_count": len(vectors),
            }

        # ── Overall centroid ──
        all_vectors = [v for vecs in type_bins.values() for v in vecs]
        overall_centroid = None
        if all_vectors:
            arr = np.array(all_vectors, dtype=np.float32)
            centroid = arr.mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm > 1e-8:
                centroid = centroid / norm
            overall_centroid = {
                "centroid": centroid.tolist(),
                "norm": round(float(np.linalg.norm(centroid)), 6),
                "dim": int(centroid.shape[0]),
                "window_count": len(all_vectors),
            }

        # ── Duration computation (merge overlapping windows per type) ──
        def _merge_intervals(intervals):
            """Merge overlapping/adjacent (start, end) intervals, return total seconds."""
            if not intervals:
                return 0.0, []
            sorted_iv = sorted(intervals)
            merged = [sorted_iv[0]]
            for s, e in sorted_iv[1:]:
                if s <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], e))
                else:
                    merged.append((s, e))
            total = sum(e - s for s, e in merged)
            return total, merged

        type_windows = {"moan": [], "speech": [], "breath": []}
        all_kept_intervals = []
        for win in kept_windows:
            start = win.get("window_start")
            end = win.get("window_end")
            if start is None or end is None:
                continue
            dominant = win.get("_dominant_type", "unknown")
            if dominant in type_windows:
                type_windows[dominant].append((start, end))
            all_kept_intervals.append((start, end))

        duration_by_type = {}
        for type_name, intervals in type_windows.items():
            secs, _ = _merge_intervals(intervals)
            if secs > 0:
                duration_by_type[type_name] = round(secs, 2)

        total_kept_duration, _ = _merge_intervals(all_kept_intervals)

        # ── Build response ──
        metrics["windows_total"] = len(children_results)
        metrics["windows_kept"] = len(kept_windows)
        metrics["windows_rejected_music"] = rejected_music
        metrics["windows_rejected_quiet"] = rejected_quiet
        metrics["type_counts"] = {k: len(v) for k, v in type_bins.items()}

        if hasattr(itemFuture, "data") and isinstance(itemFuture.data, dict):
            itemFuture.data.pop("pipeline", None)

        payload = {
            "embeddings": embeddings,
            "overall_embedding": overall_centroid,
            "classification_summary": {
                "windows_analyzed": len(children_results),
                "windows_kept": len(kept_windows),
                "rejected_music": rejected_music,
                "rejected_quiet": rejected_quiet,
                "type_distribution": {k: len(v) for k, v in type_bins.items()},
                "duration_seconds": duration_by_type,
                "total_kept_duration_seconds": round(total_kept_duration, 2),
            },
            "windows": [
                {
                    "index": w.get("window_index"),
                    "start": w.get("window_start"),
                    "end": w.get("window_end"),
                    "dominant_type": w.get("_dominant_type", "unknown"),
                    "scores": w.get("_scores", {}),
                }
                for w in kept_windows
            ],
            "metrics": metrics,
            "source": str(audio_path),
        }

        await itemFuture.set_data(item.output_names[0], payload)


async def detector_result_to_region_targets(data):
    max_targets_raw = os.environ.get("MAX_REGION_TARGETS_PER_ITEM", "64")
    try:
        max_targets_per_item = int(max_targets_raw)
    except Exception:
        max_targets_per_item = 64

    for item in data:
        itemFuture = item.item_future
        detections = itemFuture[item.input_names[0]]
        source_asset_id = itemFuture[item.input_names[1]]

        frame_index = None
        source_tensor = None
        detector_tensor = None
        parent_target_id = None
        bbox_scale_x = 1.0
        bbox_scale_y = 1.0

        if len(item.input_names) > 2:
            third_input = itemFuture[item.input_names[2]]
            if isinstance(third_input, torch.Tensor):
                source_tensor = third_input
            else:
                frame_index = third_input

        if len(item.input_names) > 3:
            fourth_input = itemFuture[item.input_names[3]]
            if source_tensor is None and isinstance(fourth_input, torch.Tensor):
                source_tensor = fourth_input
            elif isinstance(fourth_input, torch.Tensor):
                detector_tensor = fourth_input
            elif frame_index is None:
                frame_index = fourth_input
            else:
                parent_target_id = fourth_input

        if len(item.input_names) > 4:
            fifth_input = itemFuture[item.input_names[4]]
            if isinstance(fifth_input, torch.Tensor):
                detector_tensor = fifth_input
            else:
                parent_target_id = fifth_input

        source_height, source_width = _extract_tensor_hw(source_tensor)

        # Compute bbox scale AFTER source dimensions are known.
        if isinstance(detector_tensor, torch.Tensor):
            det_h, det_w = _extract_tensor_hw(detector_tensor)
            if (det_h and det_w and source_height and source_width
                    and (det_h != source_height or det_w != source_width)):
                bbox_scale_x = source_width / det_w
                bbox_scale_y = source_height / det_h

        candidate_detections = _extract_detection_items(detections)

        if max_targets_per_item > 0 and len(candidate_detections) > max_targets_per_item:
            sortable = []
            unsorted = []
            for det in candidate_detections:
                if isinstance(det, dict) and isinstance(det.get("score", None), (int, float)):
                    sortable.append(det)
                else:
                    unsorted.append(det)
            sortable.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
            candidate_detections = (sortable + unsorted)[:max_targets_per_item]
            logger.warning(
                "Capped region targets to %s for source '%s'",
                max_targets_per_item,
                source_asset_id,
            )

        region_targets = []
        region_errors = []
        for detection_index, detection in enumerate(candidate_detections):
            bbox = _extract_detection_bbox(detection)
            if bbox is None:
                region_errors.append(
                    {
                        "index": detection_index,
                        "error": "missing_bbox",
                        "raw_detection": detection,
                    }
                )
                continue

            # Scale bbox from detector-input space to region-source space
            if bbox_scale_x != 1.0 or bbox_scale_y != 1.0:
                bbox = [bbox[0] * bbox_scale_x, bbox[1] * bbox_scale_y,
                        bbox[2] * bbox_scale_x, bbox[3] * bbox_scale_y]

            # Scale kps similarly
            kps_raw = detection.get("kps") if isinstance(detection, dict) else None
            if kps_raw and (bbox_scale_x != 1.0 or bbox_scale_y != 1.0):
                kps_raw = [[x * bbox_scale_x, y * bbox_scale_y] for x, y in kps_raw]

            detection_labels = _extract_detection_labels(detection)
            metadata = {
                "detection_index": detection_index,
                "kps": kps_raw,
            }
            if detection_labels:
                metadata["label"] = detection_labels[0]
                metadata["labels"] = detection_labels

            try:
                target = build_region_target(
                    source_asset_id=str(source_asset_id),
                    bbox=bbox,
                    frame_index=frame_index,
                    parent_target_id=parent_target_id,
                    source_width=source_width,
                    source_height=source_height,
                    metadata=metadata,
                )
                region_targets.append(target)
            except Exception as exc:
                region_errors.append(
                    {
                        "index": detection_index,
                        "error": str(exc),
                        "raw_detection": detection,
                    }
                )

        # Write client-safe detections (without kps, with normalized bbox) back
        # to the detection key.  Downstream steps (region_children_builder,
        # embedding models) already extracted kps/bbox into region-target
        # metadata, so the original value is no longer needed.  Direct
        # mutation avoids re-triggering the pipeline event handler — safe
        # because the detection key has already been consumed by this step.
        if itemFuture.data is not None:
            can_normalize = (source_width is not None and source_height is not None
                             and source_width > 0 and source_height > 0)
            client_detections = []
            for det in candidate_detections:
                if isinstance(det, dict):
                    client_det = {k: v for k, v in det.items() if k != "kps"}
                    if can_normalize and "bbox" in client_det:
                        bx1, by1, bx2, by2 = client_det["bbox"]
                        # Scale from detector-input space to region-source
                        # space before normalising to 0–1.
                        bx1 *= bbox_scale_x; by1 *= bbox_scale_y
                        bx2 *= bbox_scale_x; by2 *= bbox_scale_y
                        client_det["bbox"] = [
                            max(0.0, min(1.0, bx1 / source_width)),
                            max(0.0, min(1.0, by1 / source_height)),
                            max(0.0, min(1.0, bx2 / source_width)),
                            max(0.0, min(1.0, by2 / source_height)),
                        ]
                    client_detections.append(client_det)
                else:
                    client_detections.append(det)
            itemFuture.data[item.input_names[0]] = client_detections

        await itemFuture.set_data(item.output_names[0], targets_to_dicts(region_targets))
        if len(item.output_names) > 1:
            await itemFuture.set_data(item.output_names[1], region_errors)


async def region_children_builder(data):
    for item in data:
        itemFuture = item.item_future
        source_tensor = itemFuture[item.input_names[0]]
        region_targets = itemFuture[item.input_names[1]] or []
        threshold = itemFuture[item.input_names[2]] if len(item.input_names) > 2 else None
        return_confidence = itemFuture[item.input_names[3]] if len(item.input_names) > 3 else None
        skipped_categories = itemFuture[item.input_names[4]] if len(item.input_names) > 4 else None
        label_filter = _extract_region_label_filter(item.output_names)

        children = []
        if not isinstance(source_tensor, torch.Tensor):
            raise ValueError("region_children_builder requires source_tensor input as a torch.Tensor")

        source_height, source_width = _extract_tensor_hw(source_tensor)
        if source_height is None or source_width is None:
            raise ValueError("region_children_builder could not determine source tensor dimensions")

        for region_target in region_targets:
            if label_filter and not _region_target_matches_label_filter(region_target, label_filter):
                continue

            bbox = region_target.get("bbox") if isinstance(region_target, dict) else None
            if bbox is None:
                continue

            # Extract detection_index for correlation; the full region
            # target stays available for models that need it (e.g. kps
            # for face alignment) but won't be coalesced into the result.
            det_index = None
            if isinstance(region_target, dict):
                meta = region_target.get("metadata") or {}
                det_index = meta.get("detection_index")

            crop_tensor = _crop_and_resize_region(source_tensor, bbox)
            payload = {
                item.output_names[1]: crop_tensor,
                item.output_names[2]: region_target,
                item.output_names[3]: threshold,
                item.output_names[4]: return_confidence,
                item.output_names[5]: skipped_categories,
            }
            if len(item.output_names) > 6:
                payload[item.output_names[6]] = source_tensor
            if len(item.output_names) > 7:
                payload[item.output_names[7]] = det_index
            child_future = await ItemFuture.create(item, payload, item.item_future.handler)
            children.append(child_future)

        # Release the parent frame's reference to the full-resolution
        # source tensor.  Children that need it (face alignment) already
        # hold their own reference.  Clearing it here allows GC to reclaim
        # the large tensor as soon as all children finish with it.
        if itemFuture.data is not None:
            src_key = item.input_names[0]
            if src_key in itemFuture.data:
                itemFuture.data[src_key] = None

        await itemFuture.set_data(item.output_names[0], children)


def _extract_tensor_hw(source_tensor):
    if isinstance(source_tensor, torch.Tensor):
        if source_tensor.dim() == 3:
            return int(source_tensor.shape[-2]), int(source_tensor.shape[-1])
        if source_tensor.dim() == 4:
            return int(source_tensor.shape[-2]), int(source_tensor.shape[-1])
    return None, None


def _extract_detection_items(detections):
    if detections is None:
        return []
    if isinstance(detections, dict):
        if "detections" in detections and isinstance(detections["detections"], list):
            return detections["detections"]
        for value in detections.values():
            if isinstance(value, list) and value:
                first = value[0]
                if isinstance(first, dict) and ("bbox" in first or "box" in first):
                    return value
        return [detections]
    if isinstance(detections, list):
        return detections
    return []


def _extract_detection_labels(detection):
    if not isinstance(detection, dict):
        return []

    labels = []
    for key in (
        "label",
        "labels",
        "class",
        "classes",
        "class_name",
        "class_names",
        "category",
        "categories",
        "name",
        "names",
    ):
        labels.extend(_normalize_label_values(detection.get(key)))
    return _dedupe_label_values(labels)


def _extract_region_label_filter(output_names):
    marker = "__labels__"
    for output_name in output_names or []:
        text = str(output_name or "")
        if marker not in text:
            continue
        suffix = text.split(marker, 1)[1]
        labels = [item for item in suffix.split("__or__") if item]
        return {_normalize_label_key(label) for label in labels if _normalize_label_key(label)}
    return set()


def _region_target_matches_label_filter(region_target, label_filter):
    if not label_filter or not isinstance(region_target, dict):
        return True
    metadata = region_target.get("metadata") or {}
    labels = _normalize_label_values(metadata.get("labels"))
    labels.extend(_normalize_label_values(metadata.get("label")))
    normalized = {_normalize_label_key(label) for label in labels if _normalize_label_key(label)}
    return bool(normalized & label_filter)


def _normalize_label_values(value):
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        return [str(key).strip() for key, enabled in value.items() if enabled and str(key).strip()]
    if isinstance(value, Iterable):
        return [str(item).strip() for item in value if item is not None and str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _dedupe_label_values(values):
    seen = set()
    deduped = []
    for value in values:
        key = _normalize_label_key(value)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(str(value).strip())
    return deduped


def _normalize_label_key(value):
    text = str(value or "").strip().lower()
    if not text:
        return ""
    return re.sub(r"[^0-9a-z_]+", "_", text)


def _extract_detection_bbox(detection):
    if detection is None:
        return None
    if isinstance(detection, dict):
        bbox = detection.get("bbox", None)
        if bbox is not None:
            return bbox
        alt = detection.get("box", None)
        if alt is not None:
            return alt
    if isinstance(detection, (list, tuple)) and len(detection) == 4:
        return detection
    return None


def _crop_and_resize_region(source_tensor: torch.Tensor, bbox: Sequence[float]) -> torch.Tensor:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]

    tensor = source_tensor
    squeeze = False
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
        squeeze = True

    source_h = int(tensor.shape[-2])
    source_w = int(tensor.shape[-1])

    x1 = min(max(x1, 0), source_w)
    x2 = min(max(x2, 0), source_w)
    y1 = min(max(y1, 0), source_h)
    y2 = min(max(y2, 0), source_h)

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid bbox after clamping; zero-area crop")

    # .clone() to own memory independently of the source tensor so
    # the full-resolution source can be garbage-collected promptly.
    cropped = tensor[..., y1:y2, x1:x2].clone()
    if squeeze:
        cropped = cropped.squeeze(0)
    return cropped


def _round_up_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _extract_structured_frames(raw_frames):
    clean_frames = []
    frame_structured_outputs = []
    frame_errors = []

    if raw_frames is None:
        return clean_frames, frame_structured_outputs, frame_errors

    for frame_index, frame in enumerate(raw_frames):
        if isinstance(frame, Exception):
            frame_errors.append(
                {
                    "frame": frame_index,
                    "error": str(frame),
                    "status": "error",
                }
            )
            continue

        if not isinstance(frame, dict):
            frame_errors.append(
                {
                    "frame": frame_index,
                    "error": f"Unexpected frame payload type: {type(frame).__name__}",
                    "status": "error",
                }
            )
            continue

        frame_outputs = frame.get("_outputs") or []
        for output in frame_outputs:
            output_copy = dict(output)
            output_copy["frame"] = frame.get("frame_index", frame_index)
            frame_structured_outputs.append(output_copy)

        for error in frame.get("_errors") or []:
            error_copy = dict(error)
            error_copy["frame"] = frame.get("frame_index", frame_index)
            frame_errors.append(error_copy)

        clean_frame = {key: value for key, value in frame.items() if not str(key).startswith("_")}
        if "frame_index" not in clean_frame:
            clean_frame["frame_index"] = frame_index
        clean_frames.append(clean_frame)

    return clean_frames, frame_structured_outputs, frame_errors


def _collect_pipeline_metrics(item_future, currently_active_models):
    root_future = getattr(item_future, "root_future", item_future)
    metrics_source = getattr(root_future, "_pipeline_metrics", {}) or {}
    metrics = dict(metrics_source)
    metrics.setdefault("preprocess_seconds", 0.0)
    metrics.setdefault("ai_inference_seconds", 0.0)
    metrics["ai_model_count"] = len(currently_active_models)
    metrics["total_runtime_seconds"] = metrics.get("preprocess_seconds", 0.0) + metrics.get("ai_inference_seconds", 0.0)
    return metrics


def _normalize_requested_model_names(raw_value):
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        raw_value = [raw_value]
    if not isinstance(raw_value, Iterable):
        return []
    normalized = []
    for item in raw_value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _select_used_models(ai_models_info, requested_model_names):
    if not requested_model_names:
        return ai_models_info
    requested = set(requested_model_names)
    selected = []
    for model_info in ai_models_info:
        config_name = getattr(model_info, "config_name", None)
        if config_name in requested or model_info.name in requested:
            selected.append(model_info)
    return selected


def _build_category_capability_index(ai_models_info):
    capability_index = {}
    for model_info in ai_models_info:
        capabilities = list(getattr(model_info, "capabilities", None) or [])
        capability = capabilities[0] if capabilities else "tagging"
        for category in list(getattr(model_info, "categories", None) or []):
            capability_index[str(category)] = capability
    return capability_index


def _normalize_analysis_payload(payload, capability_index):
    capabilities = {
        "tagging": {},
        "detection": {},
        "embedding": {},
        "classification": {},
    }
    region_branches = {}
    other = {}

    if not isinstance(payload, dict):
        return {"raw": payload}

    for key, value in payload.items():
        if key.startswith("regions__") and isinstance(value, list):
            region_branches[key] = [_normalize_region_analysis_item(item, capability_index) for item in value]
            continue

        capability = capability_index.get(key)
        if capability in capabilities:
            capabilities[capability][key] = value
        else:
            other[key] = value

    normalized = {}
    non_empty_capabilities = {key: value for key, value in capabilities.items() if value}
    if non_empty_capabilities:
        normalized["capabilities"] = non_empty_capabilities
    if region_branches:
        normalized["region_branches"] = region_branches
    if other:
        normalized["other"] = other
    return normalized


def _normalize_region_analysis_item(payload, capability_index):
    if not isinstance(payload, dict):
        return {"raw": payload}

    metadata = {}
    if "dynamic_detection_index" in payload:
        metadata["detection_index"] = payload.get("dynamic_detection_index")

    cleaned = {
        key: value
        for key, value in payload.items()
        if key not in {"dynamic_detection_index"} and not str(key).startswith("_")
    }
    normalized = _normalize_analysis_payload(cleaned, capability_index)
    if metadata:
        normalized.update(metadata)
    payload_outputs = list(payload.get("_outputs") or [])
    payload_errors = list(payload.get("_errors") or [])
    if payload_outputs:
        normalized["outputs"] = payload_outputs
    if payload_errors:
        normalized["errors"] = payload_errors
    return normalized


def _is_tag_list(value):
    """Return True if *value* looks like a tagging model output (list of
    strings or (string, float) tuples) suitable for timespan building.
    An empty list is considered a valid (empty) tag list."""
    if not isinstance(value, list):
        return False
    for item in value:
        if isinstance(item, str):
            continue
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str):
            continue
        return False
    return True


def _filter_timespan_compatible_frames(frames):
    filtered_frames = []
    for frame_index, frame in enumerate(frames or []):
        if not isinstance(frame, dict):
            continue

        filtered = {"frame_index": frame.get("frame_index", frame_index)}
        for key, value in frame.items():
            if key == "frame_index":
                continue
            if value and _is_tag_list(value):
                filtered[key] = value

        filtered_frames.append(filtered)

    return filtered_frames


def _extract_per_frame_data(clean_frames):
    """Extract non-tag per-frame data (detections, regions, etc.).

    Returns a list of dicts, each containing ``frame_index`` plus any
    keys whose values are *not* tag lists (i.e. structured data like
    detection dicts or region result lists).  Frames with no such data
    are omitted to keep the payload compact."""
    per_frame = []
    for frame_index, frame in enumerate(clean_frames or []):
        if not isinstance(frame, dict):
            continue

        entry = {"frame_index": frame.get("frame_index", frame_index)}
        for key, value in frame.items():
            if key == "frame_index":
                continue
            # Skip tag lists (handled by timespans), None, and empty values.
            if not value:
                continue
            if _is_tag_list(value):
                continue
            entry[key] = value

        # Only include if there's actual structured data beyond frame_index.
        if len(entry) > 1:
            per_frame.append(entry)

    return per_frame
