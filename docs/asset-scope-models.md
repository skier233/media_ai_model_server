# Asset-scope models

Most models in this server run **per frame**: the shared video preprocessor
decodes at `frame_interval`, fans each frame out into its own `ItemFuture`, and
every active model scores frames independently. Some models cannot work that
way — they need dense, *adjacent* frames across the whole asset, at their own
resolution. Shot-boundary detection is the motivating example.

`asset` already names the scope of whole-image and whole-audio analysis, but a
video had no way to run a model over the whole asset with its own decode. The
`dynamic_asset_ai` stage fills that gap. Which models belong there is decided by
model type, not by scope: a `shot_boundary` model (see below) analyses the whole
video itself, and it is the only kind of model the stage runs. A model that
merely lists `asset` among its scopes keeps its place in the image, audio and
frame stages.

## How the stage works

`config/pipelines/video_pipeline_dynamic_v4.yaml` runs two stages in parallel
off the same request:

```yaml
  - name: dynamic_video_ai          # per-frame fan-out
    outputs: [childrenResults]
  - name: dynamic_asset_ai          # whole-asset models
    inputs:  [video_path, vr_video, skipped_categories, requested_model_names]
    outputs: [assetResults]
  - name: video_result_postprocessor_v4
    inputs:  [childrenResults, video_path, time_interval, assetResults]
```

The pipeline is a data-flow DAG, so the postprocessor simply waits until both
stages have produced their key.

An asset-scope model receives the stage's declared inputs directly (the asset
path first) and returns one result for the whole asset. There is no
preprocessor and no child fan-out. `asset_result_collector` gathers the models'
outputs into `assetResults`; it is wired on the asset path alone, so the stage
still completes — with an empty result — on the common deployment that has **no
asset-scope model installed at all**.

Selection follows the same rules as every other stage:

| Request | Effect |
|---|---|
| `want: [{capability: "temporal_segmentation"}]` | runs asset models with that capability |
| `want: [{category: "shot_boundaries"}]` | runs models declaring that category |
| `want: [{models: ["<config name>"]}]` | runs exactly that model |
| `categories_to_skip: ["shot_boundaries"]` | suppresses it |
| a request naming only frame-scope models | suppresses it |
| a request naming no models (legacy `/process_video/`, `/v3/*`, `/v4` without `want`) | suppresses it |

An asset-scope model decodes every frame of the whole video, so it runs only
when a request asks for it. A request that names no models means "the usual
per-frame analysis" and never pays for it. On a video, a `want` by capability or
category matches the per-frame models and the whole-video models, and nothing
else; image and audio requests keep their default scope.

A model that will not run for a request is skipped when the request arrives,
not when a worker reaches it, so a request that does not want a busy asset model
never waits behind the analyses queued for it. Each skip is reported in
`asset_outputs` with a `reason`: `not_named`, `not_requested` or
`category_skipped`. An analysis that fails is reported in `asset_errors`; the
request itself still succeeds with its per-frame results. An analysis whose
request has already finished, for example because it timed out, is not
started, and the video path is normalised and checked exactly as for the
per-frame decode.

Asking for *only* asset-scope models also skips the frame decode entirely,
rather than decoding the video to build children that all resolve to `Skip()`.

`config/model_capabilities.yaml` selects which models the stage may use:

```yaml
video_pipeline_dynamic_v4:
  asset_models: ALL     # every active whole-video (shot_boundary) model
```

A pipeline entry without `asset_models` means `ALL`, and the stage never wires
anything but whole-video models, whatever the entry lists. `ALL` honours an
`exclude` list (`asset_models: {mode: ALL, exclude: [<config name>]}`). An
explicit list loses the models that are not active, when the server starts and
whenever models are loaded or unloaded; a list that ends up empty stays empty
rather than falling back to `ALL`, so the stage then runs nothing. Custom
pipelines registered with `POST /v4/pipelines/custom` get the stage too, with
`use_all_asset_models` defaulting to true; a `want` that names a model the
target pipeline does not wire is ignored.

Custom pipelines written by an older server gain the stage from the schema 4.1
migration in `migrate.py`, which the install scripts run; an update through
`update.sh` or `update.ps1` does not run it, so run `python migrate.py` after
updating. Skipping it is harmless: a pipeline without the stage keeps working
exactly as before and simply never produces asset-scope output, and a
`model_capabilities.yaml` entry without `asset_models` already means `ALL`.

## Response

Asset-scope output is emitted as a top-level `analysis` node with exactly the
shape the **image** pipeline already uses for its asset-level analysis, so a
client needs one code path for both. Known capabilities bucket under
`analysis.capabilities`; anything else — temporal segmentation included — under
`analysis.other`, keyed by category.

The key is absent when the stage did not run, so a caller can tell "not
requested" from "ran and found nothing".

```jsonc
{
  "frames": [ /* unchanged per-frame results */ ],
  "analysis": {
    "other": {
      "shot_boundaries": {
        "schema_version": 2,
        "model": "omnishotcut_shot_boundaries",
        "model_version": "1.0",
        "mode": "clean_shot",
        "fps": 30.0,
        "duration_seconds": 60.0,
        "source_frame_count": 1800,
        "duration_mismatch_seconds": 0.0,
        "window_frames": 100,
        "context_frames": 0,
        "decode_backend": "ffmpeg_cpu",
        "boundaries": [
          { "start_frame": 0,  "end_frame": 60,   "start_seconds": 0.0,   "end_seconds": 2.0,
            "shot_type": "General",  "transition_in": null,                "filled": false },
          { "start_frame": 60, "end_frame": 76,   "start_seconds": 2.0,   "end_seconds": 2.533,
            "shot_type": "Dissolve", "transition_in": "Transition_Source", "filled": false },
          { "start_frame": 76, "end_frame": 1800, "start_seconds": 2.533, "end_seconds": 60.0,
            "shot_type": "General",  "transition_in": "Transition",        "filled": false }
        ],
        "label_counts": { "shot_type": {...}, "transition_in": {...} },
        "shots": [
          { "start_frame": 0,  "end_frame": 60,   "intra": "General", "inter": "New_Start" },
          { "start_frame": 76, "end_frame": 1800, "intra": "General", "inter": "Transition" }
        ]
      }
    }
  },
  "asset_outputs": [ { "model_step": "shot_boundaries", "status": "ok" } ]
}
```

The node also carries timing diagnostics such as `inference_seconds`; they are
not part of the contract.

`boundaries` is a **contiguous, gapless partition of the whole asset** that a
consumer can store as it is. `tests/test_shot_boundary_contract.py` pins every
guarantee:

* **Frames are authoritative.** Each element covers the half-open decoded-frame
  range `[start_frame, end_frame)`. The first starts at 0, each starts where the
  previous ended, and the last ends at `source_frame_count`.
* **Seconds** are `frame / fps` rounded to milliseconds. The first is `0.0`, each
  starts where the previous ended, and the last is `duration_seconds`. An
  element with no length left in seconds is absorbed by its neighbour: one
  shorter than the rounding, or, when `duration_seconds` is shorter than the
  decoded frames last, one that starts at or after it. Its frames stay covered.
  `duration_mismatch_seconds` is `duration_seconds` minus the decoded frames'
  own length, so a large value explains a stretched last shot (positive) or
  absorbed tail shots (negative). When the container declares no frame rate,
  `fps` is measured as the decoded frame count over the duration.
* **`shot_type`** is the model's shot type, its intra label (`General`,
  `Dissolve`, `Fade`, …).
* **`transition_in`** is how the element is entered from the one before it, the
  inter label (`Hard_Cut`, `Transition`, `Sudden_Jump`, …). The first element
  has no incoming cut, so it is always `null`. Inside a video, `New_Start` means
  the model saw no preceding frames: a shot that begins at a window seam and
  could not be joined to the one before it. It may not be a cut at all.
* **`filled: true`** marks frames that no model shot covered, with both labels
  `null`. Each window is placed at its own first frame, so this appears where a
  window's shots stop short of its end, and nowhere else.

`boundaries` always includes transitions as their own labelled elements, in
every `mode`. `mode` only filters `shots`, the model's own frame-accurate shot
list: `clean_shot` lists the ordinary (`General`) shots, `default` lists them
all. `schema_version` changes whenever this shape does. Version 1 only ever came
from pre-release builds; it had no frames and a misnamed `transition_after`, so
a consumer should reject it.

# The `shot_boundary` model type

The artifact is a self-contained `.pt2` (`torch.export`) or `.pt` (TorchScript)
module under `./models/`, exactly like every other model here — the server
contains no model-specific code. It must be unencrypted: the model runs in
float32, which the licensed runner cannot be asked for, so an encrypted
artifact is listed as incompatible in the catalog, cannot be activated, and is
refused at load. The model yaml gives exactly one `model_category`, the key its
result is emitted under.

**Signature**

```
forward(clip: [1, T, 3, H, W]) -> (shot_logits, intra_logits, inter_logits)
```

* `clip` is `T` consecutive frames, ImageNet-normalised, at the model's trained
  geometry `H x W`. The geometry comes from the yaml's `preprocess_config`
  `width`/`height` or, when those are left out, from the sidecar's
  `frame_width`/`frame_height`; when both are given they must agree. There is
  no default size, and `normalization` may only be `0` (ImageNet), which is
  also its default.
* `shot_logits` is `[1, Q, T + 2]`: per query, a distribution over the
  clip-relative end frame. The final class is a "no object" slot and is dropped.
* `intra_logits` / `inter_logits` are `[1, Q, C]`: per query, the shot-type and
  transition-type class distributions, again with a trailing "no object" slot.

All three are dense and fixed-shape, which is what keeps the artifact
exportable. Everything else — windowing, the greedy query walk, context
pruning, cross-window merging, `clean_shot` filtering and boundary
normalization — is generic and lives in
`lib/model/ai_shot_boundary_model.py`.

**Label sidecar** — `./models/<model_file_name>.labels.json`, alongside the
artifact (mirroring how tagging models read `<model_file_name>.tags.txt`):

```json
{
  "window_frames": 100,
  "frame_height": 96,
  "frame_width": 128,
  "num_queries": 24,
  "intra": { "0": "General", "1": "Dissolve" },
  "inter": { "0": "New_Start", "1": "Hard_Cut" },
  "intra_name_to_index": { "general": 0 },
  "inter_name_to_index": { "new_start": 0 }
}
```

`intra_name_to_index` / `inter_name_to_index` let the generic code find the two
labels it must reason about without hard-coding them: the "general" shot type
used by `clean_shot`, and the "new start" transition used to join a shot that
continues across a window seam. A model without its sidecar cannot be
activated, and one whose sidecar lacks the "new start" label is refused at load,
since every shot crossing a window seam would otherwise be split in two. The
sidecar and the settings that depend on it are checked before the artifact is
loaded.

**Model yaml** (`config/models/<name>.yaml`, supplied by the user like all
model configs — this directory is gitignored):

```yaml
type: shot_boundary
model_file_name: omnishotcut_shot_boundaries
model_category: [shot_boundaries]
model_capabilities: [temporal_segmentation]
supported_target_scopes: [asset]
full_image_model: false
model_type: ShotBoundary
model_version: 1.0
model_image_size: null        # asset models declare no shared-preprocessor geometry
model_precision: float32
max_batch_size: 1
instance_count: 1

mode: clean_shot              # which shots `shots` lists; "default" lists transitions too
context_frames: 0             # frames of overlap between windows
preprocess_config:
  width: 128
  height: 96
  normalization: 0            # ImageNet
  device: cpu
  precision: float32
```

`mode` must be `default` or `clean_shot`. Optional keys: `window_frames`
(defaults to the sidecar's value) and `general_intra_label` /
`new_start_inter_label` (default `general` / `new_start`). Frame-to-time
conversion is always `index / fps`.

An optional `decode:` block selects how frames are produced. Every key may be
omitted; the defaults are what the model runs with when the block is absent:

```yaml
decode:
  backend: auto        # auto | ffmpeg_cpu | ffmpeg_cuda | av
  quality: exact       # exact | fast
  device_index: null   # CUDA device for ffmpeg_cuda; null = default device
```

`auto` tries `ffmpeg_cpu` and falls back to `av`; it never selects NVDEC, for
the reasons under [Decoding](#decoding) below — set `backend: ffmpeg_cuda`
explicitly to use it. Naming a backend is a preference, not a pin: if it fails
its pre-flight checks the ladder still falls through to the next one, and the
backend actually used is reported as `decode_backend` in the result.

`quality: fast` replaces the GPU scaling pyramid with a single `scale_cuda`
step. It measured no faster and it aliases, so it exists only for diagnosis;
leave it at `exact`.

## Decoding

An asset-scope model that needs every frame is decode-bound, not compute-bound:
a 4K source costs far more to decode than to score. `ffmpeg_pipe.py` runs one
ffmpeg and reads raw RGB frames from its stdout, which is substantially faster
than decoding frame-by-frame through PyAV.

Measured on a 32-core host with an RTX 4090, against a 3840x2160 60fps H.264
source scaled to 128x96, and compared against the PyAV path the model was
validated on:

| backend | throughput | max \|delta\| vs PyAV |
|---|---|---|
| `ffmpeg_cpu` (default) | 417 fps | 3 |
| `ffmpeg_cuda` (NVDEC) | 275 fps | 40 |
| `av` (PyAV) | 108 fps | - |

Two results worth knowing before reaching for the GPU:

* **NVDEC is not automatically faster**, and which one wins is a property of
  the host. Same host and card, a heavier 4096x2160 60fps H.264 source, 3000
  frames, with the decode pinned to a varying number of cores via `taskset`
  (absolute numbers move with the file; the shape does not):

  | cores | `ffmpeg_cpu` | `ffmpeg_cuda` |
  |---|---|---|
  | 2 | 91 fps | 247 fps |
  | 4 | 165 fps | 248 fps |
  | 8 | 260 fps | 248 fps |
  | 16 | 313 fps | 247 fps |
  | 32 | 324 fps | 247 fps |

  NVDEC is flat: the work sits on the card's decoder engine, so host cores buy
  nothing. Software decode scales, steeply at first and then flattening as it
  approaches its single-stream limit. They cross at roughly **8 cores**. Pin
  `ffmpeg_cuda` below that; above it, prefer the default. The other reason to pin
  it is that its 247 fps costs almost no CPU, which matters when analysis runs
  alongside transcodes or scans rather than on an idle box.
* **NVDEC's frames differ.** `scale_cuda` uses a fixed 4x4 kernel with no
  ratio-adaptive widening, unlike swscale's bicubic, so its output is not
  interchangeable with the software path and can move a borderline boundary.

For that reason `backend: auto` never selects NVDEC; it must be requested
explicitly. When it is, the descent to the target size is done as a sequence of
GPU halvings followed by one small swscale step, so no single step exceeds 2x and
the final kernel matches the software path. That pyramid is free — it measured
274.5 fps against 275.8 fps for a single `scale_cuda`.

Frame count must match a software decode exactly, because consumers map frame
index to time as `index / fps`. The pipeline therefore uses `-fps_mode
passthrough` and a `select` frame modulo, never the `fps` filter, which
resamples to a wall-clock rate. A short read is raised as an error rather than
treated as a short video: a truncated decode yields results that look entirely
plausible and are wrong. For the same reason a decode that produces no frames
at all is an error, not a video without shots.

Backends are tried in order and every check — binary features, codec and pixel
format, and a two-frame smoke decode with the real command — runs before the
first frame is produced, so a failure falls through to the next backend rather
than surfacing mid-stream.

## Notes for implementers

* Decoding happens at the exact clip geometry without preserving aspect ratio,
  reproducing an `ffmpeg -s WxH -pix_fmt rgb24` pipeline — this matters when
  matching a reference implementation's output. The conversion to RGB is left to
  the output pixel format rather than a separate `format` filter, so scaling and
  colour conversion happen in one swscale pass, as PyAV's `reformat` does.
* Frames are decoded as stored (`-noautorotate`). PyAV ignores a display
  rotation too, so every backend hands the model the same pixels.
* Only one window of frames is held at a time, so peak memory is
  `window_frames x 3 x H x W` regardless of video length. With `vr_video`,
  frames are decoded at full resolution and one eye is cropped in Python, so
  `H x W` is one eye at the source's resolution: a window of float32 crops of a
  6144x3072 source takes about 11 GB.
* Decode and inference run in an executor. They take minutes, and running them
  inline would block the event loop and stall every other model — including the
  tagging pass over the same video.
* Inference runs in float32. A transformer with sinusoidal position encodings
  overflows in fp16.
