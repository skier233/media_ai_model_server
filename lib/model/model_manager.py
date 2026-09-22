
import logging
from lib.async_lib.async_processing import ModelProcessor
from lib.config.config_utils import load_config
from lib.model.ai_model import AIModel
from lib.model.ai_tagging_model import AITaggingModel
from lib.model.ai_face_detection_model import AIFaceDetectionModel
from lib.model.ai_face_embedding_model import AIFaceEmbeddingModel
from lib.model.ai_visual_embedding_model import AIVisualEmbeddingModel
from lib.model.ai_audio_embedding_model import AIAudioEmbeddingModel
from lib.model.ai_audio_classifier_model import AIAudioClassifierModel
from lib.model.python_model import PythonModel
from lib.model.video_preprocessor import VideoPreprocessorModel
from lib.model.image_preprocessor import ImagePreprocessorModel
from lib.model.audio_preprocessor import AudioPreprocessorModel
from lib.model.ai_shot_boundary_model import AIShotBoundaryModel
from lib.utils.vram_budget import compute_batch_sizes


class ModelManager:
    def __init__(self):
        self.models = {}
        self.logger = logging.getLogger("logger")
        self.ai_models = []

    def get_or_create_model(self, modelName):
        if modelName not in self.models:
            self.models[modelName] = self.create_model(modelName)
        return self.models[modelName]

    def get_or_create_model_alias(self, base_model_name, alias):
        """Return a model cached under *alias* but created from *base_model_name*'s config."""
        if alias not in self.models:
            self.models[alias] = self.create_model(base_model_name)
        return self.models[alias]
    
    def get_and_refresh_model(self, modelName):
        if modelName not in self.models:
            self.models[modelName] = self.create_model(modelName)
        else:
            del self.models[modelName]
            self.models[modelName] = self.create_model(modelName)
        return self.models[modelName]
    
    def create_model(self, modelName):
        if not isinstance(modelName, str):
            raise ValueError("Model names must be strings that are the name of the model config file!")
        model_config_path = f"./config/models/{modelName}.yaml"
        try:
            model = self.model_factory(load_config(model_config_path))
        except Exception as e:
            self.logger.error(f"Error loading model {model_config_path}: {e}")
            self.logger.debug("Stack trace:", exc_info=True)
            return None
        return model
    
    def model_factory(self, model_config):
        match model_config["type"]:
            case "video_preprocessor":
                return ModelProcessor(VideoPreprocessorModel(model_config))
            case "image_preprocessor":
                return ModelProcessor(ImagePreprocessorModel(model_config))
            case "model":
                model_processor = ModelProcessor(AITaggingModel(model_config))
                self.ai_models.append(model_processor)
                self._rescale_legacy_batch_sizes()
                return model_processor
            case "face_torch_export":
                role = str(model_config.get("face_model_role", "detection")).lower()
                if role == "embedding":
                    model_processor = ModelProcessor(AIFaceEmbeddingModel(model_config))
                else:
                    model_processor = ModelProcessor(AIFaceDetectionModel(model_config))
                self.ai_models.append(model_processor)
                self._rescale_legacy_batch_sizes()
                return model_processor
            case "python":
                return ModelProcessor(PythonModel(model_config))
            case "visual_embedding":
                model_processor = ModelProcessor(AIVisualEmbeddingModel(model_config))
                self.ai_models.append(model_processor)
                self._rescale_legacy_batch_sizes()
                return model_processor
            case "audio_embedding":
                model_processor = ModelProcessor(AIAudioEmbeddingModel(model_config))
                self.ai_models.append(model_processor)
                self._rescale_legacy_batch_sizes()
                return model_processor
            case "audio_classifier":
                model_processor = ModelProcessor(AIAudioClassifierModel(model_config))
                self.ai_models.append(model_processor)
                self._rescale_legacy_batch_sizes()
                return model_processor
            case "audio_preprocessor":
                return ModelProcessor(AudioPreprocessorModel(model_config))
            case "shot_boundary":
                model_processor = ModelProcessor(AIShotBoundaryModel(model_config))
                self.ai_models.append(model_processor)
                self._rescale_legacy_batch_sizes()
                return model_processor
            case _:
                raise ValueError(f"Model type {model_config['type']} not recognized!")

    def _rescale_legacy_batch_sizes(self):
        """Re-apply legacy batch_size_per_VRAM_GB scaling for all AI models.

        Called after each new AI model is added so that models sharing the GPU
        get progressively smaller batches as more models are loaded — matching
        the original master logic.
        """
        model_count = len(self.ai_models)
        if model_count > 1:
            for mp in self.ai_models:
                mp.model.update_batch_with_mutli_models(model_count)
                mp.update_values_from_child_model(reset_queue=not getattr(mp, "workers_started", False))

    def compute_vram_batch_sizes(self, ai_model_processors=None):
        """Recompute batch sizes for all AI models using VRAM budget allocation.

        Should be called after all models are created but before they are loaded,
        so that weight estimates reflect the full set of active models.
        """
        model_processors = list(ai_model_processors) if ai_model_processors is not None else list(self.ai_models)
        if not model_processors:
            return
        ai_model_instances = [mp.model for mp in model_processors
                              if mp is not None and isinstance(mp.model, AIModel)]
        if not ai_model_instances:
            return
        device = ai_model_instances[0].device

        # Find max_pending_frames from any video preprocessor
        max_pending_frames = 0
        for mp in self.models.values():
            if mp is None:
                continue
            inner = getattr(mp, 'model', None)
            if inner is not None and hasattr(inner, '_max_pending_frames'):
                max_pending_frames = max(max_pending_frames,
                                         inner._max_pending_frames)

        compute_batch_sizes(ai_model_instances, device,
                            max_pending_frames=max_pending_frames)
        # Sync ModelProcessor fields with updated child model values
        for mp in model_processors:
            if mp is not None:
                mp.update_values_from_child_model(reset_queue=not getattr(mp, "workers_started", False))
