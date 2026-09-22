import logging
from lib.async_lib.async_processing import QueueItem
from lib.model.ai_model import AIModel
from lib.model.video_preprocessor import VideoPreprocessorModel
from lib.server.api_definitions import AIModelInfo


logger = logging.getLogger("logger")

class ModelWrapper:
    def __init__(self, model, inputs, outputs):
        self.model = model
        self.inputs = inputs
        self.outputs = outputs

class Pipeline:
    def __init__(self, configValues, model_manager, dynamic_ai_manager, pipeline_name=None):
        if not validate_string_list(configValues["inputs"]):
            raise ValueError("Error: Pipeline inputs must be a non-empty list of strings!")
        if not configValues["output"]:
            raise ValueError("Error: Pipeline output must be a non-empty string!")
        if not isinstance(configValues["models"], list):
            raise ValueError("Error: Pipeline models must be a non-empty list of strings!")
        
        self.short_name = configValues.get("short_name", None)
        if self.short_name is None:
            raise ValueError("Error: Pipeline short_name must be a non-empty string!")
        self.version = configValues.get("version", None)
        if self.version is None:
            raise ValueError("Error: Pipeline version must be a non-empty float!")
        self.inputs = configValues["inputs"]
        self.output = configValues["output"]


        self.models = []
        self._input_to_models = {}  # key -> list[ModelWrapper] index for O(1) event_handler lookup
        for model in configValues["models"]:
            if not validate_string_list(model["inputs"]):
                raise ValueError("Error: Model inputs must be a non-empty list of strings!")
            if not model["name"]:
                raise ValueError("Error: Model name must be a non-empty string!")
            modelName = model["name"]
            if modelName in ["dynamic_video_ai", "dynamic_image_ai", "dynamic_region_ai", "dynamic_audio_ai", "dynamic_asset_ai", "dynamic_ai"]:
                if modelName == "dynamic_video_ai":
                    dynamic_models = dynamic_ai_manager.get_dynamic_video_ai_models(model["inputs"], model["outputs"], pipeline_name=pipeline_name)
                elif modelName == "dynamic_asset_ai":
                    dynamic_models = dynamic_ai_manager.get_dynamic_asset_ai_models(model["inputs"], model["outputs"], pipeline_name=pipeline_name)
                elif modelName == "dynamic_image_ai":
                    dynamic_models = dynamic_ai_manager.get_dynamic_image_ai_models(model["inputs"], model["outputs"], pipeline_name=pipeline_name)
                elif modelName == "dynamic_region_ai":
                    dynamic_models = dynamic_ai_manager.get_dynamic_region_ai_models(model["inputs"], model["outputs"], pipeline_name=pipeline_name)
                elif modelName == "dynamic_audio_ai":
                    dynamic_models = dynamic_ai_manager.get_dynamic_audio_ai_models(model["inputs"], model["outputs"], pipeline_name=pipeline_name)
                else:
                    dynamic_models = dynamic_ai_manager.get_dynamic_models_from_config(model, pipeline_name=pipeline_name)
                self.models.extend(dynamic_models)
                continue
            returned_model = model_manager.get_or_create_model(modelName)
            self.models.append(ModelWrapper(returned_model, model["inputs"], model["outputs"]))

        self._rebuild_input_index()

        categories_set = set()
        for model in self.models:
            if isinstance(model.model.model, AIModel):
                for category in model.model.model.model_category:
                    if category in categories_set:
                        logger.warning(
                            f"Duplicate AI model category '{category}' detected in pipeline '{self.short_name}'. Allowing overlap for multi-stage flows."
                        )
                    categories_set.add(category)
    
    def _rebuild_input_index(self):
        """Build a dict mapping each input key to the ModelWrappers that need it."""
        self._input_to_models = {}
        for model in self.models:
            for input_name in model.inputs:
                if input_name not in self._input_to_models:
                    self._input_to_models[input_name] = []
                self._input_to_models[input_name].append(model)

    async def event_handler(self, itemFuture, key):
        if key == self.output:
            itemFuture.close_future(itemFuture[key])
        matching_models = self._input_to_models.get(key)
        if matching_models is None:
            return
        for model in matching_models:
            allOtherInputsPresent = all(inputName in itemFuture.data for inputName in model.inputs if inputName != key)
            if allOtherInputsPresent:
                await model.model.add_to_queue(QueueItem(itemFuture, model.inputs, model.outputs))

    async def start_model_processing(self):
        for model in self.models:
            await model.model.start_workers()

    async def stop_model_processing(self):
        seen_processors = set()
        for model in self.models:
            processor = model.model
            if processor is None:
                continue
            processor_id = id(processor)
            if processor_id in seen_processors:
                continue
            seen_processors.add(processor_id)
            await processor.stop_workers()

    def get_first_video_preprocessor(self):
        for model in self.models:
            if isinstance(model.model.model, VideoPreprocessorModel):
                return model.model.model
        return None
    
    def get_first_ai_model(self):
        for model in self.models:
            if isinstance(model.model.model, AIModel):
                return model.model.model
        return None
    
    def get_ai_models_info(self):
        ai_models_info = []
        for model in self.models:
            if isinstance(model.model.model, AIModel):
                ai_models_info.append(
                    AIModelInfo(
                        name=model.model.model.model_file_name,
                        config_name=getattr(model.model.model, "config_name", None),
                        identifier=model.model.model.model_identifier,
                        version=model.model.model.model_version,
                        categories=model.model.model.model_category,
                        type=model.model.model.model_type,
                        capabilities=model.model.model.model_capabilities,
                        supported_scopes=model.model.model.supported_target_scopes,
                    )
                )
        return ai_models_info

def validate_string_list(input_list):
    if not isinstance(input_list, list):
        return False
    for item in input_list:
        if not isinstance(item, str):
            return False
    if len(input_list) == 0:
        return False
    return True
    