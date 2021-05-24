import os
import typing
from typing import Any, Dict, List, Optional, Text, Type

import rasa.shared.utils.io
from fuzzywuzzy import process
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.extractors.extractor import EntityExtractor
from rasa.nlu.utils import write_json_to_file
from rasa.shared.nlu.constants import ENTITIES
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata

# subclass EntityExtractor to skip featurize_message() in rasa.nlu.model.Interpreter
class EntityTypoFixer(EntityExtractor):
    """A new component"""

    # Which components are required by this component.
    # Listed components should appear before the component itself in the pipeline.
    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        """Specify which components need to be present in the pipeline."""

        return [EntityExtractor]

    # Defines the default configuration parameters of a component
    # these values can be overwritten in the pipeline configuration
    # of the model. The component should choose sensible defaults
    # and should be able to create reasonable results with the defaults.
    defaults = {"score_cutoff": 80}

    # Defines what language(s) this component can handle.
    # This attribute is designed for instance method: `can_handle_language`.
    # Default value is None which means it can handle all languages.
    # This is an important feature for backwards compatibility of components.
    supported_language_list = None

    # Defines what language(s) this component can NOT handle.
    # This attribute is designed for instance method: `can_handle_language`.
    # Default value is None which means it can handle all languages.
    # This is an important feature for backwards compatibility of components.
    not_supported_language_list = None

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        entities: Optional[List[Text]] = None,
    ) -> None:
        super().__init__(component_config)
        self.entities = entities if entities else []
        self.score_cutoff = component_config.get(
            "score_cutoff", self.defaults["score_cutoff"]
        )

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Train this component.

        This is the components chance to train itself provided
        with the training data. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`components.Component.pipeline_init`
        of ANY component and
        on any context attributes created by a call to
        :meth:`components.Component.train`
        of components previous to this one."""
        self.entities = list(training_data.entity_synonyms.keys())

    def process(self, message: Message, **kwargs: Any) -> None:
        """Process an incoming message.

        This is the components chance to process an incoming
        message. The component can rely on
        any context attribute to be present, that gets created
        by a call to :meth:`components.Component.pipeline_init`
        of ANY component and
        on any context attributes created by a call to
        :meth:`components.Component.process`
        of components previous to this one."""
        extracted_entities = message.get(ENTITIES, [])
        self.fix_entity_typo(extracted_entities)
        message.set(ENTITIES, extracted_entities, add_to_output=True)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this component to disk for future loading."""

        if self.entities:
            file_name = file_name + ".json"
            entity_file = os.path.join(model_dir, file_name)
            write_json_to_file(entity_file, self.entities)

            return {"file": file_name}
        else:
            return {"file": None}

    def fix_entity_typo(self, entities: List[Dict[Text, Any]]) -> None:
        for entity in entities:
            entity_value = str(entity["value"])
            fuzzy_match = process.extractOne(
                entity_value, self.entities, score_cutoff=self.score_cutoff
            )
            if fuzzy_match:
                fuzzy_entity = fuzzy_match[0]
                if fuzzy_entity != entity_value:
                    entity["value"] = fuzzy_entity
                    self.add_processor_name(entity)

    def add_processor_name(self, entity: Dict[Text, Any]) -> Dict[Text, Any]:
        if "processors" in entity:
            entity["processors"].append(self.name)
        else:
            entity["processors"] = [self.name]

        return entity

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text,
        model_metadata: Optional["Metadata"] = None,
        cached_component: Optional["Component"] = None,
        **kwargs: Any,
    ) -> "Component":
        """Load this component from file."""

        file_name = meta.get("file")
        if not file_name:
            entities = None
            return cls(meta, entities)

        entities_file = os.path.join(model_dir, file_name)
        if os.path.isfile(entities_file):
            entities = rasa.shared.utils.io.read_json_file(entities_file)
        else:
            entities = None
        return cls(meta, entities)
