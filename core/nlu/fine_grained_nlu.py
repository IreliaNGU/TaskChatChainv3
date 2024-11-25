#粗粒度的NLU
import argparse
from loguru import logger
import json

from api.provider import OllamaProvider
from core.domain import DomainManager
from core.dialog import ContextManager
from core.output_parser import FineGrainedNluParser
from core.nlu.nlu import FineGrainedIntent
from constants.dialog import Role
from constants.nlu import COARSE_GRAINED_INTENT as Coa, FINE_GRAINED_INTENT as Fin
from template.nlu import FINE_GRAINED_NLU_ACTION_REQUESTING_TEMPLATE,FINE_GRAINED_NLU_INFO_PROVIDING_TEMPLATE,FINE_GRAINED_NLU_INFO_REQUIRING_TEMPLATE
from utils import read_yml

class FineGrainedNlu:
    def __init__(self,
                args: argparse.Namespace,
                domain_manager: DomainManager,
                context_manager: ContextManager):

        self.client = None
        self.template = {
            Coa.ACTION_REQUESTING.value: FINE_GRAINED_NLU_ACTION_REQUESTING_TEMPLATE,
            Coa.INFO_PROVIDING.value: FINE_GRAINED_NLU_INFO_PROVIDING_TEMPLATE,
            Coa.INFO_REQUIRING.value: FINE_GRAINED_NLU_INFO_REQUIRING_TEMPLATE
        }
        self.domain_manager = domain_manager
        self.context_manager = context_manager
        self.coarse_fine_mapping = self.load_coarse_fine_intent_mapping(args.nlu_config)
        self.parser = FineGrainedNluParser(self.coarse_fine_mapping, self.domain_manager.get_slots(), self.domain_manager.get_tasks())

        provider = args.nlu['domain_filter']['provider']
        model = args.nlu['domain_filter']['model']
        api_base = args.nlu['domain_filter']['api_base']
        if provider == "ollama":
            self.client = OllamaProvider(
                model_name=model,
                base_url=api_base)
        else:
            raise NotImplementedError("Provider %s not implemented" % provider)

    def load_coarse_fine_intent_mapping(self, config_path):
        """ Load the mapping from coarse grained intents to fine grained intents"""

        fine_grained_intents = read_yml(config_path)['fine_grained_intents']
        coarse_fine_intent_mapping = {}
        for fine_intent in fine_grained_intents:
            coarse_intent = Coa(fine_intent['parent_id'])
            if coarse_intent not in coarse_fine_intent_mapping:
                coarse_fine_intent_mapping[coarse_intent] = []

            schema_list = []
            if 'schema' in fine_intent:
                for schema in fine_intent['schema']:
                    schema_list.append(json.dumps(schema,ensure_ascii=False))
            coarse_fine_intent_mapping[coarse_intent].append(FineGrainedIntent(id=fine_intent['id'],
                                                                               description=fine_intent['description'],
                                                                               schema=schema_list))
        return coarse_fine_intent_mapping
        
    def run(self):
        history = self.context_manager.get_history_str()
        text = self.context_manager.get_ongoing_utterance().get_text()
        data = {"text": text, "role": Role.USER.value}

        coarse_intents = self.context_manager.get_ongoing_utterance().get_coarse_grained_intents()
        for coa_intent in coarse_intents:
            fine_intents = self.coarse_fine_mapping[coa_intent.to_enum_type()]
            fine_grained_intents_str = "\n".join([intent.model_dump_json(include={'id','description','schema'}) for intent in fine_intents])
            template =  self.template[coa_intent.id]
            if coa_intent.id == Coa.INFO_REQUIRING.value:
                data_sources = fine_grained_intents_str
                database_schema = self.domain_manager.get_database_schema()
                database_schema = "\n".join([db_sc.model_dump_json(include={'table_name','description','columns'}) for db_sc in database_schema])
                prompt = template.format(history=history, utterance=json.dumps(data,ensure_ascii=False), data_sources=data_sources, database_schema=database_schema)
            elif coa_intent.id == Coa.INFO_PROVIDING.value:
                slots = self.domain_manager.get_slots()
                slots_str = "\n".join([slot.model_dump_json(include={'name','description'}) for slot in slots])
                prompt = template.format(history=history, utterance=json.dumps(data,ensure_ascii=False), fine_grained_intents_info_providing=fine_grained_intents_str, slots=slots_str)
            elif coa_intent.id == Coa.ACTION_REQUESTING.value:
                tasks = self.domain_manager.get_tasks()
                tasks_str = "\n".join([task.model_dump_json(include={'name','description','activate'}) for task in tasks])
                prompt = template.format(history=history, utterance=json.dumps(data,ensure_ascii=False), find_grained_intents_action_requesting=fine_grained_intents_str, tasks = tasks_str)
            else:
                raise NotImplementedError("Fine-grained Template for coarse grained intent %s not implemented" % coa_intent.id)
            result = self.client.process(prompt)['text']
            # logger.info("Fine grained NLU: %s" % result)
            parsed = self.parser.parse(result,coa_intent.to_enum_type())
            self.context_manager.get_ongoing_utterance().append_fine_grained_intents(parsed['output'])