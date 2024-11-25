#粗粒度的NLU
import argparse
from loguru import logger
import json

from api.provider import OllamaProvider
from core.domain import DomainManager
from core.dialog import ContextManager
from core.output_parser import CoarseGrainedNluParser
from core.nlu.nlu import CoarseGrainedIntent
from constants.dialog import Role
from template.nlu import COARSE_GRAINED_NLU_TEMPLATE
from utils import read_yml

class CoarseGrainedNlu:
    def __init__(self,
                args: argparse.Namespace,
                domain_manager: DomainManager,
                context_manager: ContextManager):

        self.client = None
        self.template = COARSE_GRAINED_NLU_TEMPLATE
        self.domain_manager = domain_manager
        self.context_manager = context_manager
        
        self.load_coarse_grained_intents(args.nlu_config)
        self.parser = CoarseGrainedNluParser(self.intents)

        provider = args.nlu['domain_filter']['provider']
        model = args.nlu['domain_filter']['model']
        api_base = args.nlu['domain_filter']['api_base']
        if provider == "ollama":
            self.client = OllamaProvider(
                model_name=model,
                base_url=api_base)
        else:
            raise NotImplementedError("Provider %s not implemented" % provider)

    def load_coarse_grained_intents(self,config_path):
        intents = []
        intents_schema = read_yml(config_path)['coarse_grained_intents']
        for intent in intents_schema:
            x = CoarseGrainedIntent(id=intent['id'],description=intent['description'])
            intents.append(x)
        self.intents = intents
        
    def run(self):
        history = self.context_manager.get_history_str()
        text = self.context_manager.get_ongoing_utterance().get_text()
        data = {"text": text, "role": Role.USER.value}
        coarse_grained_intents_str = "\n".join([intent.model_dump_json(include={'id','description'}) for intent in self.intents])

        prompt = self.template.format(history=history, coarse_grained_intents=coarse_grained_intents_str, utterance=json.dumps(data,ensure_ascii=False))
        # logger.info("Coarse grained NLU prompt: %s" % prompt)
        result = self.client.process(prompt)
        logger.info("Coarse grained NLU: %s" % json.dumps(result))
        result_text = result['text']
        parsed = self.parser.parse(result_text)
        # logger.info("Coarse grained NLU parsed: \n")
        # for i in parsed['output']:
        #     logger.info("%s\n" % i.model_dump_json(include={'id','description'}))
        self.context_manager.get_ongoing_utterance().update_coarse_grained_intents(parsed['output'])

