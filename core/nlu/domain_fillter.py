#判断对话内容是否与任务相关
import argparse
from loguru import logger
import json

from api.provider import OllamaProvider
from core.domain import DomainManager
from core.dialog import ContextManager
from core.output_parser import DomainFilterParser
from constants.dialog import Role
from template.nlu import DOMAIN_FILTER_TEMPLATE
from utils import read_yml

class DomainFilter:
    def __init__(self,
                args: argparse.Namespace,
                domain_manager: DomainManager,
                context_manager: ContextManager):

        self.client = None
        self.template = DOMAIN_FILTER_TEMPLATE
        self.domain_manager = domain_manager
        self.context_manager = context_manager
        self.scene_description = self.domain_manager.get_scene_description()
        self.parser = DomainFilterParser()

        provider = args.nlu['domain_filter']['provider']
        model = args.nlu['domain_filter']['model']
        api_base = args.nlu['domain_filter']['api_base']
        if provider == "ollama":
            self.client = OllamaProvider(
                model_name=model,
                base_url=api_base)
        else:
            raise NotImplementedError

    def run(self):
        history = self.context_manager.get_history_str()
        text = self.context_manager.get_ongoing_utterance().get_text()
        data = {"text": text, "role": Role.USER.value}
        prompt = self.template.format(history=history, domain=self.scene_description, utterance=json.dumps(data,ensure_ascii=False))
        # logger.info("Domain filter prompt: %s" % prompt)
        result = self.client.process(prompt)
        # logger.info("Domain filter: %s" % json.dumps(result,ensure_ascii=False))
        result_text = result['text']
        parsed = self.parser.parse(result_text)
        self.context_manager.get_ongoing_utterance().update_domain_relavant(parsed['output'])
        return parsed['output']

