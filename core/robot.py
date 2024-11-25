import argparse

from loguru import logger

from core.nlu import NluPipeline
from core.domain import DomainManager
from core.dialog import ContextManager
from constants.dialog import Role

class FinancialRobot:

    def __init__(self, args: argparse.Namespace):
        self.context_manager = ContextManager()
        self.domain_manager = DomainManager(args.domain_path)
        self.nlu_pipeline = NluPipeline(args, self.domain_manager, self.context_manager)


    def run_text(self, text):
        
        self.context_manager.create_utterance(text, Role.USER)
        self.nlu_pipeline.run()
        self.context_manager.complete_utterance()

        #生成系统对话
        chat_output = "你好"

        #维护系统对话
        self.context_manager.create_utterance(chat_output, Role.SYSTEM)
        #维护系统对话的状态
        pass
        self.context_manager.complete_utterance()

        return chat_output,{}
