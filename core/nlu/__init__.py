import argparse
from .domain_fillter import DomainFilter
from .coarse_grained_nlu import CoarseGrainedNlu
from .fine_grained_nlu import FineGrainedNlu
from loguru import logger
from core.domain import DomainManager
from core.dialog import ContextManager
from utils import read_yml

class NluPipeline:
    def __init__(self, 
                args: argparse.Namespace,
                domain_manager: DomainManager,
                context_manager: ContextManager):
        self.domain_filter = DomainFilter(args, domain_manager, context_manager)
        self.context_manager = context_manager
        self.coarse_grained_nlu = CoarseGrainedNlu(args, domain_manager, context_manager)
        self.fine_grained_nlu = FineGrainedNlu(args, domain_manager, context_manager)
    
    
    def run(self):
        x1 = self.domain_filter.run()
        if  x1.if_relavant:
            self.coarse_grained_nlu.run()
            self.fine_grained_nlu.run()

        utterance = self.context_manager.get_ongoing_utterance()
        logger.info("NLU pipeline result: %s" % utterance.model_dump_json())
        
        return