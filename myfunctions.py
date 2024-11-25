from core.function import Function,Tracker
from loguru import logger


class GenerateInvestmentPlan(Function):
    def __init__(self, args):
        super().__init__(args)

    def run(self, 
            tracker: Tracker) -> None:
        logger.info("Executing GenerateInvestmentPlan")