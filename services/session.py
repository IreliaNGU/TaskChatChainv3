import argparse
from typing import Dict, List

from core.robot import FinancialRobot

class Session:
    # user_input_list: List[str] = []
    # llm_output_list: List[str] = []
    # bot: FinancialRobot

    def __init__(self, args: argparse.Namespace):
        self.bot = FinancialRobot(args)
        self.user_input_list: List[str] = []
        self.llm_output_list: List[str] = []
        self.slot_dict_list: List[dict] = []
        self.llm_new_output = None
        self.wrong = None
        self.wrong_info = None

    @property
    def latest_message(self):
        return self.user_input_list[-1]

    @property
    def latest_slot_dict(self):
        if len(self.slot_dict_list):
            return self.slot_dict_list[-1]
        return None

    def new_response_arrive(self, agent_response, slot_dict):
        self.llm_output_list.append(agent_response)
        self.slot_dict_list.append(slot_dict)
        self.llm_new_output = agent_response

    def fetch_new_output(self):
        if self.llm_new_output is not None:
            output = self.llm_new_output
            self.llm_new_output = None
            return output
        else:
            return None