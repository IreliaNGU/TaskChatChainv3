# Description: Function model，用户自定义的函数会继承这个类
from loguru import logger
from typing import Dict,List
from core.nlu.nlu import Slot
from core.dialog import ContextDict

class Tracker:
    """Maintains the context of a conversation."""

    @classmethod
    def from_dict(cls, state: "ContextDict") -> "Tracker":
        """Create a tracker from dump."""
        return Tracker(
            state.get("slots", []),
            state.get("history", []),
        )
    
    def __init__(self,
                 slots: List[Slot],
                 history: List[ContextDict]) -> None:
        self.slots = slots
        self.history = history
    

class Function:

    def name(self) -> str:
        raise NotImplementedError("A function must implement a name method")
    
    def run(self, 
            tracker: Tracker) -> str:
        raise NotImplementedError("A function must implement a run method")