from constants.dialog import Role
from core.nlu.nlu import DomainRelavant,Intent,CoarseGrainedIntent,FineGrainedIntent
from core.domain import Task,Slot

from typing import List,Dict,Optional
from pydantic import BaseModel,Field,field_validator,PrivateAttr
from pathlib import Path

from loguru import logger

class Utterance(BaseModel):
    """单句对话"""
    text: str
    role: str
    id: int = Field(default_factory=lambda: Utterance._get_next_id())

    # 定义一个全局计数器,自增
    _id_counter: int = 0

    @classmethod
    def _get_next_id(cls) -> int:
        """获取下一个全局ID"""
        if not isinstance(cls._id_counter, int):
            cls._id_counter = 1
        else:
            cls._id_counter = cls._id_counter + 1
        return cls._id_counter
    
    @field_validator("role")
    @classmethod
    def role_must_valid(cls, v):
        """v的合法取值是Role枚举类中的值(字符串)"""
        try:
            _ = Role(v)
        except Exception as e:
            raise ValueError("role must be a valid value.")
    
    def check_completion(self):
        """检查对话是否完整，如果其所有属性的值都不为None，则认为对话完整"""
        for key, value in self.model_dump().items():
            if value is None:
                raise ValueError(f"{key} is None, the utterance is incomplete.")
    
    def get_text(self):
        return self.text
        
class UserUtterance(Utterance):
    """用户对话"""
    role:str = Role.USER.value
    domain_relavant: Optional[DomainRelavant] = None
    coarse_grained_intents: Optional[List[CoarseGrainedIntent]] = None
    fine_grained_intents: Optional[List[FineGrainedIntent]] = None

    def get_coarse_grained_intents(self):
        return self.coarse_grained_intents

    def check_completion(self):
        """如果domain_relavant属性为false，则不用判断除了text外的属性"""
        if self.domain_relavant is not None and self.domain_relavant.if_relavant == False:
            return
        super().check_completion()

    def update_domain_relavant(self, domain_relavant:DomainRelavant):
        self.domain_relavant = domain_relavant
    
    def update_coarse_grained_intents(self, coarse_grained_intents:List[CoarseGrainedIntent]):
        self.coarse_grained_intents = coarse_grained_intents
    
    def update_fine_grained_intents(self, fine_grained_intents:List[FineGrainedIntent]):
        self.fine_grained_intents = fine_grained_intents
    
    def append_fine_grained_intents(self, fine_grained_intents:List[FineGrainedIntent]):
        if self.fine_grained_intents is None:
            self.fine_grained_intents = []
        self.fine_grained_intents.extend(fine_grained_intents)

class SystemUtterance(Utterance):
    """系统对话"""
    role:str = Role.SYSTEM.value

class ContextData(BaseModel):
    """对话上下文"""
    history: List[Utterance] = []
    ongoing_utterance: Optional[Utterance] = None
    slots: List[Slot] = []
    active_tasks: List[Task] = []

class ContextManager:
    """对话上下文管理器"""
    def __init__(self) -> None:
        self.data: ContextData = ContextData()
    
    def get_history_str(self) -> str:
        return "\n".join([history.model_dump_json(include={'id','text','role'}) for history in self.data.history])
    
    def get_ongoing_utterance(self) -> Optional[Utterance]:
        return self.data.ongoing_utterance
    
    def complete_utterance(self) -> None:

        #检查当前对话附带的信息是否完整
        self.data.ongoing_utterance.check_completion()
        self.data.history.append(self.data.ongoing_utterance)
        self.data.ongoing_utterance = None

    def create_utterance(self,
                         utterance_str:str,
                         utterance_role:Role,
                         intents:Optional[List[Intent]]=None,
                         domain_relavant:Optional[bool]=None) -> None:
        
        #检查上一句话是否提交
        if self.data.ongoing_utterance is not None:
            self.complete_utterance()

        if utterance_role == Role.USER:
            self.data.ongoing_utterance = (
                UserUtterance(
                    text=utterance_str,
                    intents=intents,
                    domain_relavant=domain_relavant
                )
            )
        else :
            self.data.ongoing_utterance = (
                SystemUtterance(
                    text=utterance_str
                )
            )

    def update_slots(self, slots:List[Slot]):
        self.data.slots = slots