#define the domain model and load the domain.yml file 
from loguru import logger
from typing import List, Dict
from typing_extensions import TypedDict
from utils import read_yml

import yaml
from pydantic import BaseModel, Field, validator, ValidationError
from typing import List, Optional, Union

# 定义数据库列模型
class Column(BaseModel):
    name: str
    type: str
    description: Optional[str]

# 定义数据库表模型
class TableSchema(BaseModel):
    table_name: str
    description: Optional[str]
    columns: List[Column]

# 定义功能调用模型
class FunctionCalling(BaseModel):
    name: str
    description: Optional[str]

# 定义槽位模型
class Slot(BaseModel):
    name: str
    type: str #categorical, numerical, text
    description: str
    valid_values: Optional[List[str]] = None # 槽位的合法取值,只有当type为numerical时，该字段才有意义
    reset_after_task: Optional[List[str]] = None # 任务完成后是否重置槽位
    check: Optional[str] = None # 检查槽位值是否合法的表达式

    def check_value(self, value: str) -> bool:
        """检查槽位值是否合法"""
        if self.type == 'categorical' and self.valid_values and value not in self.valid_values:
            return False
        if self.type == 'numerical':
            try:
                float(value)
            except ValueError:
                return False
            if self.check:
                try:
                    if eval(self.check.replace('value', value)) == False:
                        return False
                except Exception as e:
                    logger.error(f"Check slot value error: {e}")
                    return False
        return True

# 定义任务步骤模型
class TaskStep(BaseModel):
    step_type: str
    slot: Optional[str] = None
    function: Optional[str] = None

# 定义任务模型
class Task(BaseModel):
    name: str
    description: Optional[str]
    steps: List[TaskStep]
    activate: Optional[bool] = False

# 定义主配置模型
class DomainData(BaseModel):
    scene: str
    description: str
    slots: List[Slot]
    function_calling: List[FunctionCalling]
    tasks: List[Task]
    database_schema: List[TableSchema]

# 定义DomainManager类
class DomainManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.data = self.load_config()

    def load_config(self) -> DomainData:
        """加载并验证配置文件"""
        data = read_yml(self.config_path)
        try:
            config = DomainData(**data)
            print("配置文件加载成功")
            return config
        except ValidationError as e:
            print("配置文件格式有误:", e)
            raise

    def get_scene_name(self) -> str:
        """获取场景名称"""
        return self.data.scene

    def get_scene_description(self) -> str:
        """获取场景描述"""
        return self.data.description
    
    def get_slots(self) -> List[Slot]:
        """获取槽位列表"""
        return self.data.slots

    def get_slot_info(self, slot_name: str) -> Optional[Slot]:
        """根据槽位名称获取槽位信息"""
        return next((slot for slot in self.data.slots if slot.name == slot_name), None)

    def get_tasks(self) -> List[Task]:
        """获取任务列表"""
        return self.data.tasks
    
    def get_task_steps(self, task_name: str) -> Optional[List[TaskStep]]:
        """获取指定任务的步骤列表"""
        task = next((task for task in self.data.tasks if task.name == task_name), None)
        return task.steps if task else None

    def get_database_schema(self) -> List[TableSchema]:
        """获取数据库表的模式信息"""
        return self.data.database_schema

    def get_database_schema_by_name(self, table_name: str) -> Optional[TableSchema]:
        """根据表名称获取数据库表的模式信息"""
        return next((table for table in self.data.database_schema if table.table_name == table_name), None)
