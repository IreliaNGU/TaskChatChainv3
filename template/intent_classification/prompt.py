DEFAULT_IF_COMPLETION_TEMPLATE = r'''
你是一个语言行为专家。你需要判断下面给出的一个对话行为体系（dialogue act framework）是否完善。
如果一个对话行为体系是完善的，那么对于所有的对话场景中的每一条对话（utterance），都能将其标签为该体系中的一个和多个对话行为。
如果你认为该体系不够完善，请给出修改建议，例如增加、删除或者替换其中的某些意图，并给出完善后的体系。
完善后的体系请以json的格式进行返回，使用```json及```包裹。内容为一个列表，每一个元素是一个字典，字典的键为name、description，分别表示对话行为的名称及对该行为的描述。

## dialogue act framework
{intent_framework}

## output
'''

FEW_SHOTS = '''
## Example1
### dialogue act framework
    - 咨询
    - 反馈
    - 阐述
    - 确认
    - 解释
    - 请求
    - 接受
    - 问候
    - 闲聊

### utterance1
这附近有什么比较好吃的餐厅吗？
### answer
咨询
### utterance

'''