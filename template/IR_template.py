
class IntentClassficationTemplate:
    BASE_INTENT_RECOGNITION_PREFIX='''
    你是一个对话意图分类器。你需要判断一句话是在表达什么意图。
    **只能从可选的意图中选择一个返回。不需要分析和解释。不要返回其他内容**

    ## 可选的意图
    {candidate_intent}
    '''

    BASE_INTENT_RECOGNITION_SUFFIX='''
    ## 你需要判断的句子
    {utterance}
    其表达的意图是：
    '''

    BASE_HISTORY_INTENT_RECOGNITION_TEMPLATE = '''
    ## 对话历史（你的判断需要考虑这段对话历史）
    {history}
    ''' 

    @classmethod
    def with_history_intent_recognition_template(cls, candidate_intent:str, history:str, utterance:str):
        return cls.BASE_INTENT_RECOGNITION_PREFIX.format(candidate_intent=candidate_intent) + cls.BASE_HISTORY_INTENT_RECOGNITION_TEMPLATE.format(history=history) + cls.BASE_INTENT_RECOGNITION_SUFFIX.format(utterance=utterance)

    @classmethod
    def no_history_intent_recognition_template(cls, candidate_intent:str, utterance:str):
        return cls.BASE_INTENT_RECOGNITION_PREFIX.format(candidate_intent=candidate_intent) + cls.BASE_INTENT_RECOGNITION_SUFFIX.format(utterance=utterance)


STEP_INTENT_RECOGNITION_TEMPLATE='''
你是一位分析对话行为的专家。你将得到两部分内容：
- 一段发生在网购场景下{roles}间的对话历史，放在<history></history>中；
- {role}最新的一句话，用query来表示，放在<query></query>中。
你需要严格遵循如下两个步骤来将{role}的话归类到合适的意图中。
1. 第一步。你需要在**{role}**的对话历史中筛选出与query表达意图最为接近的{top_n}句话，以json的形式返回。json应为一个包含{top_n}个字典的列表，每个字典内容包括"id","intent","reason"三个键，对应的值分别为句子序号，这个句子的意图（不需要改动）以及为什么这个句子与query表达意图接近。如果你认为满足条件的不足{top_n}句话，则只返回你认为满足条件的句子。
2. 第二步。你在第一步中已经得到了{top_n}个与query表达的意图最为接近的句子。你现在需要参考这些句子以及它们的意图，最终推断出query的意图。你需要给出你的分析过程，并从可选的意图中选择一个最符合query的意图。
你应该以如下json格式返回。
```json
{{
    "step1": [
        {{
            "id": "",
            "intent":"",
            "reason": ""
        }}，
        ...
    ],
    "step2": {{
        "analysis": "你的分析过程。分析如何从第一步得到的{top_n}个句子中推断出query的意图",
        "query_intent":"你的最终判断。必须是可选意图中的一个"
    }}
}}
```

**你只能分析和考虑下述意图。**
## 可选的意图
{candidate_intent}

{Example}
## 开始
<history>
{history}
</history>

<query>
{sentence}
</query>

<answer>
'''

STEP_EXAMPLE='''
## Example1
<history>
1 买家:我看了评价，说只能充电，不能下载 意图：告知
2 卖家:  那一款 意图：回复
3 卖家:  录音笔 可以的亲   意图：回复
4 买家:这2款都可以下载的吗 意图：询问
5 卖家:  是读取数据 亲 意图：回复
6 买家:就是把录音笔中的数据，放到电脑里吗 意图：询问
7 卖家:  可以的 意图：回复
8 买家:也可以把歌曲放到录音笔中吗， 意图：询问
9 卖家:  可以的亲 意图：回复
10 卖家:  要放在根目录里面 意图：告知
</history>

<query>
也就是相当于MP3的功能吗
</query>

<answer>
{{
    "step1": [
        {{
            "id": 4,
            "intent": "询问",
            "reason": "这一句话是买家在询问2款录音笔是否都可以下载，query是在询问录音笔的功能是否相当于MP3的功能。这两句话都是在针对录音笔某功能进行询问"
        }},
        {{
            "id": 6,
            "intent": "询问",
            "reason": "这一句话是买家在询问是否可以将录音笔中的数据放到电脑里，query是在询问录音笔的功能是否相当于MP3的功能。这两句话都是在针对录音笔某功能进行询问。"
        }},
        {{
            "id": 8,
            "intent": "询问",
            "reason": "这一句话是买家在询问歌曲能否放到录音笔中，query是在询问录音笔的功能是否相当于MP3的功能。这两句话都是在针对录音笔某功能进行询问。"
        }}
    ],
    "step2": {{
        "analysis": "从第一步得知，与query最接近的几句话意图都是询问。而query本身也是在对录音笔功能进行询问。因此，我认为query的意图是询问录音笔的功能。",
        "query_intent": "询问"
    }}
}}

'''

HISTORY_WITH_LABEL_TEMPLATE = '''{ROLE}:{SENTENCE} 意图：{INTENT}\n'''

HISTORY_NO_LABEL_TEMPLATE = '''{ROLE}:“{SENTENCE}”\n'''



SUMMARIZE_HISTORY_TEMPLATE = """
你是一个对话总结的专家。你需要根据对话历史，用一段话来总结对话。
你的总结将会被用作对话意图识别的上文，你的总结中应该尽量描述清楚每一次互动双方的对话行为。请突出描述最后一轮对话中双方的对话行为，并猜测下一句话小明会说什么，并给出解释。
<conversation></conversation> 标签中的内容属于聊天对话历史。
将你的总结放在 <summerization>标签中。summerization标签中应包含若干个<turn>标签和一个<final turn>标签。
<turn>标签后的内容为每一次互动中双方产生的对话行为。
<final turn>后的内容为最后一轮对话的描述。

## Example
<conversation>
小红: "在吗？"
小明: "您好 在的"
小红: "我想买个录音笔，哪种好呀？"
小明: " <C-hyperlink> "
小红: "推荐下"
小明: "这款是客户评价最好的 亲 "
小红: "还能优惠吗？"
小明: "亲 都是最低促销价了，不议价的哦 "
小红: "有那个咖啡色吗？"
<conversation>

<summerization>
<turn 1>
小红向小明打了招呼后，小明回应。
<turn 2>
小红表示自己想买个录音笔，希望小明推荐哪一种较好。
<turn 3>
小明推荐了一款客户评价最好的录音笔。小红希望更优惠一些，小明表示不议价。
<turn 4>
最后，小红询问这款录音笔是否有咖啡色的款式。
<final turn>
小明接下来有可能会告知小红这款录音笔是否有咖啡色的款式，因为小红希望得到这个问题的答案。
</summerization>

## 开始
<conversation>
{history}
</conversation>

<summerization>
"""

COMPLETION_TEMPLATE = """
使用聊天对话中的上下文补全小明的话，使其成为一个语义独立完整的句子。
<conversation></conversation> 标签中的内容属于聊天对话历史。
<origin_utterance></origin_utterance> 标签中的内容属于小明的话。
将独立问题放在 <standalone_utterance>标签中。
**省略开场白，不要解释，不要改变原话的句式特点**。

## Example
### Example1
<conversation>
小红: "在吗？"
小明: "您好 在的"
小红: "我想买个录音笔，哪种好呀？"
小明: " <C-hyperlink> "
小红: "推荐下"
小明: "这款是客户评价最好的 亲 "
小红: "还能优惠吗？"
小明: "亲 都是最低促销价了，不议价的哦 "
小红: "有那个咖啡色吗？"
</conversation>

<origin_utterance>
"目前没有哦 亲"
</origin_utterance>

<standalone_utterance>
"目前没有咖啡色的录音笔哦 亲"

## 开始
<conversation>
{history}
</conversation>

<origin_utterance>
{sentence}
</origin_utterance>

"""

CROSSWOZ_NLU_TEMPLATE = '''
你是一个语言行为专家。你需要从对话中分析出对话行为(dialogue_act)，包括意图(intent)、领域(domain)、槽名(slot)及相应的槽值(value)四个部分，并仿照如下json格式返回。
```json
{{
    \"dialogue_act\":[
        {{
            \"intent\": 从[General,Inform,Request,Recommend,NoOffer,Select]中选择一个,
            \"domain\": 当intent为General时，从[bye,greet,reqmore,thank,welcome]选择一个；否则，从[酒店、景点、餐馆、地铁、出租]中选择一个,
            \"slot\": 当intent为General时，为空；否则，从domain对应的槽名中选择一个,
            \"value\": 当intent为General时，为空；否则，槽名对应的槽值
        }},
        ...
    ]
}}
```

**你需要用到的信息：**
- General意图表示一般性对话行为，包含了如下5个子意图：
    - bye 含义：再见
    - greet 含义：打招呼，问候
    - reqmore 含义：需要更多信息
    - thank 含义：表示感谢
    - welcome 含义：表示欢迎
- 不同领域对应的槽名如下：
    - 酒店：[名称、酒店类型、酒店设施-XXX、价格、评分、周边景点、周边餐馆、周边酒店]
    - 景点：[名称、门票、游玩时间、评分、周边景点、周边餐馆、周边酒店]
    - 餐馆：[名称、推荐菜、人均消费、评分、周边景点、周边餐馆、周边酒店]
    - 地铁：[出发地、目的地]
    - 出租：[出发地、目的地]

以下为一个示例：
对话：您好，我朋友想找个酒店住宿，请帮忙找一家评分是4.5分以上，最低价格是200-300元的酒店，并且他希望提供无烟房和暖气。
```json
{
    "dialog_act": [
        {
            "intent":"General",
            "domain":"greet",
            "slot":"none",
            "value":"none"
        },
        {
            "intent":"Inform",
            "domain":"酒店",
            "slot":"价格",
            "value":"200-300元"
        },
        {
            "intent":"Inform",
            "domain":"酒店",
            "slot":"评分",
            "value":"4.5分以上"
        },
        {
            "intent":"Inform",
            "domain":"酒店",
            "slot":"酒店设施-无烟房",
            "value":"是"
        },
        {
            "intent":"Inform",
            "domain":"酒店",
            "slot":"酒店设施-暖气",
            "value":"是"
        },
        {
            "intent":"Request",
            "domain":"酒店",
            "slot":"名称",
            "value":""
        }
    ]
}
```

请注意，一句话中可能包含多个对话行为，请尽量提取所有的行为。

对话历史：
{history}

请根据以上对话历史分析最后一句话中的对话行为。
'''