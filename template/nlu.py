

DOMAIN_FILTER_TEMPLATE = """
你是一个对话分析的专家，你需要在一个人机对话的场景中判断用户最后说的话是否与场景有关。
你会得到：
- 一段以json表示的对话历史
- 用户最后一轮说的话
- 场景描述
你只需要给出判断结果（有关/无关），不需要给出分析过程。
你只需要判断最后一轮用户说的话是否与场景有关。注意，用户最后的话可能是之前对话的延续，也即可能省略了一些信息，请不要将这类回答判断为无关。
如果用户最后一轮说的话既包含了与场景无关的内容，也包含了有关的内容，判断为有关。

# 对话历史
{history}

最后，用户说：
{utterance}

# 场景描述
{domain}

# 结果
"""

COARSE_GRAINED_NLU_TEMPLATE = """
你是一个对话分析的专家，你的任务是在一个人机对话的场景中分析对话，从可选意图中挑选出用户最后一轮说的话包含了哪些意图。
你会得到：
- 一段以json表示的对话历史
- 用户最后一轮说的话
- 可选意图
你需要输出：
- 一个python列表，元素为你认为包含的意图id。

你必须从可选意图中挑选，不要创造新的意图。
你可以从可选意图中选择一个或者多个意图。
你不需要给出分析过程。
你需要判断的是最后一轮用户说的话，但你可以参考对话历史。

#对话历史
{history}

最后，用户说：
{utterance}

#可选意图
{coarse_grained_intents}

#输出
"""

FINE_GRAINED_NLU_INFO_REQUIRING_TEMPLATE="""
你是一个对话分析的专家，在一个人机对话的场景中，已知用户目前粗略的意图是：信息获取，你的任务是判断用户所需要获取的信息最有可能来源于何种类型的数据库。
你会得到：
- 一段以json表示的对话历史
- 用户最后一轮说的话
- 可选的信息来源
- 数据库快照
你需要输出：
- 一个json字符串。内容为一个字典，至少包含一个键为“id”,其值为你认为最有可能的信息来源的id，例如“SQL_QUERY”、“VECTOR_QUERY”。另外如果需要，你还需要提供额外的键值对，具体如下。

注意：
如果信息来源于关系型数据库，也就是“SQL_QUERY”，你还需要提供一个键名为“sql”的键值对，其值为查询对应信息的sql语句。
你必须从可选的来源中进行选择，不要创造未提供的来源。
你不需要给出分析过程。
你需要判断的是最后一轮用户说的话，但你可以参考对话历史。

#对话历史
{history}

最后，用户说：
{utterance}

#可选的信息来源
{data_sources}

## sql数据库快照
{database_schema}

#输出
"""

FINE_GRAINED_NLU_INFO_PROVIDING_TEMPLATE="""
你是一个对话分析的专家，在一个人机对话的场景中，已知用户目前粗略的意图是：提供信息，你的任务是更加具体地用户的意图，从更加细粒度的分类中选择一个或多个。
你会得到：
- 一段以json表示的对话历史
- 用户最后一轮说的话
- 可选的细粒度意图。每个细粒度意图可能会有需要附带的信息。通过schema字段指定。
- 当前场景的槽位信息
你需要输出：
- 一个json字符串。内容为一个python列表，其中每个元素为一个字典。字典至少包含一个键名为“id”，其值为你认为最有可能的细粒度意图的id。如果选择的意图需要附带信息，你需要根据提供的信息增加额外的键值对，键名与给出的附带信息描述一致。
注意：
如果一个意图需要附带槽位信息，请在外层用一个名为"slot_value"的键包裹，值为具体槽位信息的键值对。
确保每个意图至多附带一个槽位信息。如果一句话涉及了多个槽位，请将它们放在不同的外层字典元素中，这些字典的"id"键名都是同个意图的名字。这样做可以让你为每个槽指定不同的fill_type。
你必须从可选的细粒度意图中进行选择，不要创造未提供的细粒度意图。
你不需要给出分析过程。
你需要判断的是最后一轮用户说的话，但你可以参考对话历史。

#对话历史
{history}

最后，用户说：
{utterance}

#可选的细粒度意图，以及选择这个意图时你需要附带的信息
{find_grained_intents_info_providing}

#当前场景的槽位信息
{slots}

#输出
"""

FINE_GRAINED_NLU_ACTION_REQUESTING_TEMPLATE="""
你是一个对话分析的专家，在一个人机对话的场景中，已知用户目前粗略的意图是：请求系统执行某些操作。你需要判断系统需要执行的操作种类，如果操作涉及到场景的某个具体任务，你还要根据给定的任务列表以及历史对话中判断出关联的具体任务。
你会得到：
- 一段以json表示的对话历史
- 用户最后一轮说的话
- 系统可执行的操作列表
- 场景内所有的任务列表（其中已经激活的任务会通过activate字段标记出来）
你需要输出：
- 一个json字符串。内容为一个python列表，其中每个元素为一个字典。字典至少包含一个键名为“id”，其值为系统需要执行的操作id。如果操作涉及到具体任务，你需要根据给定的任务列表增加额外的键值对，例如一个名为“task_name”的键。

注意：
你必须从可执行的操作列表中选择操作，以及从定义的任务列表中进行选择，不要改写或者创造新的操作或者任务。
你不需要给出分析过程。
你需要判断的是最后一轮用户说的话，但你可以参考对话历史。

#对话历史
{history}

最后，用户说：
{utterance}

#系统可执行的操作列表
{find_grained_intents_action_requesting}

#场景内所有的任务列表
{tasks}

#输出
"""
