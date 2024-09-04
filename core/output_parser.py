from data.tb_data.dataset import ACTION


class IntentParser:

    def __init__(self,original_input) -> None:
        self.original_input = original_input
        # print(original_input)
        self.intents =  [intent.value for intent in ACTION]
    
    def wash(self):
        #去除空格
        input = self.original_input.replace(" ","")
        #去除最后的句号
        if len(input)!=0 and input[-1] == "。":
            input = input[:-1]
        #去除“/:”等表情符号的干扰
        input = input.replace("/:","")
        #取“：”后的内容
        if "：" in input:
            input = input.split("：")[-1]
        # print("清洗后:",input)
        return input

    def parse_intent(self):
        #看字符串中是否有唯一出现的意图
        input = self.wash()
        has_intent_cnt = 0
        has_intent = []
        for intent in self.intents:
            if intent in input:
                has_intent_cnt += 1
                has_intent.append(intent)
        
        if has_intent_cnt == 1:
            return has_intent[0]
        else:
            return self.original_input
    
