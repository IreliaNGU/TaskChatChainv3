## 基本概念
### 前提
1. 设计的意图都是为了完成任务服务的，所以有些对任务完成不重要的意图就不去定义，否则难度太大
2. 设计的意图都是为用户视角服务的，比如用户一般不会解释原因，所以就取消了EXPLIAN这个意图

### 规则

#### 意图
意图分成ASK、MODIFY、COMPLEMENT、REPLY、REQUEST、GREET、THANK、COMPLAINT、BYE、OTHER十个大类。部分意图逻辑上会细分子意图。  
识别的时候是直接识别到叶子意图。   
一句话允许识别到多个意图 

ASK：  
- ASK_INFO
- ASK_REASON
- ASK_IF 

REPLY:
- REPLY_INFO
- REPLY_DONT_KNOW
- REPLY_REFUSE
- REPLY_CONFIRM

#### 槽位
槽位有ENTITY、PROPERTY、VALUE、ACTION、NECESSARY_SLOT、CONFUSED、CHOICE几种。  
在通用意图识别中，客户自定义的场景知识会成为以上这些槽位的值出现。  
槽值之间允许嵌套。

## 由客户自定义的场景知识：

### Entity  
- TYPE
- PROPERTY1
- PROPERTY2
- PROPERTYN

根据RELATED和TYPE能唯一标识一个ENTITY

例如：  
网购场景下：  
S221移动电源：
- TYPE： 移动电源
- PROPERTY1: 价格
- PROPERTY2: 容量
- ...

金融场景下：  
东方证券:
- TYPE: 证券
- PROPERTY1: 价格 
- PROPERTY2：成交量
- ...  

订票场景下:  
F1234航班:
- TYPE: 航班
- PROPERTY1: 出发时间
- PROPERTY2: 到达时间
- ...

ENTITY是可以层叠的。
比如，“商品”作为一个ENTITY，其一定有开拍时间，价格等等的属性。而“录音笔”这个ENTITY属于“商品”的子类，就不需要定义价格这个属性了，只需要定义其特有的属性，比如“距离限制”等等。类似继承的概念。

注意：user本身应声明为一个entity，其PROPERTY要自定义，比如年龄、性别、已拍商品等等

### ACTION
- PURPOSE（可选）
- NECESARY_INFO1
- NECESARY_INFO2
- ...

例如：  
订票:    
- NECESARY_INFO1: 预期出发时间
- NECESARY_INFO2: 预期到达时间
- ...

投资方案推荐:    
- NECESARY_INFO1: 风险承受能力
- NECESARY_INFO2: 现金流
- ... 


### 另
在淘宝数据集上做意图识别的时候，需要一些方法来根据历史对话来识别