import json
import os
import zipfile
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, TEST_SPLIT, CORRECT_TAG, Input, Output


CLEVA_DATA_URL = "http://emnlp.clevaplat.com:8001/data"
CLEVA_DATA_PATH = "benchmark_output/scenarios/cleva"


@dataclass(frozen=True)
class PromptSetting:
    instructions: str
    input_noun: Optional[str] = None
    newline_after_input_noun: bool = False
    output_noun: str = ""
    newline_after_output_noun: bool = False


class CLEVAScenario(Scenario):
    """
    Scenario for CLEVA benchmark (https://arxiv.org/pdf/2308.04813.pdf).
    """

    name = "cleva"

    def __init__(
        self,
        version: str,
        task: str,
        subtask: str,
    ):
        """
        Initializes CLEVA scenario.
        Args:
            version: String identifier for version in a format of 'v[1-9]*([0-9])'.
            task: String identifier for task.
            subtask: String identifier for subtask.
        """
        super().__init__()
        self.task = task
        self.subtask = subtask
        self.version = version
        self.splits: Dict[str, str] = {
            "train": TRAIN_SPLIT,
            "test": TEST_SPLIT,
        }

    @classmethod
    def download_dataset(cls, version: str):
        download_url: str = CLEVA_DATA_URL + f"/{version}/data.zip"
        data_dir: str = os.path.join(CLEVA_DATA_PATH, "data", version)
        ensure_directory_exists(data_dir)
        ensure_file_downloaded(source_url=download_url, target_path=os.path.join(data_dir, "data.zip"))

        with zipfile.ZipFile(os.path.join(data_dir, "data.zip"), "r") as zip_ref:
            zip_ref.extractall(data_dir)

    def load_dataset(self) -> Dict[str, List[Dict[str, Any]]]:
        data_dir: str = os.path.join(CLEVA_DATA_PATH, "data", self.version, self.task)
        if self.subtask:
            data_dir = os.path.join(data_dir, self.subtask)

        dataset: Dict[str, List[Dict[str, Any]]] = {}
        for split in self.splits.keys():

            with open(os.path.join(data_dir, f"{split}.jsonl"), "r") as fin:
                dataset[split] = []
                for line in fin.readlines():
                    dataset[split].append(json.loads(line))

        return dataset

    @staticmethod
    def load_prompt_settings(task: str, subtask: Optional[str], version: str) -> Dict[str, Any]:
        prompt_dir: str = os.path.join(CLEVA_DATA_PATH, "data", version, task)
        if subtask:
            prompt_dir = os.path.join(prompt_dir, subtask)
        file_path = os.path.join(prompt_dir, "prompt_setting.json")
        if os.path.isfile(file_path):
            with open(file_path, "r") as fin:
                prompt_settings = json.load(fin)
        else:
            raise ValueError(f"Missing prompt setting file at '{file_path}'")
        return prompt_settings[0]  # Currently, we only take the first prompt setting.

    def get_instances(self) -> List[Instance]:
        # Download the raw data
        dataset = self.load_dataset()

        # Read all the instances
        instances: List[Instance] = []
        for split in self.splits:
            for row in dataset[split]:
                instances.append(self.process_instance(row, self.splits[split]))

        return instances

    @staticmethod
    def multiple_choice_answer_to_reference(answer: str, correct_answer: List[str]) -> Reference:
        return Reference(Output(text=answer), tags=[CORRECT_TAG] if answer in correct_answer else [])

    def process_instance(self, row: Dict[str, Any], split: str) -> Instance:
        text: str = row["text"]
        if "choices" in row.keys():
            answers: List[str] = row["choices"]
            correct_choice: List[int] = row["label"]
            correct_answer: List[str] = [answers[idx] for idx in correct_choice]
            references: List[Reference] = [
                self.multiple_choice_answer_to_reference(answer, correct_answer) for answer in answers
            ]
        else:
            correct_answer = row["label"]
            references = [Reference(Output(text=answer), tags=[CORRECT_TAG]) for answer in correct_answer]

        instance = Instance(
            input=Input(text=text),
            references=references,
            split=split,
        )
        return instance

    @classmethod
    def get_prompt_setting(cls, task: str, subtask: Optional[str], version: str) -> PromptSetting:
        prompt_setting_dict = cls.load_prompt_settings(task, subtask, version)
        prompt_setting = PromptSetting(
            instructions=prompt_setting_dict.get("instructions", ""),
            input_noun=prompt_setting_dict.get("input_noun", None),
            newline_after_input_noun=prompt_setting_dict.get("newline_after_input_noun", False),
            output_noun=prompt_setting_dict.get("output_noun", ""),
            newline_after_output_noun=prompt_setting_dict.get("newline_after_output_noun", False),
        )
        return prompt_setting


class CLEVATextClassificationScenario(CLEVAScenario):
    """
    The text classification task of CLEVA benchmark.

    An example is:
        以下文本属于哪个类别？

        问题: 自考本科选择什么专业好？
        A. 体育
        B. 财经
        C. 娱乐
        D. 军事
        E. 文化
        F. 旅游
        G. 游戏
        H. 农业
        I. 股票
        J. 教育
        K. 国际
        L. 科技
        M. 汽车
        N. 房屋
        O. 故事
        答案: J

        问题: 劲爆！新能源电池全新变化，固态电池有望成风口，受益龙头蓄势待
        A. 体育
        B. 财经
        C. 娱乐
        D. 军事
        E. 文化
        F. 旅游
        G. 游戏
        H. 农业
        I. 股票
        J. 教育
        K. 国际
        L. 科技
        M. 汽车
        N. 房屋
        O. 故事
        答案:

    Target: M
    """

    # name = "cleva_text_classification"
    description = "Text classification task in CLEVA benchmark"
    tags = ["text_classification", "multiple_choice"]


class CLEVAOpinionMiningScenario(CLEVAScenario):
    """
    The opinion mining task of CLEVA benchmark.

    An example is:
        请根据以下陈述，挖掘出陈述中的观点目标。

        陈述: 从亚龙湾出发，大概40分钟左右的车程即可到达码头，转乘轮渡抵达蜈支洲岛。
        主体: 蜈支洲岛

        陈述: 这是一座被称为最美大学的校园，座山面海是厦门大学得天独厚的自然条件。
        主体:

    Target: 厦门大学
    """

    # name = "cleva_opinion_mining"
    description = "Opinion mining task in CLEVA benchmark"
    tags = ["opinion_mining"]


class CLEVAPinyinTransliterationScenario(CLEVAScenario):
    """
    The Pinyin transliteration task of CLEVA benchmark.

    An example of pinyin2zh subtask is:
        把以下汉语拼音转换成相应的汉语句子。

        拼音：pín kùn xiàn xiàng yǐ jīng yǒu suǒ hǎo zhuǎn 。
        汉字：贫困现象已经有所好转。

        拼音：wǒ men shǒu tóu mù qián dōu bǐ jiào kuān yù
        汉字：

    Target: 我们手头目前都比较宽裕

    An example of zh2pinyin subtask is:
        把以下汉语句子转换成相应的汉语拼音。

        汉字：她俩之间是陌生人关系
        拼音：tā liǎ zhī jiān shì mò shēng rén guān xì

        汉字：这是球类比赛
        拼音：

    Target: zhè shì qiú lèi bǐ sài
    """

    # name = "cleva_pinyin_transliteration"
    description = "Pinyin transliteration task in CLEVA benchmark"
    tags = ["pinyin_transliteration"]


class CLEVAClassicalChineseUnderstandingScenario(CLEVAScenario):
    """
    The classical Chinese understanding task of CLEVA benchmark.

    An example is:
        这句现代文可以用哪句古文来表达？

        现代文：详细地表述了自己的苦衷。
        A. 流觞款叙情
        B. 款曲话情亲
        C. 款曲会良姻
        D. 款曲陈此情
        答案：D

        现代文：也不要埋怨故乡太遥远。
        A. 莫恨故乡遥
        B. 谁道故乡遥
        C. 故乡应渐遥
        D. 莫动故乡情
        答案：

    Target: A
    """

    # name = "cleva_classical_chinese_understanding"
    description = "Classical Chinese understanding task in CLEVA benchmark"
    tags = ["classical_chinese_understanding", "multiple_choice"]


class CLEVASentimentAnalysisScenario(CLEVAScenario):
    """
    The sentiment analysis task of CLEVA benchmark.

    An example is:
        这个产品评价是正面还是负面的？

        评价：不是快充，被坑了
        A. 负面
        B. 正面
        答案：A

        评价：商城就是快好省，快好省
        A. 负面
        B. 正面
        答案：

    Target: B
    """

    # name = "cleva_sentiment_analysis"
    description = "Sentiment analysis task in CLEVA benchmark"
    tags = ["sentiment_analysis"]


class CLEVAInstructionFollowingScenario(CLEVAScenario):
    """
    The instruction following task of CLEVA benchmark.

    An example is:
        将e视为48+12。问：e的第一位数字是啥？答：
        A. 6
        B. 2

    Target: A
    """

    # name = "cleva_instruction_following"
    description = "Instruction following task in CLEVA benchmark"
    tags = ["instruction_following", "multiple_choice"]

    def __init__(
        self,
        version: str,
        task: str,
        subtask: str,
    ):
        super().__init__(version, task, subtask)
        self.splits: Dict[str, str] = {
            "test": TEST_SPLIT,
        }


class CLEVAFactCheckingScenario(CLEVAScenario):
    """
    The fact checking task of CLEVA benchmark.

    An example is:
        请判断以下陈述是否为真实的。

        陈述：献血会让人上瘾。
        A. 真实
        B. 虚假
        C. 信息不足
        答案：B

        陈述：2021年，金星上没有发现生命。
        A. 真实
        B. 虚假
        C. 信息不足
        答案：

    Target: A
    """

    description = "Fact checking task in CLEVA benchmark"
    tags = ["fact_checking", "harms", "multiple_choice"]


class CLEVATranslationScenario(CLEVAScenario):
    """
    The translation task of CLEVA benchmark.

    An example of en2zh subtask is:
        请把下面的英文句子翻译成相应的中文句子。

        英语：On Thursday, a bronze statue of Vidyasagar was installed at Vidyasagar College.
        中文：星期四，维迪亚萨加学院矗立了维迪亚萨加铜像。

        英语：This will help the present generation to know about the man, who had waged a war against women oppression
             and propagated widow remarriage, he said.
        中文：

    Target: 他说，这将有助于当代人了解这位名人，他发动了一场反对妇女压迫的战争，并鼓励寡妇再婚。

    An example of zh2en subtask is:
        请把下面的中文句子翻译成相应的英文句子。

        中文：北京市场监管部门节前开展综合执法 商品价格基本稳定
        英语：Beijing Market Regulator Initiates Comprehensive Law Enforcement before the Holiday,
             CPI Basically Remains Stable

        中文：中国驻柬大使馆外交官仲跻法、柬华理事总会代表、柬埔寨江西商会会长魏思钰等为获奖嘉宾颁奖。
        英语：

    Target: Zhong Jifa, diplomat of the Chinese Embassy in Cambodia, and Wei Siyu, representative of the Cambodian
            Chinese Council and President of Jiangxi Chamber of Commerce in Cambodia,
            presented the awards to the winners.
    """

    description = "Translation task in CLEVA benchmark"
    tags = ["translation"]


class CLEVAIntentUnderstandingScenario(CLEVAScenario):
    """
    The intent understanding task of CLEVA benchmark.

    An example is:
        阅读以下材料，回答单项选择题：

        美国科学家声称，每人生来有两个脑，即颅脑与肠脑。肠脑位于食管、胃脏、小肠与结肠内层组织的鞘中，含有神经细胞、神经传递质、蛋白质和
        复杂的环行线路。结肠炎、过敏性肠综合症等都与肠脑内产生的问题有关。肠脑中几乎能找到颅脑赖以运转和控制的所有物质，如血清素、多巴胺、
        谷氨酸、去甲肾上腺素、一氧化氮等。此外，肠脑中还存在多种被称为神经肽的脑蛋白、脑啡肽以及对神经起显著作用的化学物质。颅脑面临惊恐时
        释出的应激激素会冲击胃脏以生痉挛；惊恐又引起交感神经影响肠脑的血清素分泌量。应激激素过分刺激还会导致腹泻。当情绪压抑时，食管神经
        受到高度刺激会感到吞咽困难；颅脑释出的应激激素还会改变胃脏与食管间的神经功能，导致胃灼热。最初的脑神经系统起始于管形动物，生存竞争
        需要更复杂的颅脑，从而发展了中枢神经系统。重要的肠神经系统不能进入头颅与胃肠相联，而为了适应高级动物进食和消化的需要，自然法则就
        保存了有独立功能的肠神经系统。就人而言，早期胚胎发育中产生的神经脊，一部分进入了中枢神经系统，另一部分变成肠神经系统，通过迷走神经
        连接两者——颅脑和肠脑。

        下列解说，符合原文意思的一项是：
        A. 应激激素作用于肠脑引起肠神经系统化学物质的改变。
        B. 情绪的变化是肠脑和颅脑发生联系的重要渠道。
        C. 进食和消化的需要是肠神经系统形成的基础条件。
        D. 重要的肠神经系统因不能进入头颅而成为独立系统。
        答案：D

        1990年，加拿大多伦多大学的米切尔·洛林宣布，在2.6亿年以前，栖息在美国得克萨斯山区一种外形像蜥蜴的名叫四角龙的爬行动物，确实是
        哺乳动物的远古“亲戚”，从而填补了进化链中从爬行动物到哺乳动物中缺少的一环。\n1987年，米切尔·洛林研究了一块盘龙类的头骨化石。
        随着研究的深入，化石上的一些细节却使他困惑不解。因为大多数的盘龙类在腭部有很大的孔，而在较进化的兽孔类身上，这个孔已被封闭，四角龙
        也有一个腭孔，但已明显缩小，其直径仅仅为0.635厘米。此外，盘龙类在头部背面有一块很大的骨，用以支持颌骨，在兽孔类中，这块骨头已大大
        缩小了，而四角龙的这块骨要较兽孔类大，又较盘龙类稍小。更为重要的是，四角龙的头角上有个骨架，穿越颞孔的咀嚼肌像兽孔类那样直接依附
        其上，而不像盘龙类那样由肌腱相接。\n这些发现使洛林相信，四角龙是盘龙类和兽孔类之间的一个过渡类型。他又把从12块盘龙类和兽孔类动物化石
        中获得的信息输入电脑（包括腭孔、颞孔形状，头颅骨形状，牙齿数量和着生位置等），然后由电脑判断出两者之间的联系。结果表明，在进化树上，
        通向兽孔类一边的第一个分叉就是四角龙。

        文中“直接依附其上”的“其”字指代的是：
        A. 四角龙的头角
        B. 头角上的骨架
        C. 被穿越的颞孔
        D. 穿越颞孔的肌肉
        答案：

    Target: B
    """

    description = "Intent understanding task in CLEVA benchmark"
    tags = ["intent_understanding", "multiple_choice"]

    def process_instance(self, row: Dict[str, Any], split: str) -> Instance:
        context: str = row["context"]
        question: str = row["question"]
        text: str = f"{context}\n\n问题：{question}"
        answers: List[str] = row["choices"]
        correct_choice: List[int] = row["label"]
        correct_answer: List[str] = [answers[idx] for idx in correct_choice]
        references: List[Reference] = [
            self.multiple_choice_answer_to_reference(answer, correct_answer) for answer in answers
        ]

        instance = Instance(
            input=Input(text=text),
            references=references,
            split=split,
        )
        return instance


class CLEVACoreferenceResolutionScenario(CLEVAScenario):
    """
    The coreference resolution task of CLEVA benchmark.

    An example is:
        不过还有一峰骆驼更是出尽洋相，不但脖子上缠了四五朵塑料花，耳朵上还各绑了一团红红绿绿的花布，背上还抹了一大团鲜艳的红。
        时常见它花枝招展、喜气洋洋地在驻地附近走来走去。
        在上文中，“背”和“它”是否指代了同一个对象？
        A. 不是
        B. 是
        答案：A

        渐渐地，汤中凝结出一团团块状物，将它们捞起放进盆里冷却，肥皂便出现在世上了。
        在上文中，“块状物”和“它们”是否指代了同一个对象？
        A. 不是
        B. 是
        答案：

    Target: B
    """

    description = "Coreference resolution task in CLEVA benchmark"
    tags = ["coreference_resolution", "multiple_choice"]

    def process_instance(self, row: Dict[str, Any], split: str) -> Instance:
        context: str = row["context"]
        span1: str = row["span1"]
        span2: str = row["span2"]
        text: str = f"{context}\n在上文中，“{span1}”和“{span2}”是否指代了同一个对象？\n"
        answers: List[str] = row["choices"]
        correct_choice: List[int] = row["label"]
        correct_answer: List[str] = [answers[idx] for idx in correct_choice]
        references: List[Reference] = [
            self.multiple_choice_answer_to_reference(answer, correct_answer) for answer in answers
        ]

        instance = Instance(
            input=Input(text=text),
            references=references,
            split=split,
        )
        return instance


class CLEVAReadingComprehensionScenario(CLEVAScenario):
    """
    The coreference resolution task of CLEVA benchmark.

    An example is:
        阅读以下内容，选择合适的选项回答问题。

        尽管方便快捷的“网络阅读”已经成为了一种生活时尚，但纸质阅读仍然发挥着很大的作用。日前，我们通过问卷调查、现场采访的方式对不同
        阶层的读者进行了调查，结果显示，市民电子阅读的兴趣日渐提高，但很多人仍在坚守传统的纸质阅读。\n在为期三天的阅读习惯调查中，记者
        发现，经常上网浏览书籍的读者占被调查者的60%，而从不选择上网浏览的读者仅占25%；在喜欢的阅读方式调查中，喜爱纸质阅读的读者高达90%，
        占绝对优势，而喜欢网上阅读的人只占到8%，很明显，读者还是更喜爱传统的阅读方式。\n许多读者表示传统图书提供了非常明了、有用的信息，
        阅读时没有广告等干扰，有效地防止了一些时间的浪费。另外，多数读者认为长期对着屏幕阅读，也容易带来很多后遗症：眼干、肩膀疼、腰疼等。
        纸质阅读更有利于保护眼睛。采访中，很多读者认为纸质书带给我们的不仅仅是书中的文字，更是手捧文化的一种美妙感觉，这是任何形式的电子
        阅读器都无法做到的。

        问题：被调查者的阅读习惯有：
        A. 少数人接受纸质阅读
        B. 年轻人喜欢网络阅读
        C. 多数人经常上网阅读
        D. 大部分记者习惯网上阅读
        答案：C

        去年中国汽车生产和销售分别为1379.10万辆和1364.48万辆，首次成为世界汽车生产销售第一大国。其中家庭用车的销售量是汽车销售
        总量的51%，占乘用车销售总量的44%。

        问题：请选出与试题内容一致的一项。
        A. 去年中国汽车销售量大于生产量
        B. 去年中国再次成为汽车第一大国
        C. 去年中国乘用车的销售量比例是44%
        D. 去年中国家庭用车的销售量超过总销售量的一半
        答案：

    Target: D
    """

    description = "Reading comprehension task in CLEVA benchmark"
    tags = ["reading_comprehension", "multiple_choice"]

    def process_instance(self, row: Dict[str, Any], split: str) -> Instance:
        context: str = row["context"]
        question: str = row["question"]
        text: str = f"{context}\n\n问题：{question}\n"
        answers: List[str] = row["choices"]
        correct_choice: List[int] = row["label"]
        correct_answer: List[str] = [answers[idx] for idx in correct_choice]
        references: List[Reference] = [
            self.multiple_choice_answer_to_reference(answer, correct_answer) for answer in answers
        ]

        instance = Instance(
            input=Input(text=text),
            references=references,
            split=split,
        )
        return instance


class CLEVADialogueGenerationScenario(CLEVAScenario):
    """
    The dialogue generation task of CLEVA benchmark.

    An example is:
        请根据对话历史回复用户询问。

        用户：你好，我想找一个价格是1000元以上，评分是4.5分以上的酒店，有什么好的地方给我推荐吗？
        系统：给你推荐北京昆泰嘉华酒店，完全符合你的条件呢。
        用户：是吗，该酒店是什么类型啊？
        系统：豪华型酒店。
        用户：好的，能帮我查一下它家是否提供商务中心吗？
        系统：酒店提供商务中心的。
        用户：太好了，定完酒店，我打算找个评分是4.5分以上，游玩时长是1小时 - 2小时，票价是200元以上的景点游玩，给我点建议好吗？
        系统：乐多港奇幻乐园是个不错的去处，非常好玩的。
        用户：好啊，就去乐多港奇幻乐园玩吧，景点周边有酒店吗？
        系统：

    Target: 嗯，周边有一个如家快捷酒店(北京昌平鼓楼西街店)。
    """

    description = "Dialogue generation task in CLEVA benchmark"
    tags = ["dialogue_generation"]

    def get_instances(self) -> List[Instance]:
        # Download the raw data
        dataset = self.load_dataset()

        # Read all the instances
        instances: List[Instance] = []
        for split in self.splits:
            for row in dataset[split]:
                # One row could contain multiple conversation instances.
                instances.extend(self.process_dialogue_instance(row, self.splits[split]))

        return instances

    def process_dialogue_instance(self, row: Dict[str, Any], split: str) -> List[Instance]:
        instances: List[Instance] = []
        text: str = ""
        speaker_mapping = {"sys": "系统", "usr": "用户"}
        dialog = row["dialogue"]

        for turn_id, utt in enumerate(dialog):

            content: str = utt["content"]
            speaker: str = utt["role"]

            # For task-oriented dialogue tasks, agents should response to users' questions according to the history.
            if speaker == "sys" and turn_id > 0:
                correct_answer = [content]
                references = [Reference(Output(text=answer), tags=[CORRECT_TAG]) for answer in correct_answer]

                instance = Instance(
                    input=Input(text=text),
                    references=references,
                    split=split,
                )
                instances.append(instance)

            # append history utterances
            if turn_id > 0:
                text += "\n"
            text += "{speaker}: {content}".format(speaker=speaker_mapping[speaker], content=content)

        return instances


class CLEVASubjectKnowledgeScenario(CLEVAScenario):
    """
    The subject knowledge task of CLEVA benchmark.
    We follow https://github.com/stanford-crfm/helm/tree/main/scripts/fact_completion to construct the Chinese dataset.
    Considering the Chinese characteristics, we rewrite and extend the relations.

    An example is:
        补全下列句子中下划线处的实体。

        输入：礼记所处的年代是__。
        输出：周朝

        输入：慕容复出现在作品《__》中。
        输出：天龙八部

        输入：古剑奇谭在__首次播放。
        输出：

    Target: 湖南卫视
    """

    description = "Subject knowledge task in CLEVA benchmark"
    tags = ["subject_knowledge"]

    def process_instance(self, row: Dict[str, Any], split: str) -> Instance:
        # Currently, HELM will concat all references as an answer in few-shot setting.
        # This function ensure that there is only one reference for each instance.
        text: str = row["text"]

        # truncate the label
        correct_answer = row["label"][:1] if split == "train" else row["label"]
        references = [Reference(Output(text=answer), tags=[CORRECT_TAG]) for answer in correct_answer]

        instance = Instance(
            input=Input(text=text),
            references=references,
            split=split,
        )
        return instance


class CLEVACulturalKnowledgeScenario(CLEVAScenario):
    """
    The cultural knowledge task of CLEVA benchmark.

    An idiom example is:
        请根据文段内容补全下划线处的成语。

        文本: 王俊凯登杂志封面挑战日式发型少年____娱乐微小二2021-03-09 16:07:14 1/12王俊凯为时尚杂志拍摄的写真曝光,封面中的他,突破以往风格,首次尝试日系长发造型,
        身穿黑色剪裁西装,搭配金丝边眼镜,雅痞气质无比吸睛。0...
        A. 大错特错
        B. 化为乌有
        C. 红颜薄命
        D. 委曲求全
        E. 富丽堂皇
        F. 逢凶化吉
        G. 初露锋芒
        答: G

        文本: 1997年上映的电影《宋家王朝》中,影星杨紫琼,张曼玉,邬君梅,分别扮演宋霭龄,宋庆龄,宋美龄,其片头语“遥远的旧中国有三姐妹,一个爱钱,一个爱国,一个爱权”不胫而走,
        却也____,成为对宋氏三姐妹的总体评价。图中是《宋家王朝》的...
        A. 异想天开
        B. 时移世易
        C. 半生半熟
        D. 言之凿凿
        E. 大有可为
        F. 喧宾夺主
        G. 焕然一新
        答:

    Target: D
    """

    description = "Cultural knowledge task in CLEVA benchmark"
    tags = ["cultural_knowledge", "multiple_choice"]


class CLEVAClosedBookQuestionAnsweringScenario(CLEVAScenario):
    """
    The closed-book QA task of CLEVA benchmark.

    An example is:
        问题：云南省少数民族语言文字工作条例是什么时候施行的？
        答案：2013年5月1日

        问题：天主教圣菲总教区什么时候创立的？
        答案：

    Target: 1850年
    """

    description = "Closed-book Question Answering task in CLEVA benchmark"
    tags = ["closed_book_question_answering"]


class CLEVASummarizationScenario(CLEVAScenario):
    """
    The summarization task of CLEVA task.

    An example of dialogue summarization is:
        用户：咨询订单号:[订单编号]
        客服：有什么问题我可以帮您处理或解决呢?
        用户：想退单
        客服：亲爱哒，请问是什么原因您要退款呢是有其他人员通过微信或者QQ联系您刷单或者兑换门票的吗
        用户：拍错了
        用户：是的
        客服：亲亲，虚拟商品属于及时交易到账，交易成功之后无法拦截，这就好比您去充值话费是一样的道理，已经交易到账，无法进行拦截呢
        用户：没别的方法了?
        客服：亲爱哒，虚拟订单一旦购买成功无法退回呢，请问您是否有将卡密截图提供给不法分子如还没有建议您可通过网址http://huishou.jd.com/card?cid=[数字]&pid=166168&skuId=[电话]查询是否有相关产品类型，可进行回收以此减少您的损失哦
        客服：亲亲，请问您是否有将卡密截图提供给不法分子?
        用户：这就是不法分子的卡密
        客服：如果[姓名]没有使用的话还请您登录上面的网址链接进行回收操作
        客服：如果提供了卡密虚拟订单一旦充值成功无法撤回呢，请您不要相信参与刷单，小妹这面建议您报警处理呢
        客服：请问还有其他还可以帮到您的吗?
        总结：

    Target: 用户拍错了想申请退单。客服回答虚拟商品交易成功之后无法退单。
    """

    description = "Summarization task in CLEVA benchmark"
    tags = ["summarizarion"]


class CLEVAMathematicalCalculationScenario(CLEVAScenario):
    """
    The mathematical calculation task of CLEVA benchmark.
    The datasets are modified from
    https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/modified_arithmetic.

    An example of two-digit addition is:
        在接下来的文本中，符号 -> 代表着一个简单的数学运算。

        677 + 89 -> 766

        678 + 246 ->

    Target: 924
    """

    description = "Mathematical calculation task in CLEVA benchmark."
    tags = ["mathematical_calculation"]


class CLEVAInductiveReasoningScenario(CLEVAScenario):
    """
    The inductive reasoning task of CLEVA benchmark.
    The datasets are modified from
    https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/modified_arithmetic.

    An example of two-digit substract with adding one is:
        在接下来的文本中，符号 -> 代表着一个简单的数学运算。

        935 - 927 -> 9

        921 - 385 ->

    Target: 537
    """

    description = "Inductive Reasoing task in CLEVA benchmark."
    tags = ["inductive_reasoning"]


class CLEVAReasoningPrimitiveScenario(CLEVAScenario):
    """
    The reasoning primitive task of CLEVA benchmark.
    We modify the following codes to construct the Chinese version.
        https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/scenarios/dyck_language_scenario.py
        https://github.com/stanford-crfm/helm/blob/main/src/helm/benchmark/scenarios/synthetic_reasoning_scenario.py


    An example of dyck_language is:
        下面是合法的dyck-n序列（只输出右括号）。

        ( { { ( { ( ) } ) }

    Target:  } )

    An example of pattern_induction is :
        给定两个从同一模式串生成的字符串，请推理出它们对应的模式串（模式串中仅包含变量X，Y，Z和符号+-*/）。

        字符串1：鹳 海豹 桃子 眼镜蛇 桃子 眼镜蛇 * - =
        字符串2：黑莓 马 马 * - =
        答：（输出任一一个合法的模式串即可）

    Target: Y Z Z * - =

    An example of pattern_matching is:
        给定一个结果串，请从4个模式串中找出对应的模式，并输出出来。

        结果串：+ 桃子 葡萄 +
        模式串：
        X Y + +
        X + Y +
        + X + Y
        + X Y +
        答：（输出对应的模式）

    Target: + X Y +

    An example of variable_sub is:
        请对模式串中的变量按照替换规则进行替换。

        模式：Z X X * - =
        替换规则：X -> “桃子 眼镜蛇”，Z -> “鹳 海豹”
        答：（替换后的结果）

    Target: 鹳 海豹 桃子 眼镜蛇 桃子 眼镜蛇 * - =
    """

    description = "Reasoning primitive task in CLEVA benchmark."
    tags = ["reasoning_primitive"]

    def process_instance(self, row: Dict[str, Any], split: str) -> Instance:
        text: str = row["text"]
        correct_answer = row["label"][:1] if split == "train" else row["label"]
        references = [Reference(Output(text=answer), tags=[CORRECT_TAG]) for answer in correct_answer]

        instance = Instance(
            input=Input(text=text),
            references=references,
            split=split,
        )
        return instance
