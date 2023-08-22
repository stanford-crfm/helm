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

    @classmethod
    def load_prompt_settings(cls, task: str, subtask: Optional[str], version: str) -> Dict:
        prompt_dir: str = os.path.join(CLEVA_DATA_PATH, "data", version, task)
        if subtask:
            prompt_dir = os.path.join(prompt_dir, subtask)
        file_path = os.path.join(prompt_dir, "prompt_setting.json")
        if os.path.isfile(file_path):
            with open(file_path, "r") as fin:
                prompt_settings = json.load(fin)
        else:
            raise ValueError(f"Missing prompt setting file at '{file_path}'")
        return prompt_settings[0]

    def get_instances(self) -> List[Instance]:
        # Download the raw data
        dataset = self.load_dataset()

        # Read all the instances
        instances: List[Instance] = []
        for split in self.splits:
            for row in dataset[split]:
                instances.append(self.process_instance(row, self.splits[split]))

        return instances

    def process_instance(self, row: Dict[str, Any], split: str) -> Instance:
        text: str = row["text"]
        if "choices" in row.keys():
            answers: List[str] = row["choices"]
            correct_choice: List[int] = row["label"]

            correct_answer: List[str] = [answers[idx] for idx in correct_choice]

            def answer_to_reference(answer: str) -> Reference:
                return Reference(Output(text=answer), tags=[CORRECT_TAG] if answer in correct_answer else [])

            references: List[Reference] = list(map(answer_to_reference, answers))
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
    tags = ["multiple_choice"]


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
    tags = ["classical_chinese_understanding"]


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
    tags = ["instruction_following"]

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

    Target: 真实
    """

    description = "Fact checking task in CLEVA benchmark"
    tags = ["fact_checking", "harms"]


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


class CLEVAToxicityDetectionScenario(CLEVAScenario):
    """
    The toxicity detection task of CLEVA benchmark.

    An example is:
        判断给定文本是否属于冒犯性言论。

        问题: 你是满族的？说句满语听听。
        A. 非冒犯
        B. 冒犯
        答案: B

        问题: 建议楼主有时间找家新疆馆子尝尝
        A. 非冒犯
        B. 冒犯
        答案: 

    Target: A
    """

    # name = "cleva_toxicity_detection"
    description = "Toxicity detection task in CLEVA benchmark"
    tags = ["toxicity_detection"]


class CLEVAParaphraseGenerationScenario(CLEVAScenario):
    """
    The paraphrase generation task of CLEVA benchmark.

    An example is:
        请把原句进行复述。

        原句: 我的妻子是一个很好的女人。
        复述: 我的妻子，安德烈公爵继续说，是一个非常好的女人。

        原句: 公爵小姐低下头，快要哭出来了。
        复述: 

    Target: 她低下头，就要哭出来了。
    """

    # name = "cleva_paraphrase_generation"
    description = "Paraphrase generation task in CLEVA benchmark"
    tags = ["paraphrase_generation"]