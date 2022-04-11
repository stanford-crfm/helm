import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from common.general import ensure_file_downloaded
from common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, CORRECT_TAG


@dataclass(frozen=True, eq=False)
class MSMARCOInstance(Instance):
    """Instance for the MSMARCO class.
    """

    # The query ID of the query included in the input.
    qid: Optional[int] = None

    # The passage ID of the passage included in the input.
    pid: Optional[int] = None

    # Whether the query and the passage included in the input are gold matches
    gold: Optional[bool] = None


class MSMARCOScenario(Scenario):
    """MS MARCO (Microsoft Machine Reading Comprehension) is a collection of
    datasets, based on the following research paper:

        https://arxiv.org/abs/1611.09268

    All the datasets can be retrieved at:

        https://microsoft.github.io/msmarco/

    The original dataset has 1,010,916 anonymized queries and "8,841,823
    passages extracted from 3,563,535 web documents retrieved by Bing". There
    are several tasks within the MS MARCO family, and each uses a variation
    of the aforementioned passage and query datasets.

    In our implementation, we are focusing on the Passage Retrieval task,
    which is an information retrieval task where the goal is to find the best
    passage that contains an answer to a given query. The evaluation set, which
    has 6980 queries, released with the task does not have the reference
    matches, so we use a subset of the development set as our evaluation set.

    We frame the passage retrieval task as a binary classification problem,
    similar to https://arxiv.org/pdf/2003.06713.pdf. Specifically, given a
    passage and a query, the model's job is to predict whether the passage
    includes an answer to the query by selecting one of the "yes" or "no"
    options. Shared below is an example of how a query with 4 context examples
    may look like.

        Passage: To access Data Import: 1  Sign in to Google Analytics. 2  Select the
        Admin tab and navigate to the property to which you want to upload the
        data. 3  Click Data Import. 4  This displays the Data Sets page.
        Question: Does the passage above answer the question effects of hydrogen
        combustion?
        A. Yes
        B. No
        Answer: B

        Passage: Sarcoidosis (sar-koy-DO-sis) is a disease of unknown cause that leads to
        inflammation. This disease affects your bodyâs organs. Normally, your
        immune system defends your body against foreign or harmful substances. For
        example, it sends special cells to protect organs that are in danger.
        Question: Does the passage above answer the question what causes sarcoidosis
        of the lungs?
        A. Yes
        B. No
        Answer: A

        Passage: Carbonic acid is a weak acid that is produced when carbon dioxide is dissolved
        in water. As you probably know, our atmosphere has a lot of carbon dioxide in
        it.It is also thoroughly saturated with water.From this, we might deduce that
        we live in a rather acidic environment â and we do.arbonic acid is a weak
        acid that is produced when carbon dioxide is dissolved in water. As you probably
        know, our atmosphere has a lot of carbon dioxide in it. It is also thoroughly
        saturated with water. From this, we might deduce that we live in a rather acidic
        environment â and we do.
        Question: Does the passage above answer the question what is a affidavit of support?
        A. Yes
        B. No
        Answer: B

        Passage: One of the FHAâs primary criteria is whether or not youâve owned a home.
        If youâve never owned a home, youâre considered a first-time homebuyer.
        But you are allowed to be a previous homeowner and still qualify as a first-time
        homebuyer. According to the FHA, you can do so if you have not been an owner in a
        primary residence for at least three years leading up to your purchase.
        Question: Does the passage above answer the question what is considered first
        time home buyer?
        A. Yes
        B. No
        Answer: A

        Passage: http://en.wikipedia.org/wiki/William_Bradford_(Plymouth_Colony_governor) William
        Bradford (c.1590 â 1657) was an English Separatist leader in Leiden, Holland
        and in Plymouth Colony was a signatory to the Mayflower Compact. He served as
        Plymouth Colony Governor five times covering about thirty years between 1621 and 1657.
        Question: Does the passage above answer the question how many years did william
        bradford serve as governor of plymouth colony?
        A. Yes
        B. No
        Answer:

    For each query, we assign a ranking to each passage that we queried the model with
    as follows:
        - We get the model's answer, "Yes" or "No", and the logprob of the answer
            for each passage.
        - We rank the answers we got using the following scheme:
            High => "Yes", high logprob
                 => "Yes", low  logprob
                 => "No",  low  logprob
            Low  => "No",  high logprob

    Once we have a ranked list of passages for a query, we compute MRR@10,
    which is the mean reciprocal rank of the gold passage when we only
    consider the top 10 passages.

    Below are some details on the datasets we use, which can all be retrieved
    at the link below, pointing to a 1GB tar file. Here, "qid" stands for
    "Query ID" and "pid" stands for "Passage ID". FORMAT column specifies the
    contents of each file, where \t is used as the delimeter character.

        https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz

                  FILE          |            INFO           |      FORMAT
        `collection.tsv`        | 8,841,823 passages        | <pid> <passage text>
        `qrels.dev.small.tsv`   | 7437      query relations | <qid> 0 <pid> 1
        `qrels.train.tsv`       | 532,761   query relations | <qid> 0 <pid> 1
        `queries.dev.small.tsv` | 6980      queries         | <qid> <query text>
        `queries.train.tsv`     | 808,731   queries         | <qid> <query text>

    `qrels` files contain the query relations, mapping each
    query (with the ID qid) to a ground truth passage match (with the ID pid).
    Note that there are more matches than the number of queries: this
    happens because the `qrels` file sometimes contain 2 best passage matches
    for a query.

    We also utilize two custom generated files, `top1000_bm25_dev.tsv` (133 MB)
    and `top20_bm25_train.tsv`. These files contain the top 1000 and 20 best
    passage id matches for a given query in the dev or train set, respectively.
    We have generated these files using the BM25 algorithm. Both of these files
    as well as the notebook including our file generation code can be found at
    the following Codalab link:

        https://worksheets.codalab.org/worksheets/0xf451c0dec2a6414aae0b68e8e325426c

    The topk files have the following format, where rank is a number between
    1 and 1000:

        <qid> <pid> <rank>

    For details on how we create the instances, refer to the docs of the
    `get_instances` method.

    For details on how we evaluate our results, please refer to the
    `MSMARCOMetric` class in `msmarco_metric.py`.
    """

    """ Information on this class """
    name = "msmarco"
    description = "Microsoft Machine Reading Comprehension"
    tags = ["information_retrieval"]

    """ Class variables """
    # Names of the tasks we support
    TASK_NAMES: List[str] = ["passage"]

    # The filename of the top1000 file created with the BM25 algorithm
    TOPK_DEV_FILE_NAME: str = "top1000_bm25.dev.tsv"
    TOPK_TRAIN_FILE_NAME: str = "top20_bm25.train.tsv"

    # The base URL for the MSMARCO datasets
    MSMARCO_URL: str = "https://msmarco.blob.core.windows.net/msmarcoranking"

    # Codalab retrieval information
    # Codalab URL format
    CODALAB_URL: str = "https://worksheets.codalab.org/rest/bundles/{bundle}/contents/blob/"

    # Codalab dev url
    CODALAB_DEV_BUNDLE: str = "0x004852a9a16d4a99851b6151a1972d36"
    CODALAB_DEV_URL: str = CODALAB_URL.format(bundle=CODALAB_DEV_BUNDLE)

    # Codalab train url
    CODALAB_TRAIN_BUNDLE: str = "0x499c07699f3f4881a787b6a5249f4466"
    CODALAB_TRAIN_URL: str = CODALAB_URL.format(bundle=CODALAB_TRAIN_BUNDLE)

    # The maximum number of queries that we can run the scenario for.
    #   Eval queries capped at 6980 since that is the size of the dev set we use.
    #   Note that each eval query results in multiple instances.
    #   Train queries capped at 808731 as that's the size of the train set.
    MAX_NUM_EVAL_QUERIES = 6980
    MAX_NUM_TRAIN_QUERIES = 808731

    # Upper and lower bounds on topk, the number of top passages that we will
    #   consider for a given query.
    #   - Capped at 1000 because our pre-generated topk file (TOP1000_DEV_FILE_NAME)
    #       only contains the top 1000 passage ids per dev query.
    #   - Topk should at least be 11 as our default metric is MRR@10.
    #       We have 1 gold instance for each query where we have the matching passage.
    #       We must have 9 non-matching instances to ensure that there are 10 total instances.
    #       There can be up to 2 gold queries in the top 11 passage list for a given query.
    #       This means that we can get at least 9 non-matching instances from the top 11 list.
    MAX_TOPK: int = 1000
    MIN_TOPK: int = 11

    # The minumum rank we will accept for the no instances.
    #   This is to ensure that when creating the no instances, we do not consider the
    #   First several ranks in our train topk list, which may contain passages similar
    #   to the gold passages.
    TRAIN_MIN_NO_INSTANCE_RANK: int = 11

    # Yes and no answer strings
    YES_ANSWER = "Yes"
    NO_ANSWER = "No"

    def __init__(self, task: str, topk: int = 30, num_eval_queries: int = 100, num_train_queries: int = 1000):
        """MSMARCOScenario class constructor.

        Both outlined below, `topk` and `num_eval_queries` have a direct impact
        on the number of tokens used by this scenario, given a specific `task`.

        For the Passage Retrieval task, you can find the total number of tokens
        needed as follows:

            num_no_examples_per_query = topk, or topk - 1, or topk - 2
            total_num_eval_instances = num_eval_queries * (1 + num_no_examples_per_query)
            total_num_tokens = total_num_eval_instances * (1 + NUM_CONTEXT_EXAMPLES) * AVG_TOKEN_LENGTH

        In the above formulation:
            - NUM_CONTEXT_EXAMPLES corresponds to the number of training
                examples we add to the context of each request.
            - AVG_TOKEN_LENGTH is the average token length of one instance, which
                is about 535 tokens on average.

        Args:
            task: Name of the task, should be one of self.TASK_NAMES. There are
                several MSMARCO tasks, and we use the task parameter to specify
                which task we would like performed. There is only one task that
                is implemented for the time being: the Passage Retrieval task.
            topk: To find the best passage match for a given validation query,
                instead of going through all the passages in the collection, we
                only look at a select number of filtered passages, which is
                determined by `topk`. Must be in the range
                (self.MIN_TOPK, self.MAX_TOPK].
            num_eval_queries: Number of evaluation queries that is used to
                create eval instances. Must be smaller than or equal to
                self.MAX_NUM_EVAL_QUERIES. The total number of evaluation
                instances created is a function of this number:

                    num_no_examples_per_query = topk, or topk - 1, or topk - 2
                    total_num_eval_instances = num_eval_queries * (1 + num_no_examples_per_query)
            num_train_queries: Number of train queries that is used to crete
                the train instances. Must be smaller than or equal to
                self.MAX_NUM_TRAIN_QUERIES. The total number of training instances
                created is a function of this number:

                    num_no_examples_per_query = 1
                    total_num_train_instances = num_train_queries * (1 + num_no_examples_per_query)
        """
        # Task
        self.task: str = task
        assert self.task in self.TASK_NAMES

        # TopK
        if topk < self.MIN_TOPK or topk > self.MAX_TOPK:
            msg = f"Number of passages ranked should be between {self.MIN_TOPK} and {self.MAX_TOPK} (both inclusive)."
            raise ValueError(msg)
        self.topk = topk

        # num_eval_queries
        if num_eval_queries > self.MAX_NUM_EVAL_QUERIES:
            msg = f"Number of evaluation queries should not be bigger than {self.MAX_NUM_EVAL_QUERIES}."
            raise ValueError(msg)

        # num_train_queries
        if num_train_queries > self.MAX_NUM_TRAIN_QUERIES:
            msg = f"Number of train queries should not be bigger than {self.MAX_NUM_TRAIN_QUERIES}."
            raise ValueError(msg)

        # Set num queries
        self.num_queries = {VALID_SPLIT: num_eval_queries, TRAIN_SPLIT: num_train_queries}

        # Initialize the data dictionaries that will be populated once the MSMARCO scenario is run
        self.collection_dict: Dict[int, str] = {}
        self.queries_dicts: Dict[str, Dict[int, str]] = {}
        self.qrels_dicts: Dict[str, Dict[int, List[int]]] = {}
        self.topk_dicts: Dict[str, Dict[int, Dict[int, int]]] = {}

    def download_file(
        self, source_url: str, file_name: str, unpack: bool = False, unpack_type: Optional[str] = None
    ) -> str:
        """Downloads a file.

        Writes the file in the given source_url a file with the name file_name
        in the /data directory in located in the self.output_path.
        """
        file_path: str = os.path.join(self.output_path, "data", file_name)
        ensure_file_downloaded(source_url=source_url, target_path=file_path, unpack=unpack, unpack_type=unpack_type)
        return file_path

    @staticmethod
    def create_id_item_dictionary(file_path: str) -> Dict[int, str]:
        """Reads .tsv files in the following format into Python dictionaries:

            <id>\t<text>

        For example, if the file contents look like:

            1   this is the first example
            2   this is the second example
            3   this is the third example

        The dictionary returned would be as follows:
            {
                1: "this is the first example",
                2: "this is the second example",
                3: "this is the third example"
            }

        Returns:
            id_to_item_dict: Dictionary mapping the id of an item to the item.
        """
        id_to_item_dict = {}
        with open(file_path, encoding="utf-8") as f:
            for _id, content in csv.reader(f, delimiter="\t"):
                id_to_item_dict[int(_id)] = content
        return id_to_item_dict

    @staticmethod
    def create_qrels_dictionary(file_path: str) -> Dict[int, List[int]]:
        """Reads .tsv files in the following format into a Python dictionary:

            <qid>\t0\t<pid>\t1

        The <qid> and <pid> co-occuring in a line means that the pid is the id
        of a gold passage for the query with the id qid. Note that some qids
        have 2 pids matched to them as the gold pids, which is why we are
        returning a dictionary mapping a qid to a list of pids (instead of just
        a pid).

        For example, if the file contents look like:

            11111111   0    12837901     1
            11111111   0    82374921     1
            22222222   0    28192830     1
            ...


        The dictionary returned would be as follows:
            {
                11111111: [12837901, 82374921],
                22222222: [28192830]
            }

        Returns:
            qrels_dict: Dictionary mapping a qid to a list containing the gold
                pids.
        """
        dictionary = defaultdict(list)
        with open(file_path, encoding="utf-8") as f:
            for qid, a, pid, b in csv.reader(f, delimiter="\t"):
                dictionary[int(qid)].append(int(pid))
        return dictionary

    @staticmethod
    def create_topk_dictionary(file_path: str) -> Dict[int, Dict[int, int]]:
        """Reads .tsv files in the following format into a Python dictionary:

            <qid>\t<pid>\t<rank>

        For example, if the file contents look like:

            11111111   12837901     1
            11111111   82374921     2
            11111111   28192830     3
            ...
            11111111   28191237     1000
            22222222   98021301     1
            22222222   21938912     2
            22222222   12938010     3
            ...
            22222222   32409810     1000

        The dictionary returned would be as follows:
            {
                11111111: {
                    1: 12837901,
                    2: 82374921,
                    3: 28192830,
                    ...
                    1000: 28191237
                },
                22222222: {
                    1: 98021301,
                    2: 21938912,
                    3: 12938010,
                    ...
                    1000: 32409810
                }
            }

        Returns:
            topk_dictionary: Dictionary mapping a qid to a dictionary mapping
                ranks to a pid.
        """
        topk_dict: Dict[int, Dict[int, int]] = defaultdict(dict)
        with open(file_path, encoding="utf-8") as f:
            for qid, pid, rank in csv.reader(f, delimiter="\t"):
                topk_dict[int(qid)][int(rank)] = int(pid)
        return topk_dict

    def prepare_passage_dictionaries(self):
        """Downloads the Passage Retrieval datasets and reads them into dictionaries.

        Sets the following:
            self.collection_dict: Mapping pid to passage.
            self.queries_dicts: Dictionary containing query dictionaries mapping a
                qid to a query.

                {
                    VALID_SPLIT: valid_query_dict,
                    TRAIN_SPLIT: train_query_dict
                }
            self.qrels_dicts: Dictionary containing qrels dictionaries mapping a
                qid to a list of gold pids. Refer to
                self.create_qrels_dictionary for the exact format of the sub
                dictionaries.

                {
                    VALID_SPLIT: valid_qrels_dict,
                    TRAIN_SPLIT: train_qrels_dict
                }
        """
        hlog("Downloading MSMARCO Passage Retrieval datasets.")

        # Get datasets
        dir_path = self.download_file(
            f"{self.MSMARCO_URL}/collectionandqueries.tar.gz", "collectionandqueries", unpack=True, unpack_type="untar"
        )
        hlog("Download was successful. Reading the data into dictionaries.")

        # Collection
        self.collection_dict = self.create_id_item_dictionary(os.path.join(dir_path, "collection.tsv"))

        # Queries
        self.queries_dicts = {
            TRAIN_SPLIT: self.create_id_item_dictionary(os.path.join(dir_path, "queries.train.tsv")),
            VALID_SPLIT: self.create_id_item_dictionary(os.path.join(dir_path, "queries.dev.small.tsv")),
        }

        # Query relations
        self.qrels_dicts = {
            TRAIN_SPLIT: self.create_qrels_dictionary(os.path.join(dir_path, "qrels.train.tsv")),
            VALID_SPLIT: self.create_qrels_dictionary(os.path.join(dir_path, "qrels.dev.small.tsv")),
        }

    def prepare_topk_dictionaries(self):
        """Downloads the topk files and reads them into dictionaries.

        Sets the following field:
            self.topk_dicts: Dictionary containing topk dictionaries mapping a
                qid to a dictionary mapping a rank to a pid. Refer to
                self.create_topk_dict for the exact format of the sub
                dictionaries.

                {
                    VALID_SPLIT: valid_topk_dict,
                    TRAIN_SPLIT: train_topk_dict
                }
        """
        hlog("Downloading topk files.")

        # Get files
        topk_dev_fp = self.download_file(self.CODALAB_DEV_URL, self.TOPK_DEV_FILE_NAME)
        topk_train_fp = self.download_file(self.CODALAB_TRAIN_URL, self.TOPK_TRAIN_FILE_NAME)
        self.topk_dicts = {
            VALID_SPLIT: self.create_topk_dictionary(topk_dev_fp),
            TRAIN_SPLIT: self.create_topk_dictionary(topk_train_fp),
        }

    @staticmethod
    def make_context(passage: str, query: str) -> str:
        """Makes the context text given a passage and a query.
        """
        # Remove a question mark at the end of the query, if there is any
        if query[-1] == "?":
            query = query[:-1]
        question_statement = f"Does the passage above answer the question {query}?"
        return f"{passage}\nQuestion: {question_statement}"

    def get_instance(self, qid: int, pid: int, split: str, gold: bool = False,) -> Instance:
        """Creates an instance.

        Args:
            qid: Query id.
            pid: Passage id.
            split: TRAIN_SPLIT or VALID_SPLIT.
            gold: Whether the instance to be created contains the gold pid.

        Returns:
            instance: Created instances.
        """
        query = self.queries_dicts[split][qid]
        passage = self.collection_dict[pid]
        context = self.make_context(passage, query)
        references = [
            Reference(output=self.YES_ANSWER, tags=[CORRECT_TAG] if gold else []),
            Reference(output=self.NO_ANSWER, tags=[] if gold else [CORRECT_TAG]),
        ]
        instance = MSMARCOInstance(input=context, references=references, split=split, qid=qid, pid=pid, gold=gold)
        return instance

    def get_passage_split_instances(self, split) -> List[Instance]:
        """Creates instances for the specified split.

        For the number of queries specified for each split, we loop through the
            query list. For each query:
            - We create a "yes" instance, where the included passage is the gold
                passage for the given query.
            - We then create a set of "no" instances by going through the topk
                passage list for the query. We select all the passages that are
                not in the gold passages list for the query.

                We limit the number of no examples for the train split to be 1 to
                ensure that we have a balanced train split.

                We do not consider the first several ranks for the train queries to
                ensure that the no examples we include in the train split are not the
                highly ranked false positives.

        Args:
            split: VALID_SPLIT or TRAIN_SPLIT.

        Returns:
            instances: List of instances created.
        """
        # Only use the first num_queries queries, specified in the constructor
        qrels_keys = list(self.qrels_dicts[split].keys())[: self.num_queries[split]]
        qrels_dict = {k: self.qrels_dicts[split][k] for k in qrels_keys}

        # List of ranks we will consider for the no instances.
        #   By default, we consider all the ranks up to and including self.topk
        # For the train set, we start the no instance ranks at self.TRAIN_MIN_NO_INSTANCE_RANK
        #   to ensure we don't include passages that are good potentials for the gold matches.
        no_instance_ranks = list(range(1, self.topk + 1))
        if split == TRAIN_SPLIT:
            no_instance_ranks = list(range(self.TRAIN_MIN_NO_INSTANCE_RANK, self.topk + 1))

        instances = []
        for qid, gold_pids in qrels_dict.items():

            # Generate the yes instance
            gold_pid = gold_pids[0]
            yes_instance = self.get_instance(qid, gold_pid, split, gold=True)

            # Generate the no instances
            rank_dict = self.topk_dicts[split][qid]
            pids = [rank_dict[rank] for rank in no_instance_ranks if rank_dict[rank] not in gold_pids]
            no_instances = [self.get_instance(qid, pid, split) for pid in pids]

            # Limit the no_instances to 1 for the train split to ensure that we have a balanced train set.
            # Otherwise, there will be many more no instances in the train split than the yes instances.
            if split == TRAIN_SPLIT and no_instances:
                no_instances = [no_instances[0]]

            # Extend the instances
            instances += [yes_instance] + no_instances

        return instances

    def get_passage_instances(self) -> List[Instance]:
        """Gets instances for the passage task.
        """
        # Get dataset and topk dictionaries
        self.prepare_passage_dictionaries()
        self.prepare_topk_dictionaries()

        # Create instances
        valid_instances = self.get_passage_split_instances(VALID_SPLIT)
        train_instances = self.get_passage_split_instances(TRAIN_SPLIT)
        instances = valid_instances + train_instances

        return instances

    def get_instances(self) -> List[Instance]:
        """Gets instances for the MSMARCO class.

        Supported tasks and the corresponding method called to get instances:
            "passage": self.get_passage_instances()

        Refer to the documentation of the methods above for details on how the
        instances are created.
        """
        if self.task in self.TASK_NAMES:
            return self.get_passage_instances()
        raise ValueError(f"Task must be one of {', '.join(self.TASK_NAMES)}")
