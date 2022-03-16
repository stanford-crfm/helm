import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from common.general import ensure_file_downloaded
from common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, Reference, TRAIN_SPLIT, VALID_SPLIT, CORRECT_TAG


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
    passage that contains the answer to a given query. Evaluation set, which 
    has 6980 queries, released with the task do not have the reference 
    matches, so we use a subset of the development set as our evaluation set.

    Here are some examples of how you would run this task:

        venv/bin/benchmark-run -r msmarco:task=passage --dry-run
        venv/bin/benchmark-run -r msmarco:task=passage
        venv/bin/benchmark-run -r msmarco:task=passage&num_eval_queries=10
        venv/bin/benchmark-run -r msmarco:task=passage&num_eval_queries=10&topk=20
    
    Below are some details on the datasets we use, which can all be retrieved 
    at the link below, which points to a 1GB tar file. Here, "qid" stands for 
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
    Note that there are more matches than then the number of queries: this
    happens because the `qrels` file sometimes contain 2 best passage matches
    for a query.

    We also utilize two custom generated files: `top1000.dev.tsv` (133 MB) and 
    `top1000.train.tsv` (10 GB). These files contain the top 1000 best 
    passage id matches for a given query id. We generate these files using the 
    following `Colab` document: 
    
        https://colab.research.google.com/drive/1BX-YRUY7H5IFHF-6iG-lH_Xc7qg46Bp1?usp=sharing
    
    The top 1000 files have the following format, where rank is a number 
    between 1 and 1000:

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

    # The base URL for the MSMARCO datasets
    MSMARCO_URL: str = "https://msmarco.blob.core.windows.net/msmarcoranking"

    # The filenames of the top1000 files created with the BM25 algorithm
    #   stored in the /data directory in self.input_path
    # TODO START Change the following two lines once we have the final files.
    TOP1000_TRAIN_FILE_NAME: str = 'top750_bm25_dev.tsv'
    TOP1000_DEV_FILE_NAME: str = 'top750_bm25_dev.tsv'
    # TODO END

    # The maximum number of evaluation queries we will have.
    #   Note that each evaluation query results in multiple instances.
    MAX_NUM_EVAL_QUERIES = 50

    # Upper and lower bounds on topk, the number of top passages that we will 
    #   consider for a given query. 
    MAX_TOPK: int = 50
    MIN_TOPK: int = 11

    # Number of example context instances
    NUM_CONTEXT_INSTANCES = 4

    def __init__(self, task: str, num_eval_queries: int = 2, topk: int = 20):
        """MSMARCOScenario class constructor.

        Both outlined below, `topk` and `num_eval_queries` have a direct impact 
        on the number of tokens used by this scenario, given a specific `task`.

        For the Passage Retrieval task, you can find the total number of tokens
        needed as follows:

            total_num_eval_instances = num_eval_queries * (1 + (topk [-1 or -2]))
            total_num_tokens = total_num_eval_instances * (1 + NUM_CONTEXT_EXAMPLES) * AVG_TOKEN_LENGTH

        In the above formulation:
            - NUM_CONTEXT_EXAMPLES correspond to the number of training
                examples we add to the context of each request.
            - AVG_TOKEN_LENGTH is the average token length of one instance, which
                is about ~496 for our task when we use 4 in context examples.

        Args:
            task: Name of the task, should be one of self.TASK_NAMES. There are
                several MSMARCO tasks, and we use the task parameter to specify 
                which task we would like performed. There is only one task that
                is implemented for the time being: the Passage Retrieval task.
            num_eval_queries: Number of evaluation queries that we will use to 
                create eval instances. Must be smaller than or equal to 
                self.MAX_NUM_EVAL_QUERIES. The total number of evaluation 
                instances created is a function of this number: 

                    total_num_eval_instances = num_eval_queries * (1 + (topk [-1 or -2]))
            topk: To find the best passage match for a given query, instead of 
                going through all the passages in the collection, we only look 
                at a select number of filtered passages, which is determined 
                by `topk`. Must be in the range (self.MIN_TOPK, self.MAX_TOPK].

                The passages we look at our filtered as follows:
                - VALID_SPLIT: For the validation examples, we took the 
                    official dev set and for each query we ran the BM25 
                    algorithm on all the passages in the official collections
                    set to get the top 1000 passage ids for each query id.
                    This file is stored locally, and we use the `topk` passed
                    in to limit our search space using the language model.
                - TRAIN_SPLIT: We use the official top 1000 file released along
                    with the MSMARCO challenge. Similar to the dev set, the
                    `topk` number is used to filter this file further.
        """

        # Task
        self.task: str = task
        assert self.task in self.TASK_NAMES

        # TopK
        if topk < self.MIN_TOPK or topk > self.MAX_TOPK:
            msg = f"Number of passages ranked should be between {self.MIN_TOPK} and {self.MAX_TOPK} (inclusive) to \
                  conserve tokens."
            raise ValueError(msg)
        self.topk = topk     

        # num_eval_queries   
        if num_eval_queries > self.MAX_NUM_EVAL_QUERIES:
            msg = f"Number of evaluation queries should not be bigger than {self.MAX_NUM_EVAL_QUERIES} to \
                  conserve tokens."
            raise ValueError(msg)
        self.num_eval_queries = num_eval_queries

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
    def create_topk_dictionary(file_path) -> Dict[int, Dict[int, int]]:
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
        topk_dict = defaultdict(dict)
        with open(file_path, encoding="utf-8") as f:
            for qid, pid, rank in csv.reader(f, delimiter="\t"):
                topk_dict[int(qid)][int(rank)] = int(pid)
        return topk_dict

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

    def download_file(self, source_url: str, file_name: str,
                      unpack: bool = False, unpack_type: Optional[str] = None) -> str:
        """Downloads a file.
        
        Writes the file in the given source_url a file with the name file_name 
        in the /data directory in located in the self.output_path.
        """
        file_path: str = os.path.join(self.output_path, "data", file_name)
        ensure_file_downloaded(source_url=source_url, target_path=file_path, unpack=unpack, unpack_type=unpack_type)
        return file_path

    def prepare_passage_dataset_dictionaries(self) -> Tuple[
                                                            Dict[int, str],
                                                            Dict[str, Dict[int, str]],
                                                            Dict[str, Dict[int, List[int]]]
                                                        ]:
        """Downloads the Passage Retrieval datasets and reads them into dicts.

        Returns:
            collection_dict: Mapping pid to passage.
            queries_dicts: Dictionary containing query dictionaries mapping a 
                qid to a query.

                {
                    VALID_SPLIT: valid_query_dict,
                    TRAIN_SPLIT: train_query_dict
                }

            qrels_dicts: Dictionary containing qrels dictionaries mapping a 
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
        dir_path = self.download_file(f"{self.MSMARCO_URL}/collectionandqueries.tar.gz", "collectionandqueries",
                                      unpack=True, unpack_type="untar")
        hlog("Download was successful. Reading the data into dictionaries.")
        
        # Collection
        collection_dict = self.create_id_item_dictionary(os.path.join(dir_path, 'collection.tsv'))
       
        # Queries
        queries_dicts = {
            TRAIN_SPLIT: self.create_id_item_dictionary(os.path.join(dir_path, 'queries.train.tsv')),
            VALID_SPLIT: self.create_id_item_dictionary(os.path.join(dir_path, 'queries.dev.small.tsv'))
        }

        # Query relations
        qrels_dicts = {
            TRAIN_SPLIT: self.create_qrels_dictionary(os.path.join(dir_path, 'qrels.train.tsv')),
            VALID_SPLIT: self.create_qrels_dictionary(os.path.join(dir_path, 'qrels.dev.small.tsv'))
        }

        # TODO START: Remove the following lines once we add the train files.
        queries_dicts[TRAIN_SPLIT] = queries_dicts[VALID_SPLIT]
        qrels_dicts[TRAIN_SPLIT] = qrels_dicts[VALID_SPLIT]
        # TODO END

        return collection_dict, queries_dicts, qrels_dicts

    def prepare_passage_topk_dictionaries(self) -> Dict[str, Dict[int, Dict[int, int]]]:
        """Prepares the topk files for the MSMARCO Passage Retrieval task.
        
        Returns:
            topk_dicts: Dictionary containing topk dictionaries mapping a 
                qid to dictionary mapping a rank to a pid. Refer to 
                self.create_topk_dictionary for the exact format of the sub 
                dictionaries.

                {
                    VALID_SPLIT: valid_topk_dict,
                    TRAIN_SPLIT: train_topk_dict
                }
        """
        hlog("Preparing the topk dictionaries.")

        # Create topk dictionaries
        topk_dicts = {}

        # Set the input directory, in which we store the custom generated top1000.dev.tsv
        input_path = self.output_path.replace('output', 'input')

        # TRAIN topk
        top1000_train_fp = os.path.join(input_path, "data", self.TOP1000_TRAIN_FILE_NAME)
        topk_dicts[TRAIN_SPLIT] = self.create_topk_dictionary(top1000_train_fp)

        # VALID topk
        top1000_dev_fp = os.path.join(input_path, "data", self.TOP1000_DEV_FILE_NAME)
        topk_dicts[VALID_SPLIT] = self.create_topk_dictionary(top1000_dev_fp)

        return topk_dicts

    @staticmethod
    def make_context(passage: str, query: str) -> str:
        """Makes the context text given a passage and a query.
        """
        # Remove a question mark at the end of the query, if there is any
        if query[-1] == '?':
            query = query[:-1]
        question_statement = f"Does the passage above answer the question {query}?"
        return f"{passage}\nQuestion: {question_statement}"

    @staticmethod
    def create_instance_id(qid: str, pid: str, gold=False):
        """Creates a unique id for an instance.
        """
        return f"qid_{qid}-pid_{pid}-gold_{gold}"
    
    def create_instance(self, passage: str, query: str, instance_id: str, split: str,
                        yes_tags: Optional[List] = None, no_tags: Optional[List] = None) -> Instance:
        """Creates an instances.

        Args:
            passage: The passage text to be included.
            query: The query text to be included.
            instance_id: ID to differentiate this instance from the other 
                instances.
            split: `TRAIN_SPLIT` or `VALID_SPLIT`.
            yes_tags: Tags to be assigned to the yes reference. Defaults to [].
                Should be `[CORRECT_TAG]` if the yes answer is the correct 
                answer.
            no_tags: Version of `yes_tags` for the no reference.

        Returns:
            instance: Instance created with the given parameters.
        """
        # Prepare the answers
        yes_answer = "yes"
        no_answer = "no"

        # Prepare the instance
        context = self.make_context(passage, query)
        references = [
            Reference(output=yes_answer, tags=yes_tags if yes_tags else []),
            Reference(output=no_answer, tags=no_tags if no_tags else [])
        ]
        instance = Instance(input=context, references=references, split=split, id=instance_id)

        return instance

    def create_split_instances(self, split: str, collection_dict: Dict[int, str], queries_dict: Dict[int, str],
                               qrels_dict: Dict[int, List[int]], topk_dict: Dict[int, Dict[int, int]],
                               num_queries: int, min_rank: int, max_rank: int) -> List[Instance]:
        """Creates instances for the specified split.

        Args:
            split: `TRAIN_SPLIT` or `VALID_SPLIT`.
            collection_dict: Dictonary mapping pid to passages.
            queries_dict: Dictionary mapping qid to queries.
            qrels_dict: Dictionary mapping qid to list of gold pids.
            topk_dict: Dictionary mapping qid to a dictionary mapping topk rank
                to pid.
            num_queries: Number of queries we want to generate the instances 
                for.
            min_rank: Minumum rank we will include for the no examples.
            max_rank: Maximum rank we will include for the no examples.

        Returns:
            instances: List of instances created.
        """
        instances = []
        for ind, (qid, gold_pids) in enumerate(qrels_dict.items()):

            # Only continue generating instances if we haven't hit the desired number of queries
            if ind < num_queries:

                # Get query
                query = queries_dict[qid]
                
                # Yes
                gold_pid = gold_pids[0]
                passage = collection_dict[gold_pid]
                instance_id = self.create_instance_id(qid, gold_pid, gold=True)
                instance = self.create_instance(passage, query, instance_id, split, yes_tags=[CORRECT_TAG])
                instances.append(instance)

                hlog(f"Yes instance {ind+1} =>")
                hlog(f"\tsplit: {split}, qid: {qid}, instance_id: {instance_id}, "
                     f"\tgold_pid: {gold_pid}, gold_pids: {gold_pids}, "
                     f"\tCondition == ind: {ind} < num_queries: {num_queries}")

                # No
                for ind2, (rank, pid) in enumerate(topk_dict[qid].items()):
                    if min_rank <= rank <= max_rank and pid not in gold_pids:
                        passage = collection_dict[pid]
                        instance_id = self.create_instance_id(qid, pid, gold=False)
                        instance = self.create_instance(passage, query, instance_id, split, no_tags=[CORRECT_TAG])
                        instances.append(instance)

                        hlog(f"No instance {ind2+1} =>")
                        hlog(f"\tsplit: {split}, qid: {qid}, instance_id: {instance_id}, "
                             f"\tpid: {pid}, gold_pids: {gold_pids}, "
                             f"\tCondition == pid not in gold_pids: {pid not in gold_pids}")

        return instances

    def get_passage_instances(self) -> List[Instance]:
        """Gets instances for the passage task. 

        VALID_SPLIT instances are generated as follows: For 
            `self.num_eval_instances` number of queries, do the following:

            - We first create a "yes" instance, where the included passage is 
                the gold passage for the given query.
            - We then create `self.topk` [optionally topk-1 or topk-2] "no" 
                instances where the included passage does not contain an answer 
                to the given query. Sometimes the topk examples include the 
                1 or 2 gold passages, which is why we are optionally 
                subtracting them.
        
        TRAIN_SPLIT instances are generated as follows: For 
            `self.num_eval_instances` * `self.NUM_CONTEXT_INSTANCES` many 
            queries, do the following:

            - We first create a "yes" instance, where the included passage is 
                the gold passage for the given query.
            - We then create at most 1 "no" instance where the included passage 
                does not contain an answer to the given query. We limit the 
                no examples to 1 to have a balanced training set.

        Example of a "yes" instance:

            McDonald's Corporation is one of the most recognizable corporations 
            in the world. A corporation is a company or group of people 
            authorized to act as a single entity (legally a person) and 
            recognized as such in law. Early incorporated entities were 
            established by charter (i.e. by an ad hoc act granted by a monarch 
            or passed by a parliament or legislature).
            Question: Does the passage above answer the question. what is a 
            corporation?
            A. yes
            B. no
            Answer: A
        
        Example of a "no" instance:

            Average physician assistant salary. Physician assistantâs
            salary is ranging from $68,587 to $117,554 pay per year. The average
            physician assistantâs salary is $87,749. Generally, a new
            physician assistant earns an hourly pay ranging from $28.13 to 50.00.
            Read more about average PA salary in USA.
            Question: Does the passage above answer the question do physicians
            pay for insurance from their salaries?
            A. yes
            B. no
            Answer: B
        """
        # Get dataset dictionaries
        collection_dict, queries_dicts, qrels_dicts = self.prepare_passage_dataset_dictionaries()
        
        # Get topk dictionaries
        topk_dicts = self.prepare_passage_topk_dictionaries()

        # Create VALID_SPLIT instances
        num_queries = self.num_eval_queries
        min_rank, max_rank = 1, self.topk
        dev_instances = self.create_split_instances(VALID_SPLIT, collection_dict, queries_dicts[VALID_SPLIT],
                                                    qrels_dicts[VALID_SPLIT], topk_dicts[VALID_SPLIT],
                                                    num_queries=num_queries, min_rank=min_rank, max_rank=max_rank)

        # Create TRAIN_SPLIT instances

        # We set the num_queries to a number proportional to the num_eval_queries
        num_queries = self.num_eval_queries * self.topk

        # We set min_rank to be self.MIN_TOPK so that we skip min_topk number
        #   of top candidates before we say that a passage doesn't contain
        #   an answer to a query.
        # We set max_rank to be self.MIN_TOPK to set the max number of
        #   no examples for the queries in the train split to 1. This allows
        #   us to have a label balanced set. 
        min_rank, max_rank = self.MIN_TOPK, self.MIN_TOPK

        train_instances = self.create_split_instances(TRAIN_SPLIT, collection_dict, queries_dicts[TRAIN_SPLIT],
                                                      qrels_dicts[TRAIN_SPLIT], topk_dicts[TRAIN_SPLIT],
                                                      num_queries=num_queries, min_rank=min_rank, max_rank=max_rank)
    
        # Combine the instances
        instances = dev_instances + train_instances

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
