import csv
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from common.general import ensure_file_downloaded, ensure_directory_exists
from common.hierarchical_logger import hlog
from .scenario import Scenario, MultipleRequestInstance, Reference, TRAIN_SPLIT, VALID_SPLIT, CORRECT_TAG


class MSMARCOScenario(Scenario):
    """ Scenario implementing MS MARCO challenge tasks.

    I. Overview

        MS MARCO (Microsoft MAchine Reading COmprehension) is a collection of
        large search datasets, collected using BING search questions, first
        released in (Bajaj et. al., 2016) and expanded ever since. The official
        MS MARCO website details all the available datasets and the proposed
        tasks: https://microsoft.github.io/msmarco/.

    II. Task

        In this scenario, we are focusing on information retrieval tasks from
        the MS MARCO benchmark. We frame the information retrieval task as a
        binary classification problem, similar to
        (Nogueira and Jiang et. al., 2020). Specifically, given a context and a
        question, the model's job is to predict whether the context includes an
        answer to the question by producing either a correct answer or a wrong
        answer. The specific tokens used for the correct and wrong answers are
        stored as class variables. Shared below is an example of how we would
        construct a prompt for a question, using 4 in-context training
        instances. Note that the last instance in the example, which is the
        instance we are evaluating, doesn't have an answer - since we want our
        model to answer the question.

            Passage: Search for foreclosed homes for sale in Mokena, Will
            County, Illinois.
            Question: what county is mokena in?
            Prompt: Does the passage above answer the question?
            Answer: Yes

            Passage: Original conversation. User: What happens to Bob Ewell at
            the end of the novel? A. We do not know what happens to him. B. He
            goes to prison for threatening Scout and Jem. C. He commits suicide.
            D. He is stabbed by Boo Radley Weegy: A. He is stabbed by Boo Radley
            cinprincess07|Points 10|User: Which character does the mockingbird
            best represent?. We do not know what happens to him. B. He goes to
            prison for threatening Scout and Jem. C. He commits suicide. D. He
            is stabbed by Boo Radley Weegy: A. He is stabbed by Boo Radley
            cinprincess07|Points 10|User: Which character does the mockingbird
            best represent?
            Question: what is boo!
            Prompt: Does the passage above answer the question?
            Answer: No

            Passage: Mini Bio (1) Mehmet Oz was born on June 11, 1960 in
            Cleveland, Ohio, USA as Mehmet Cengiz Oz. He is known for his work
            on The Dr. Oz Show (2009), You: The Owner's Manual (2005) and Today
            (1952). He has been married to Lisa Oz since June 29, 1985. They
            have four children.
            Question: what is doctor oz's birth name
            Prompt: Does the passage above answer the question?
            Answer: Yes

            Passage: Results. A thyroid-stimulating hormone (TSH) blood test is
            used to check for thyroid gland problems. The normal values listed
            here-called a reference range-are just a guide.These ranges vary
            from lab to lab, and your lab may have a different range for what's
            normal. thyroid-stimulating hormone (TSH) blood test is used to
            check for thyroid gland problems.
            Question: what is the test called for thyroid
            Prompt: Does the passage above answer the question?
            Answer: No

            Passage: The spliceosome is a complex of small nuclear RNA (snRNA)
            and small nuclear protein (snRNP) molecules, snRNAs and
            snRNPs.snRNPs include U1, U2, U4, U5 and U6.his removal is done in a
            coimplex protein structure called the spliceosome. The spliceosome
            splices out the non-coding introns from the primary mRNA transcript,
            and stiches the exons back together into the mature mRNA transcript.
            Question: exons definition biology
            Prompt: Does the passage above answer the question?
            Answer:

        As a result of each request, the model would produce a token or a set
        of tokens. To determine the ranking of a list of contexts for a
        question, we create a separate request for each context, where we pair
        the question with the context and ask for model's answer.

        Then, in the corresponding metric for our scenario, the contexts are
        ranked using the answer token and its log probability. Specifically, the
        ordering looks like the list given below, from good contexts at the top
        and bad contexts at the bottom, where UNKNOWN_ANSWER wuld corespond
        to any token that is not one of CORRECT_ANSWER and WRONG_ANSWER, using
        case insensitive match excluding whitespace.

            (1) CORRECT_ANSWER, highest log probability
                ...
                CORRECT_ANSWER, lowest log probability
                ...
                WRONG_ANSWER, lowest log probability
                ...
                WRONG_ANSWER, highest log probability
                ...
            (n) UNKNOWN_ANSWER(s)

        We then use standard information retrieval metrics, such as MRR and
        nDCG, to score the model using the rankings obtained using the strategy
        described above.

    III. Datasets

        There are two ranking tasks in the MS MARCO benchmark: document ranking
        and passage ranking. Both of these tasks have several tracks, using
        different subsets for the evaluation of the models. This scenario
        currently supports the passage ranking task tracks.

        All the datasets used in this scenario are hosted and retrieved from one
        of the following repositories:

            Official MS MARCO Website      | https://microsoft.github.io/msmarco/
            Benchmarking CodaLab Worksheet | https://worksheets.codalab.org/worksheets/0xf451c0dec2a6414aae0b68e8e325426c  # noqa
            TREC Website                   | https://trec.nist.gov

        This scenario makes use of 4 different types of files, explanation for
        each is given below, followed by a table listing the details for each
        of the datasets used.

            object: The object files contain all the objects that could be
                ranked for a question, each specified with an object ID (oid).
                For example, for the passage track, the objects would be
                passages.
            query: The query files contain the questions for a given task,
                each specified with a query ID (qid). Each task has a query file
                including the training examples. The validation queries are
                determined by the selected track of the task. Depending on the
                task and split/track, the queries read from the queries file
                are filtered to ensure they have corresponding qrels and top-k
                information before instances for the query are created. Because
                of this filtering, the number of queries in the query file
                doesn't directly correspond the number of queries for which
                instances are created.
            qrels: Each query file is accompanied by a qrels file, which
                specifies the relationship between a query with ID qid and an
                object with ID oid. The relations values can have different
                meanings depending on the split and the track. Note that not
                all queries would have corresponding query relations in the
                accompanied file. Also note that multiple objects may have the
                same relation value with a qiven query.
            topk: Each query file is accompanied by a top-k file, which lists
                the IDs of the top k best objects for a query with their
                accompanied rank. The top objects for each query were selected
                using the BM25 algorithm. The notebook used to generate the
                top-k files used in this scenario can be found at the
                Benchmarking CodaLab Worksheet. Note that not all queries would
                have a corresponding top-k objects in the accompanied file.

            |      LOCAL FILE NAME        |  TRACK  |  TRACK  |             CONTENT             |       FORMAT        |              Host              | Notes |  # noqa
            | passage_object.tsv          | passage |    -    | 8,841,823 passages              | <oid> <text>        | Benchmarking CodaLab Worksheet | (1)   |  # noqa
            | passage_train_queries.tsv   | passage |    -    | 808,731   queries               | <qid> <text>        | Official MS MARCO Website      |       |  # noqa
            | passage_train_qrels.tsv     | passage |    -    | 532,761   query relations       | <qid> 0 <pid> <rel> | Official MS MARCO Website      | (2)   |  # noqa
            | passage_train_topk.tsv      | passage |    -    | 20        top objects per query | <qid> <pid> <rank>  | Benchmarking CodaLab Worksheet | (3)   |  # noqa
            | passage_regular_queries.tsv | passage | regular | 6980      queries               | <qid> <text>        | Official MS MARCO Website      | (4)   |  # noqa
            | passage_regular_qrels.tsv   | passage | regular | 7437      query relations       | <qid> 0 <pid> <rel> | Official MS MARCO Website      | (2)   |  # noqa
            | passage_regular_topk.tsv    | passage | regular | 1000      top objects per query | <qid> <pid> <rank>  | Benchmarking CodaLab Worksheet |       |  # noqa
            | passage_trec_queries.tsv    | passage | trec    | 200       queries               | <qid> <text>        | Official MS MARCO Website      |       |  # noqa
            | passage_trec_qrels.tsv      | passage | trec    | 502,982   query relations       | <qid> 0 <pid> <rel> | Official MS MARCO Website      | (5)   |  # noqa
            | passage_trec_topk.tsv       | passage | trec    | 1000      top objects per query | <qid> <pid> <rank>  | Benchmarking CodaLab Worksheet |       |  # noqa

                Notes:
                    (1) We use a pre-processed version of the passage
                        collection, introduced in (MacAvaney, et. al. 2021),
                        which greatly improves the quality of the passages. The
                        cleaned collection is hosted on the Benchmarking CodaLab
                        Worksheet as there is no other reliable publicly hosted
                        copy.
                    (2) The only relation is 1, which indicates that the object
                        with the ID the oid is the gold match for the query with
                        the ID qid.
                    (3) The number of top objects ranked was limited to 20 for
                        the training set as we only generate 2 instances per
                        training query, one corresponding to a gold matching
                        instance, and the other one corresponding to a probable
                        non-matching instance.
                    (4) The labels (qrels) of the official test queries are not
                        publicly released. Since we need to have access to the
                        qrels file to evaluate the models, we instead use the
                        development set ("queries.dev.small.tsv"), which can be
                        found at
                        https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
                    (5) The relations for the TREC task of the passage track
                        can be any of [0, 1, 2, 3]. We consider [0, 1] to be
                        wrong matches and [2, 3] to be gold matches.

    IV. Baselines

        Currently, we use 4 baselines for the MS MARCO scenario, details for
        which are summarized in the table below.

            Baseline | The baseline name.
            #VQ      | Effective number of validation queries, which are the
                       queries for which instances were created.
            #VI      | Number of validation instances. For each effective
                       validation query, multiple instances would be created,
                       governed by the provided parameters.

        | Baseline                | #VQ |  #VI   | Parameters
        | regular_topk            | 200 | 10,000 | task=passage,track=regular,use_topk_passages=True,valid_topk=50,num_valid_queries=200
        | regular_topk_with_qrels | 200 | 10,085 | task=passage,track=regular,use_qrels_passages=True,use_topk_passages=True,valid_topk=50,num_valid_queries=200
        | trec_topk               | 43  | 4300   | task=passage,track=trec,use_topk_passages=True,valid_topk=100
        | trec_qrels              | 43  | 9260   | task=passage,track=trec,use_qrels_passages=True

        On average, the requests for the MS MARCO scenario have ~550 tokens
        using the GPT-2 tokenizer. Multiplying this number with the #VI column
        gives an estimate on the number of request tokens that would be required
        to run the given baseline on GPT models.

    References

         (Bajaj et. al., 2016)              | https://arxiv.org/abs/1611.09268
         (Nogueira and Jiang et. al., 2020) | https://arxiv.org/abd/2003.06713
         (MacAvaney, et. al. 2021)          | https://arxiv.org/abs/2103.02280

    """

    """ Information on the MSMARCOScenario. """
    name = "msmarco"
    description = "Microsoft Machine Reading Comprehension"
    tags = ["information_retrieval"]

    """ Output strings. """
    CORRECT_OUTPUT = "Yes"
    WRONG_OUTPUT = "No"
    RELEVANCE_TO_OUTPUT = {
        True: CORRECT_OUTPUT,
        False: WRONG_OUTPUT,
    }

    """ Names of the tasks and tracks that we support. """
    PASSAGE_TASK = "passage"
    REGULAR_TRACK = "regular"
    TREC_TRACK = "trec"
    TASK_NAMES: List[str] = [PASSAGE_TASK]
    TRACK_NAMES: List[str] = [REGULAR_TRACK, TREC_TRACK]

    """ Information needed to retrieve MS MARCO datasets. """
    CODALAB_URI_TEMPLATE: str = "https://worksheets.codalab.org/rest/bundles/{bundle}/contents/blob/"
    MSMARCO_URI_TEMPLATE: str = "https://msmarco.blob.core.windows.net/msmarcoranking/{file_name}"

    DATA_URIS = {
        (PASSAGE_TASK, "object"): CODALAB_URI_TEMPLATE.format(bundle="0x50d32fc56ad04dd89510bf86f9c1c9d3"),
        (PASSAGE_TASK, TRAIN_SPLIT, "queries"): MSMARCO_URI_TEMPLATE.format(file_name="queries.train.tsv"),
        (PASSAGE_TASK, TRAIN_SPLIT, "qrels"): MSMARCO_URI_TEMPLATE.format(file_name="qrels.train.tsv"),
        (PASSAGE_TASK, TRAIN_SPLIT, "topk"): CODALAB_URI_TEMPLATE.format(bundle="0x8c43d4ec02ea48d6a727683a9676b77b"),
        (PASSAGE_TASK, REGULAR_TRACK, "queries"): CODALAB_URI_TEMPLATE.format(
            bundle="0xf5ccf54707b548f9a4c43502c6f15719"
        ),
        (PASSAGE_TASK, REGULAR_TRACK, "qrels"): MSMARCO_URI_TEMPLATE.format(file_name="qrels.dev.small.tsv"),
        (PASSAGE_TASK, REGULAR_TRACK, "topk"): CODALAB_URI_TEMPLATE.format(bundle="0xbc3dfacb2b7746809e582ee01fa5fe70"),
        (PASSAGE_TASK, TREC_TRACK, "queries"): MSMARCO_URI_TEMPLATE.format(file_name="msmarco-test2019-queries.tsv.gz"),
        (PASSAGE_TASK, TREC_TRACK, "qrels"): "https://trec.nist.gov/data/deep/2019qrels-pass.txt",
        (PASSAGE_TASK, TREC_TRACK, "topk"): CODALAB_URI_TEMPLATE.format(bundle="0x2e80572f93b748d594b817249013bdac"),
    }

    """ Dictionary mapping the separator for the datasets that don't use "\t" as the separator. """
    NON_TSV_SEPARATED_DATASETS = {(PASSAGE_TASK, TREC_TRACK, "qrels"): " "}

    """ Dictionary mapping task track tuples to the number of queries. """
    NUM_QUERIES = {
        (PASSAGE_TASK, TRAIN_SPLIT): 808731,
        (PASSAGE_TASK, REGULAR_TRACK): 6980,
        (PASSAGE_TASK, TREC_TRACK): 200,
    }

    """ Gold relations for a given task track tuple.

    This is the value that is read from the qrels file for a given
    configuration.
    """
    GOLD_RELATIONS = {
        (PASSAGE_TASK, TRAIN_SPLIT): [1],
        (PASSAGE_TASK, REGULAR_TRACK): [1],
        (PASSAGE_TASK, TREC_TRACK): [2, 3],
    }

    """ Measure names that will be used for each task track pair.

    The measure names are retrieved in run_specs.py and passed to the
    InformationRetrievalMetrics class, and correspond to the measure names in
    pytrec_eval.supported_measures.
    """
    NDCG = "ndcg"
    RECIP_RANK = "recip_rank"
    MEASURE_NAMES = {
        (PASSAGE_TASK, REGULAR_TRACK): [RECIP_RANK],
        (PASSAGE_TASK, TREC_TRACK): [RECIP_RANK, NDCG],
    }

    """ The information retrieval mode used by this scenario. """
    BINARY_LOGPROB_MODE = "binary_logprob"

    """ Upper and lower bounds on top-k.

    The top-k number represents the number of passages we will consider per
    query. Max top-k for the train and validation files are set to the number
    of passages included in the corresponding top-k files.
    """
    MIN_TOPK: int = 11
    MAX_TRAIN_TOPK: int = 20
    MAX_VALID_TOPK: int = 1000

    def __init__(
        self,
        task: str,
        track: str,
        use_qrels_passages: bool = False,
        use_topk_passages: bool = False,
        valid_topk: Optional[int] = None,
        num_valid_queries: Optional[int] = None,
        num_train_queries: int = 1000,
    ):
        """ The constructor for the MSMARCOScenario.

        Args:
            task: Name of the task, which should be one of self.TASK_NAMES.
                There are several MSMARCO tasks, and we use the task parameter
                to specify which task we would like performed. Currently,
                available values are as follows:
                    "passage": The Passage Retrieval task.
            track: Name of the track. Currently, available values are as follows:
                    "passage" task:
                        "regular": The regular passage track.
                        "trec": The TREC track.
            use_qrels_passages: Flag controlling whether validation instances
                should be made for the passages in the qrels dictionary for a
                given validation query.
            use_topk_passages: Flag controlling whether validation instances
                should be made for the passages in the top "valid_topk" of the
                topk dictionary. If use_topk_passages is set, valid_topk must
                also be set.
            valid_topk: The number of top passages for which the validation
                instances will be created if "use_topk_passages" is set. Must
                be in the range [self.MIN_TOPK, self.MAX_VALID_TOPK].
            num_valid_queries: Number of validation queries that will be used to
                create validation instances. Must be smaller than or equal to
                self.NUM_QUERIES for the selected track and task.
            num_train_queries: Number of train queries that will be used to
                create the train instances. Must be smaller than or equal to
                self.NUM_QUERIES for the train set of the selected track.
        """
        # Input validation
        assert task in self.TASK_NAMES
        self.task: str = task

        assert track in self.TRACK_NAMES
        self.track: str = track

        assert use_qrels_passages or (
            use_topk_passages and valid_topk and self.MIN_TOPK <= valid_topk <= self.MAX_VALID_TOPK
        )
        self.use_qrels_passages: bool = use_qrels_passages
        self.use_topk_passages: bool = use_topk_passages
        self.valid_topk: Optional[int] = valid_topk

        if not num_valid_queries:
            num_valid_queries = self.NUM_QUERIES[(self.task, self.track)]
        msg = f"""Number of validation queries for {(self.task, self.track)}
                  should be <= {self.NUM_QUERIES[(self.task, self.track)]}."""
        assert num_valid_queries <= self.NUM_QUERIES[(self.task, self.track)], msg
        self.num_valid_queries: int = num_valid_queries

        msg = f"Number of train queries should not be bigger than {self.NUM_QUERIES[(self.task, TRAIN_SPLIT)]}."
        assert num_train_queries <= self.NUM_QUERIES[(self.task, TRAIN_SPLIT)], msg
        self.num_train_queries: int = num_train_queries

        # Instance level variables we use throughout
        self.random: random.Random = random.Random(1885)
        self.train_topk: int = self.MAX_TRAIN_TOPK
        self.min_train_wrong_topk = self.MIN_TOPK
        self.gold_relations: Dict[str, List[int]] = {
            TRAIN_SPLIT: self.GOLD_RELATIONS[(self.task, TRAIN_SPLIT)],
            VALID_SPLIT: self.GOLD_RELATIONS[(self.task, self.track)],
        }

        # Data dictionaries that will be populated once the scenario is run
        self.object_dict: Dict[int, str] = {}
        self.query_dicts: Dict[str, Dict[int, str]] = {TRAIN_SPLIT: {}, VALID_SPLIT: {}}
        self.qrels_dicts: Dict[str, Dict[int, Dict[int, int]]] = {TRAIN_SPLIT: {}, VALID_SPLIT: {}}
        self.topk_dicts: Dict[str, Dict[int, Dict[int, int]]] = {TRAIN_SPLIT: {}, VALID_SPLIT: {}}

    def download_file(self, urlstring: str, target_file_name: str) -> str:
        """ Download the resource at urlstring and return the absolute path.

        Downloaded file is saved to a directory named 'data' in self.output_path.
        """
        data_path = os.path.join(self.output_path, "data")
        ensure_directory_exists(data_path)
        target_file_path = os.path.join(data_path, target_file_name)
        return target_file_path  # TODO remove
        ensure_file_downloaded(source_url=urlstring, target_path=target_file_path)
        return target_file_path

    @staticmethod
    def create_id_item_dict(file_path: str, delimiter: str = "\t") -> Dict[int, str]:
        """ Read the provided file as an id to item dictionary.

        Given a file with rows in the following format:
            <id>   <item>

        Return a dictionary of the form:
            {
                <id>: <item>,
                ...
            }
        """
        id_to_item_dict = {}
        with open(file_path, encoding="utf-8") as f:
            for _id, content in csv.reader(f, delimiter=delimiter):
                id_to_item_dict[int(_id)] = content
        return id_to_item_dict

    @staticmethod
    def create_qrels_dict(file_path: str, delimiter: str = "\t") -> Dict[int, Dict[int, int]]:
        """ Read the provided file as a qrels dictionary.

        Given a file with rows in the following format:
            <qid>   0   <pid>   <rel>

        Return a dictionary of the form:
            {
                <qid>: {
                    <pid>: <rel>,
                    ...
                },
                ...
            }
        """
        qrels_dict: Dict[int, Dict[int, int]] = defaultdict(dict)
        with open(file_path, encoding="utf-8") as f:
            for qid, _, pid, rel in csv.reader(f, delimiter=delimiter):
                qrels_dict[int(qid)][int(pid)] = int(rel)
        qrels_dict = {k: v for k, v in qrels_dict.items()}  # Convert to regular dict
        return qrels_dict

    @staticmethod
    def create_topk_dict(file_path: str, delimiter: str = "\t") -> Dict[int, Dict[int, int]]:
        """ Read the provided file as a topk dictionary.

        Given a file with rows in the following format:
            <qid>   <pid>   <rank>

        Return a dictionary of the form:
            {
                <qid>: {
                    <rank>: <pid>,
                    ...
                },
                ...
            }
        """
        topk_dict: Dict[int, Dict[int, int]] = defaultdict(dict)
        with open(file_path, encoding="utf-8") as f:
            for qid, pid, rank in csv.reader(f, delimiter=delimiter):
                topk_dict[int(qid)][int(rank)] = int(pid)
        topk_dict = {k: v for k, v in topk_dict.items()}  # Convert the defaultdict to a regular dict
        return topk_dict

    def download_helper(self, data_key: Tuple[str, str, str]) -> str:
        """ Call download_file for self.DATA_URIS[data_key] and return the file path to the downloaded file. """
        # Download the file
        urlstring = self.DATA_URIS[data_key]
        target_file_name = f"{'_'.join(data_key)}.tsv"
        file_path = self.download_file(urlstring, target_file_name)

        # Convert .txt file with ' ' separated values to .tsv
        if data_key in self.NON_TSV_SEPARATED_DATASETS:
            with open(file_path, "r") as f:
                tsv_content = f.read().replace(self.NON_TSV_SEPARATED_DATASETS[data_key], "\t")
            with open(file_path, "w") as f:
                f.write(tsv_content)

        # Return path
        return file_path

    def prepare_data_dicts(self):
        """ Download and load the data for all the data dictionaries. """
        self.object_dict = self.create_id_item_dict(self.download_helper((self.task, "object")))
        self.query_dicts[TRAIN_SPLIT] = self.create_id_item_dict(
            self.download_helper((self.task, TRAIN_SPLIT, "queries"))
        )
        self.qrels_dicts[TRAIN_SPLIT] = self.create_qrels_dict(self.download_helper((self.task, TRAIN_SPLIT, "qrels")))
        self.topk_dicts[TRAIN_SPLIT] = self.create_topk_dict(self.download_helper((self.task, TRAIN_SPLIT, "topk")))
        self.query_dicts[VALID_SPLIT] = self.create_id_item_dict(
            self.download_helper((self.task, self.track, "queries"))
        )
        self.qrels_dicts[VALID_SPLIT] = self.create_qrels_dict(self.download_helper((self.task, self.track, "qrels")))
        self.topk_dicts[VALID_SPLIT] = self.create_topk_dict(self.download_helper((self.task, self.track, "topk")))

    def filter_qids(self, split: str, check_topk: bool = True) -> List[int]:
        """ Return the filtered Query IDs for TRAIN_SPLIT or VALID_SPLIT, as specified by the split parameter.

        All the query IDs included satisfy the following conditions:
            (1) Corresponding qrels dictionary exists and contains at least 1
                passage ID that is in self.gold_passages[split].
            (2) If check_topk flag is set, corresponding topk dictionary exists
                and has at least topk passages, where topk is one of
                self.train_topk or self.valid_topk depending on the specified
                split.
        """
        topk = self.train_topk if split == TRAIN_SPLIT else self.valid_topk
        qids = []
        for qid in self.query_dicts[split]:
            qrels_condition = qid in self.qrels_dicts[split] and any(
                [v in self.gold_relations[split] for v in self.qrels_dicts[split][qid].values()]
            )
            topk_condition = qid in self.topk_dicts[split] and topk and len(self.topk_dicts[split][qid]) >= topk
            topk_condition = not check_topk or topk_condition
            if qrels_condition and topk_condition:
                qids.append(qid)
        return qids

    @staticmethod
    def make_context(passage: str, question: str) -> str:
        """ Make and return the instance context given the provided passage and query. """
        prompt = "Does the passage above answer the question?"
        return "\n".join([passage, f"Question: {question}", f"Prompt: {prompt}"])

    def make_instance(self, qid: int, pid: int, split: str) -> MultipleRequestInstance:
        """ Create and return an instance made using the provided parameters. """
        object_text = self.object_dict[pid]
        query_text = self.query_dicts[split][qid]
        context = self.make_context(object_text, query_text)
        rel = None if pid not in self.qrels_dicts[split][qid] else self.qrels_dicts[split][qid][pid]
        is_relevant = rel in self.gold_relations[split]
        reference = Reference(output=self.RELEVANCE_TO_OUTPUT[is_relevant], tags=[CORRECT_TAG])
        # Create instance
        instance = MultipleRequestInstance(
            input=context, references=[reference], split=split, group_id=str(qid), request_id=str(pid), relevance=rel
        )
        return instance

    def get_train_instances(self) -> List[MultipleRequestInstance]:
        """ Create and return the instances for the training set.

        For a random set of self.num_train_queries in the training set:
            1. We create 1 correct instance, where the passage included
               corresponds to the best passage for the given training query.
            2. We create 1 wrong instance, where the passage included
               corresponds to a non-gold passage for the given training query.
        """
        split = TRAIN_SPLIT
        qids = self.filter_qids(split, check_topk=True)  # Filter queries
        self.random.shuffle(qids)  # Select a random subset

        instances = []
        for qid in qids[: self.num_train_queries]:  # Limit the number of queries to the user provided number
            # Get correct pids
            sorted_qrels = sorted(self.qrels_dicts[split][qid].items(), key=lambda x: x[1], reverse=True)
            correct_pids = [pid for (pid, rel) in sorted_qrels if rel in self.gold_relations[split]]
            instances.append(self.make_instance(qid, correct_pids[0], split))  # Only use the top correct pid

            # Get wrong pids
            filtered_pids = [
                pid
                for k, pid in self.topk_dicts[split][qid].items()
                if k >= self.min_train_wrong_topk and k <= self.train_topk
            ]
            wrong_pids = [
                pid
                for pid in filtered_pids
                if pid not in self.qrels_dicts[split][qid]
                or self.qrels_dicts[split][qid][pid] not in self.gold_relations[split]
            ]
            instances.append(self.make_instance(qid, wrong_pids[0], split))  # Only use the top wrong pid

        return instances

    def get_valid_instances(self) -> List[MultipleRequestInstance]:
        """ Create and return the instances for the validation set.

        For a random set of self.num_valid_queries in the validation set:
            1. If self.use_qrels_passages flag is set, we ensure that an
               instance is created for all the passages that appear in the
               corresponding qrels dictionary for the given validation query.
            2. If self.use_topk_passages flag is set, we ensure that an
               instance is created for all the passages that appear in top
               self.valid_topk passages for the given validation query.
        """
        split = VALID_SPLIT
        qids = self.filter_qids(split, check_topk=self.use_topk_passages)  # Filter queries
        self.random.shuffle(qids)  # Select a random subset

        instances = []
        num_queries = min(self.num_valid_queries, len(qids))
        for qid in qids[:num_queries]:
            # Initialize a pid set
            pids = []
            # Add qrels passages if the flag is set
            if self.use_qrels_passages:
                pids += list(self.qrels_dicts[split][qid].keys())
            # Add topk passages if the flag is set
            if self.use_topk_passages and self.valid_topk:
                pids += [pid for k, pid in self.topk_dicts[split][qid].items() if k <= self.valid_topk]
            # Create instances
            instances += [self.make_instance(qid, pid, split) for pid in set(pids)]
        return instances

    def get_instances(self) -> List[MultipleRequestInstance]:
        """ Return the instances for this scenario.

        Refer to the documentation of the following methods for details on how
        the instances are created:
            * self.get_train_instances
            * self.get_valid_instances
        """
        # Get dataset and topk dictionaries
        hlog("MS MARCO Scenario: Preparing the datasets.")
        self.prepare_data_dicts()
        hlog("MS MARCO Scenario: Preparing the training instances.")
        train_instances = self.get_train_instances()
        hlog("MS MARCO Scenario: Preparing the validation instances.")
        valid_instances = self.get_valid_instances()
        instances = train_instances + valid_instances
        hlog("MS MARCO Scenario: Done preparing all the instances.")

        return instances
