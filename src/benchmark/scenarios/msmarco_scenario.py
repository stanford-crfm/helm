import csv
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, cast

from common.general import ensure_file_downloaded, ensure_directory_exists
from common.hierarchical_logger import hlog
from .scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    VALID_SPLIT,
    CORRECT_TAG,
    RELEVANCE_TAG_TEMPLATE,
    RANK_TAG_TEMPLATE,
)


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
        specified in the adapter specification. Shared below is an example of
        how we would construct a prompt for a question, using 4 in-context
        training instances, where the correct and wrong output tokens are
        respectively specified as `Yes` and `No` in the adapter specification.
        Note that the last instance in the example, which is the instance we
        are evaluating, doesn't have an answer - since we want our model to
        answer the question.

            Passage: Search for foreclosed homes for sale in Mokena, Will
            County, Illinois.
            Question: what county is mokena in?
            Prompt: Does the passage above answer the question?
            Answer: Yes

            Passage: 10 Acres MOKENA, Will County, Illinois $649,000.
            Spectacular secluded 10 acres of land surrounded in wooded area with
            muter trees! 2 Parcels each is 5 acres of land with building in the
            middle!
            Question: what county is mokena in?
            Prompt: Does the passage above answer the question?
            Answer: No

            Passage: boo is a term that is derived from the French word beau
            meaning beautiful. In 18th century England it meant an admirer,
            usually male. It made it's way into Afro-Caribean language perhaps
            through the French colonisation of some Caribean islands.
            Now meaning girl or boyfriend.
            Question: what is boo!
            Prompt: Does the passage above answer the question?
            Answer: Yes

            Passage: Original conversation. User: What happens to Bob Ewell at
            the end of the novel? A. We do not know what happens to him. B.
            He goes to prison for threatening Scout and Jem. C. He commits
            suicide. D. He is stabbed by Boo Radley Weegy: A. He is stabbed
            by Boo Radley cinprincess07|Points 10|User: Which character does
            the mockingbird best represent?. We do not know what happens to
            him. B. He goes to prison for threatening Scout and Jem. C. He
            commits suicide. D. He is stabbed by Boo Radley Weegy: A. He
            is stabbed by Boo Radley cinprincess07|Points 10|User: Which
            character does the mockingbird best represent?
            Question: what is boo!
            Prompt: Does the passage above answer the question?
            Answer: No

            Passage: Kaczynski himself has insisted repeatedly that he is
            sane, and a lawyer who served as an adviser on his case, Michael
            Mello, wrote a book arguing that the prisoner had been unfairly
            labeled. In my opinion he is not crazy, Mello wrote in the book,
            The United States of America Versus Theodore John Kaczynski.
            Question: who wrote the book ninjago
            Prompt: Does the passage above answer the question?
            Answer:

        As a result of each request, the model would produce a token or a set
        of tokens. To determine the ranking of a list of contexts for a
        question, we create a separate request for each context, where we pair
        the question with the context and ask for model's answer.

        Then, in the corresponding metric for our scenario, the contexts are
        ranked using the answer token and its log probability. Specifically, the
        ordering looks like the list given below, from good contexts at the top
        to bad contexts at the bottom, where UNKNOWN_ANSWER would correspond
        to any token that is not one of correct or wrong answer tokens, using
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

        We then use standard information retrieval metrics, such as RR and
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
                instances can be created.
            qrels: Each query file is accompanied by a qrels file, which
                specifies the relationship between a query with ID qid and an
                object with ID oid. The relevance values can have different
                meanings depending on the split and the track. Note that not
                all queries would have corresponding query relevances in the
                accompanied file. Also note that multiple objects may have the
                same relevance value with a qiven query.
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
                    (2) The only relevance values is 1, which indicates that the
                        object with the ID the oid is the gold match for the query
                        with the ID qid.
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
                    (5) The relevance values for the TREC task of the passage
                        track can be any of [0, 1, 2, 3]. We consider [0, 1] to
                        be wrong matches and [2, 3] to be gold matches.

    IV. Baselines

        Currently, we use 4 baselines for the MS MARCO scenario, details for
        which are summarized in the table below.

            Baseline | The baseline name.
            #VQ      | Effective number of validation queries, which are the
                       queries for which instances were created.
            #VR      | Number of validation requests. For each effective
                       validation query, multiple requests would be created,
                       governed by the provided parameters.

        | Baseline                | #VQ |  #VR   | Parameters
        | regular_topk            | 200 | 10,000 | track=regular,use_topk_passages=True,valid_topk=50,num_valid_queries=200
        | regular_topk_with_qrels | 200 | 10,085 | track=regular,use_qrels_passages=True,use_topk_passages=True,valid_topk=50,num_valid_queries=200
        | trec_topk               | 43  | 4300   | track=trec,use_topk_passages=True,valid_topk=100
        | trec_qrels              | 43  | 9260   | track=trec,use_qrels_passages=True

        On average, the requests for the MS MARCO scenario have ~550 tokens
        using the GPT-2 tokenizer. Multiplying this number with the #VR column
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

    """ Supported passage tracks. """
    REGULAR_TRACK = "regular"
    TREC_TRACK = "trec"
    TRACK_NAMES: List[str] = [REGULAR_TRACK, TREC_TRACK]

    """ Information needed to retrieve MS MARCO datasets. """
    CODALAB_URI_TEMPLATE: str = "https://worksheets.codalab.org/rest/bundles/{bundle}/contents/blob/"
    MSMARCO_URI_TEMPLATE: str = "https://msmarco.blob.core.windows.net/msmarcoranking/{file_name}"

    DATA_URIS = {
        ("objects",): CODALAB_URI_TEMPLATE.format(bundle="0x50d32fc56ad04dd89510bf86f9c1c9d3"),
        (TRAIN_SPLIT, "queries"): MSMARCO_URI_TEMPLATE.format(file_name="queries.train.tsv"),
        (TRAIN_SPLIT, "qrels"): MSMARCO_URI_TEMPLATE.format(file_name="qrels.train.tsv"),
        (TRAIN_SPLIT, "topk"): CODALAB_URI_TEMPLATE.format(bundle="0x8c43d4ec02ea48d6a727683a9676b77b"),
        (REGULAR_TRACK, "queries"): CODALAB_URI_TEMPLATE.format(bundle="0xf5ccf54707b548f9a4c43502c6f15719"),
        (REGULAR_TRACK, "qrels"): MSMARCO_URI_TEMPLATE.format(file_name="qrels.dev.small.tsv"),
        (REGULAR_TRACK, "topk"): CODALAB_URI_TEMPLATE.format(bundle="0xbc3dfacb2b7746809e582ee01fa5fe70"),
        (TREC_TRACK, "queries"): MSMARCO_URI_TEMPLATE.format(file_name="msmarco-test2019-queries.tsv.gz"),
        (TREC_TRACK, "qrels"): "https://trec.nist.gov/data/deep/2019qrels-pass.txt",
        (TREC_TRACK, "topk"): CODALAB_URI_TEMPLATE.format(bundle="0x2e80572f93b748d594b817249013bdac"),
    }

    """ Dictionary mapping dataset files to their separator. """
    DATASET_SEPARATOR = defaultdict(lambda: "\t")
    DATASET_SEPARATOR[(TREC_TRACK, "qrels")] = " "

    """ Dictionary mapping task track tuples to the number of queries. """
    NUM_TRAIN_QUERIES = 1000
    MAX_NUM_QUERIES = {
        TRAIN_SPLIT: 808731,
        REGULAR_TRACK: 6980,
        TREC_TRACK: 200,
    }

    """ Gold relations for a given task track tuple.

    These are the values that are read from the qrels file for a given
    configuration.
    """
    GOLD_RELATIONS = {
        TRAIN_SPLIT: [1],
        REGULAR_TRACK: [1],
        TREC_TRACK: [2, 3],
    }

    """ Measure names that will be used for each task track pair.

    The measure names are retrieved in run_specs.py and passed to the
    InformationRetrievalMetrics class, and correspond to the measure names in
    pytrec_eval.supported_measures.
    """
    RECALL_MEASURES = [f"recall.{k}" for k in [1, 2, 3, 5, 10, 20]]
    RECIP_RANK_MEASURES = [f"recip_rank.{k}" for k in [5, 10, 20]]
    SUCCESS_MEASURES = [f"success.{k}" for k in [1, 2, 3, 5, 10, 20]]
    NDCG_CUT_MEASURES = [f"ndcg_cut.{k}" for k in [5, 10, 20]]
    MEASURE_NAMES = {
        REGULAR_TRACK: SUCCESS_MEASURES + RECALL_MEASURES + RECIP_RANK_MEASURES,
        TREC_TRACK: SUCCESS_MEASURES + RECALL_MEASURES + RECIP_RANK_MEASURES + NDCG_CUT_MEASURES,
    }

    """ Upper and lower bounds on top-k.

    The top-k number represents the number of passages we will consider per
    query. Max top-k for the train and validation files are set to the number
    of passages included in the corresponding top-k files.
    """
    MIN_TOPK: int = 11
    MAX_TRAIN_TOPK: int = 20
    MAX_VALID_TOPK: int = 1000

    def __init__(
        self, track: str, use_topk_objects: bool = False, valid_topk: Optional[int] = None,
    ):
        """ The constructor for the MSMARCOScenario.

        Args:
            track: Name of the passage track. Currently, available values are
            as follows:
                    "regular": The regular passage track.
                    "trec": The TREC passage track.
            use_topk_objects: Flag controlling whether validation instances
                should be made for the objects in the top "valid_topk" of the
                topk dictionary. If use_topk_objects is set, valid_topk must
                also be set.
            valid_topk: The number of top objects for which the validation
                instances will be created if "use_topk_objects" is set. Must
                be in the range [self.MIN_TOPK, self.MAX_VALID_TOPK].
        """
        # Input validation
        assert track in self.TRACK_NAMES
        self.track: str = track

        self.use_topk_objects: bool = use_topk_objects
        self.valid_topk: Optional[int] = valid_topk
        if self.use_topk_objects:
            assert valid_topk and self.MIN_TOPK <= valid_topk <= self.MAX_VALID_TOPK

        # Instance level variables we use throughout
        self.random: random.Random = random.Random(1885)
        self.num_train_queries: int = self.NUM_TRAIN_QUERIES
        self.train_topk: int = self.MAX_TRAIN_TOPK
        self.min_train_wrong_topk = self.MIN_TOPK
        self.gold_relations: Dict[str, List[int]] = {
            TRAIN_SPLIT: self.GOLD_RELATIONS[TRAIN_SPLIT],
            VALID_SPLIT: self.GOLD_RELATIONS[self.track],
        }
        self.oid_index_dict = Dict[int, int]  # Used to efficiently randomize Object ID lists.

        # Data structures that will be populated once the scenario is run.
        self.object_dict: Dict[int, str] = {}
        self.oids: List[int] = []
        self.query_dicts: Dict[str, Dict[int, str]] = {TRAIN_SPLIT: {}, VALID_SPLIT: {}}
        self.qrels_dicts: Dict[str, Dict[int, Dict[int, int]]] = {TRAIN_SPLIT: {}, VALID_SPLIT: {}}
        self.topk_dicts: Dict[str, Dict[int, Dict[int, int]]] = {TRAIN_SPLIT: {}, VALID_SPLIT: {}}
        self.qids: Dict[str, List[int]] = {TRAIN_SPLIT: [], VALID_SPLIT: []}

    def download_file(self, urlstring: str, target_file_name: str) -> str:
        """ Download the resource at urlstring and return the absolute path.

        Downloaded file is saved to a directory named 'data' in self.output_path.
        """
        data_path = os.path.join(self.output_path, "data")
        ensure_directory_exists(data_path)
        target_file_path = os.path.join(data_path, target_file_name)
        ensure_file_downloaded(source_url=urlstring, target_path=target_file_path)
        return target_file_path

    def download_helper(self, data_key: Tuple[str, str]) -> str:
        """ Call download_file for self.DATA_URIS[data_key] and return the file path to the downloaded file. """
        # Download the file
        urlstring = self.DATA_URIS[data_key]
        target_file_name = f"{'_'.join(data_key)}.tsv"
        file_path = self.download_file(urlstring, target_file_name)

        # Ensure that all the files are separated with a tab character.
        seperator = self.DATASET_SEPARATOR[data_key]
        if seperator != "\t":
            with open(file_path, "r") as f:
                tsv_content = f.read().replace(seperator, "\t")
            with open(file_path, "w") as f:
                f.write(tsv_content)

        # Return path
        return file_path

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

    def prepare_data(self):
        """ Download and load all the data. """
        # Passages
        self.object_dict = self.create_id_item_dict(self.download_helper(("objects",)))
        self.oids = list(self.object_dict.keys())

        # Train queries
        self.query_dicts[TRAIN_SPLIT] = self.create_id_item_dict(self.download_helper((TRAIN_SPLIT, "queries")))
        self.qrels_dicts[TRAIN_SPLIT] = self.create_qrels_dict(self.download_helper((TRAIN_SPLIT, "qrels")))
        self.topk_dicts[TRAIN_SPLIT] = self.create_topk_dict(self.download_helper((TRAIN_SPLIT, "topk")))
        self.qids[TRAIN_SPLIT] = list(self.query_dicts[TRAIN_SPLIT].keys())

        # Validation queries
        self.query_dicts[VALID_SPLIT] = self.create_id_item_dict(self.download_helper((self.track, "queries")))
        self.qrels_dicts[VALID_SPLIT] = self.create_qrels_dict(self.download_helper((self.track, "qrels")))
        self.topk_dicts[VALID_SPLIT] = self.create_topk_dict(self.download_helper((self.track, "topk")))
        self.qids[VALID_SPLIT] = list(self.query_dicts[VALID_SPLIT].keys())

    def shuffle_ids(self):
        """ Shuffle Object and Query ID lists.

        This is the only place we perform shuffling throughout the scenario,
        which allows us to make use of caching even if the scenario is run with
        different user parameters.
        """
        self.random.shuffle(self.oids)
        self.oid_index_dict = {oid: ind for ind, oid in enumerate(self.oids)}
        self.random.shuffle(self.qids[TRAIN_SPLIT])
        self.random.shuffle(self.qids[VALID_SPLIT])

    def get_split_variables(self, split):
        """ Return variables storing data for the given split. """
        qids = self.qids[split]
        query_dict = self.query_dicts[split]
        qrels_dict = self.qrels_dicts[split]
        topk = self.train_topk if split == TRAIN_SPLIT else self.valid_topk
        topk_dict = self.topk_dicts[split]
        gold_relations = set(self.gold_relations[split])
        return qids, query_dict, qrels_dict, topk, topk_dict, gold_relations

    def filter_qids(self, split: str, check_topk: bool = True) -> List[int]:
        """ Return filtered Query IDs for the provided split.

        We filter each query based on the following conditions:
            (1) A query must have a corresponding query relations (qrels) dictionary,
                which specifies contain an answer for a query. We need to perform
                this check because the qrels dictionaries provided as part of MS MARCO
                tasks aren't guaranteed to have an entry for each query.
            (2) The qrels dictionary corresponding to the query must identify at least
                1 gold object. This is so that for each instance we create, we ensure
                that there is at least 1 reference with a correct label.
            (3) If check_topk flag is set:
                    (a) We ensure that there is a corresponding topk dictionary for
                        the query. The topk dictionary tells us which objects are
                        most likely candidates for a given query.
                    (b) We ensure that the corresponding topk dictionary ranks at least
                        self.train_topk or self.valid_topk objects depending on the
                        specified split.
        """
        # Retrieve variables for the split.
        qids, _, qrels_dict, topk, topk_dict, gold_relations = self.get_split_variables(split)

        # (1) Ensure that there is a query relations dictionary for each query.
        filtered_qids = [qid for qid in qids if qid in qrels_dict]
        # (2) Ensure that the query relations specified include at least 1 gold object.
        filtered_qids = [qid for qid in filtered_qids if gold_relations.intersection(set(qrels_dict[qid].values()))]
        # (3) Check topk.
        if check_topk:
            # (3a) Ensure that there is a topk dictionary for each query.
            filtered_qids = [qid for qid in filtered_qids if qid in topk_dict]
            # (3b) Ensure that there are at least topk objects in the topk dictionary.
            filtered_qids = [qid for qid in filtered_qids if len(topk_dict[qid]) >= topk]

        return filtered_qids

    def create_reference(self, oid: int, gold: bool, rel: Optional[int], rank: Optional[int]) -> Reference:
        """ Create and return a reference made using the provided parameters. """
        # Create tags
        tags = []
        if gold:
            tags.append(CORRECT_TAG)  # Correctness
        if rel:
            tags.append(RELEVANCE_TAG_TEMPLATE.format(relevance=rel))  # Relevance
        if rank:
            tags.append(RANK_TAG_TEMPLATE.format(rank=rank))  # Top-k rank

        # Get object text
        object_text = self.object_dict[oid]

        # Create the reference
        reference = Reference(output=object_text, tags=tags)
        return reference

    def create_instance(self, qid: int, split: str, oids: List[int]) -> Instance:
        """ Create and return an instance made using the provided parameters. """
        # Retrieve variables for the split.
        _, query_dict, qrels_dict, _, topk_dict, gold_relations = self.get_split_variables(split)

        # Construct references
        references = []
        for oid in oids:
            rel = qrels_dict[qid][oid] if oid in qrels_dict[qid] else None
            gold = rel in gold_relations
            rank = topk_dict[qid].get(oid)  # Get returns None if the value is missing.
            reference = self.create_reference(oid, gold, rel, rank)
            references.append(reference)

        # Construct instance
        query_text = query_dict[qid]
        instance = Instance(input=query_text, references=references, split=split)
        vanilla_instance = cast(Instance, instance)
        return vanilla_instance

    def get_train_instance(self, qid: int) -> Instance:
        """ Create and return a train instance for the given qid.

        References are selected as follows:
            1. We select 1 correct reference, where the objects included
               corresponds to the best object for the given train query.
            2. We create 1 wrong reference, where the object included
               corresponds to a non-gold object for the given train query.
        """
        # Retrieve variables for the split.
        _, _, qrels_dict, topk, topk_dict, gold_relations = self.get_split_variables(TRAIN_SPLIT)

        # Get 1 correct Object ID.
        # - Retrieve the Object IDs relevant for the given query.
        # - Sort the retrieved Object IDs by their relevance, from the
        #   highest to the lowest.
        # - Pick the most relevant Object ID as the correct Object ID.
        relevant_oids = list(qrels_dict[qid].keys())
        relevant_oids = sorted(relevant_oids, key=lambda oid: qrels_dict[qid][oid], reverse=True)
        correct_oid = relevant_oids[0]

        # Get 1 wrong Object ID.
        # - Retrieve all Object IDs in the top-k dictionary for the query.
        # - Filter the Object IDs to only contain those with ranks between
        #   self.min_train_wrong_topk and self.train_topk, inclusive.
        # - Limit the filtered Object IDs to:
        #   * Those that aren't in the qrels dictionary for the query, or;
        #   * Those that are in the qrels dictionary for the query, granted
        #     that their relation to the query is not in the gold_relations
        #     list.
        # - Select the top Object ID from the filtered list as the wrong one.
        #   This ensures that the wrong objects we picked are competitive with
        #   the correct ones.
        filtered_oids = [oid for k, oid in topk_dict[qid].items() if self.min_train_wrong_topk <= k <= topk]
        wrong_oids = [
            oid for oid in filtered_oids if oid not in qrels_dict[qid] or qrels_dict[qid][oid] not in gold_relations
        ]
        wrong_oid = wrong_oids[0]

        # Combine the selected oids in a list, then sort the list following the shuffled list order, which ensures
        # that the order of the correct and wrong references are varied in the prompts. No further modification is
        # needed on the adapter side.
        oids = [correct_oid, wrong_oid]
        oids = sorted(oids, key=self.oid_index_dict.get)  # type: ignore

        # Create an instance and return.
        instance = self.create_instance(qid, TRAIN_SPLIT, oids)
        return instance

    def get_valid_instance(self, qid) -> Instance:
        """ Create and return the validation instance for the given qid.

        By default, we create a reference for each Object ID for which there
        is a judgment with respect to the provided Query ID.

        If self.use_topk_objects flag is set, we ensure that a reference is
        created for all the objects that appear in top self.valid_topk objects
        for the given validation query.
        """
        # Retrieve variables for the split.
        _, _, qrels_dict, topk, topk_dict, _ = self.get_split_variables(VALID_SPLIT)

        # Get Object IDs for which there is a judgment.
        relevant_oids = list(qrels_dict[qid].keys())

        # If the use_topk_objects flag is set, we retrieve the IDs of the topk
        # most relevant objects for the query. __init__ ensures that
        # self.valid_topk is set if use_topk_objects is set.
        topk_oids = [oid for k, oid in topk_dict[qid].items() if k <= topk] if self.use_topk_objects else []

        # Combine the list of Object IDs, remove duplicates, and sort the list
        # based on the order in the previously shuffled Object ID list.
        oids = relevant_oids + topk_oids
        oids = list(set(oids))
        oids = sorted(oids, key=self.oid_index_dict.get)  # type: ignore

        # Create instance.
        instance = self.create_instance(qid, VALID_SPLIT, oids)

        return instance

    def get_train_instances(self) -> List[Instance]:
        """ Get training instances. """
        qids = self.filter_qids(TRAIN_SPLIT, check_topk=True)
        instances = [self.get_train_instance(qid) for qid in qids[: self.num_train_queries]]
        return instances

    def get_valid_instances(self) -> List[Instance]:
        """ Get validation instances. """
        qids = self.filter_qids(VALID_SPLIT, check_topk=self.use_topk_objects)
        instances = [self.get_valid_instance(qid) for qid in qids]
        return instances

    def get_instances(self) -> List[Instance]:
        """ Get instances for this scenario.

        Refer to the documentation of the following methods for details on how
        the instances are created:
            * self.get_train_instances
            * self.get_valid_instances
        """
        hlog("MS MARCO Scenario: Preparing the datasets.")
        self.prepare_data()

        hlog("MS MARCO Scenario: Shuffling Object and Query IDs.")
        self.shuffle_ids()

        hlog("MS MARCO Scenario: Preparing training instances.")
        train_instances = self.get_train_instances()

        hlog("MS MARCO Scenario: Preparing validation instances.")
        valid_instances = self.get_valid_instances()

        instances = train_instances + valid_instances
        hlog("MS MARCO Scenario: Done preparing all instances.")

        return instances
