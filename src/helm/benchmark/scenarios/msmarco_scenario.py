import csv
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union

from helm.common.general import ensure_file_downloaded, ensure_directory_exists
from helm.common.hierarchical_logger import hlog
from helm.benchmark.scenarios.scenario import (
    Scenario,
    Instance,
    Reference,
    TRAIN_SPLIT,
    VALID_SPLIT,
    CORRECT_TAG,
    make_rank_tag,
    make_relevance_tag,
    Input,
    Output,
)


class MSMARCOScenario(Scenario):
    """Scenario implementing MS MARCO challenge tasks.

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

            Passage: Its 25 drops per ml, you guys are all wrong. If it is water, the standard was changed 15 - 20 years ago to make 20 drops = 1mL. The viscosity of most things is temperature dependent, so this would be at room temperature. Hope this helps.
            Query: how many eye drops per ml
            Does the passage answer the query?
            Answer: Yes

            Passage: RE: How many eyedrops are there in a 10 ml bottle of Cosopt? My Kaiser pharmacy insists that 2 bottles should last me 100 days but I run out way before that time when I am using 4 drops per day.In the past other pharmacies have given me 3 10-ml bottles for 100 days.E: How many eyedrops are there in a 10 ml bottle of Cosopt? My Kaiser pharmacy insists that 2 bottles should last me 100 days but I run out way before that time when I am using 4 drops per day.
            Query: how many eye drops per ml
            Does the passage answer the query?
            Answer: No

            Passage: : You can transfer money to your checking account from other Wells Fargo. accounts through Wells Fargo Mobile Banking with the mobile app, online, at any. Wells Fargo ATM, or at a Wells Fargo branch. 1 Money in — deposits.
            Query: can you open a wells fargo account online
            Does the passage answer the query?
            Answer: No

            Passage: You can open a Wells Fargo banking account from your home or even online. It is really easy to do, provided you have all of the appropriate documentation. Wells Fargo has so many bank account options that you will be sure to find one that works for you. They offer free checking accounts with free online banking.
            Query: can you open a wells fargo account online
            Does the passage answer the query?
            Answer: Yes

            Passage: SIZE: There are few measurements available for this bear. Adult spectacled bears can weigh between 140 and 385 pounds (63-173 kg). However, the body length of adults is about 150 to 180 centimeters (60 to 72 inches) and males may be 30 to 40 percent larger than females.
            Query: how much does a spectacled bear weigh
            Does the passage answer the query?
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

            document: The document files contain all the documents that could be
                ranked for a question, each specified with an document ID
                (docid). For example, for the passage track, the documents would
                be passages.
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
                document with ID docid. The relevance values can have different
                meanings depending on the split and the track. Note that not
                all queries would have corresponding query relevances in the
                accompanied file. Also note that multiple documents may have the
                same relevance value with a qiven query.
            topk: Each query file is accompanied by a top-k file, which lists
                the IDs of the top k best documents for a query with their
                accompanied rank. The top documents for each query were selected
                using the BM25 algorithm. The notebook used to generate the
                top-k files used in this scenario can be found at the
                Benchmarking CodaLab Worksheet. Note that not all queries would
                have a corresponding top-k documents in the accompanied file.

            |      LOCAL FILE NAME        |  TRACK  |  TRACK  |              CONTENT              |        FORMAT         |              Host              | Notes |  # noqa
            | passage_document.tsv        | passage |    -    | 8,841,823 passages                | <docid> <text>        | Benchmarking CodaLab Worksheet | (1)   |  # noqa
            | passage_train_queries.tsv   | passage |    -    | 808,731   queries                 | <qid> <text>          | Official MS MARCO Website      |       |  # noqa
            | passage_train_qrels.tsv     | passage |    -    | 532,761   query relations         | <qid> 0 <docid> <rel> | Official MS MARCO Website      | (2)   |  # noqa
            | passage_train_topk.tsv      | passage |    -    | 20        top documents per query | <qid> <docid> <rank>  | Benchmarking CodaLab Worksheet | (3)   |  # noqa
            | passage_regular_queries.tsv | passage | regular | 6980      queries                 | <qid> <text>          | Official MS MARCO Website      | (4)   |  # noqa
            | passage_regular_qrels.tsv   | passage | regular | 7437      query relations         | <qid> 0 <docid> <rel> | Official MS MARCO Website      | (2)   |  # noqa
            | passage_regular_topk.tsv    | passage | regular | 1000      top documents per query | <qid> <docid> <rank>  | Benchmarking CodaLab Worksheet |       |  # noqa
            | passage_trec_queries.tsv    | passage | trec    | 200       queries                 | <qid> <text>          | Official MS MARCO Website      |       |  # noqa
            | passage_trec_qrels.tsv      | passage | trec    | 502,982   query relations         | <qid> 0 <docid> <rel> | Official MS MARCO Website      | (5)   |  # noqa
            | passage_trec_topk.tsv       | passage | trec    | 1000      top documents per query | <qid> <docid> <rank>  | Benchmarking CodaLab Worksheet |       |  # noqa

                Notes:
                    (1) We use a pre-processed version of the passage
                        collection, introduced in (MacAvaney, et. al. 2021),
                        which greatly improves the quality of the passages. The
                        cleaned collection is hosted on the Benchmarking CodaLab
                        Worksheet as there is no other reliable publicly hosted
                        copy.
                    (2) The only relevance values is 1, which indicates that the
                        document with the ID the docid is the gold match for the
                        query with the ID qid.
                    (3) The number of top documents ranked was limited to 20 for
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
            #VR      | Number of validation requests. For each effective
                       validation query, multiple requests would be created,
                       governed by the provided parameters.

        | Baseline                |  #VR   | Parameters
        | regular_topk            | 10,000 | track=regular,use_topk_passages=True,valid_topk=50
        | regular_topk_with_qrels | 10,085 | track=regular,use_qrels_passages=True,use_topk_passages=True,valid_topk=50
        | trec_topk               | 4300   | track=trec,use_topk_passages=True,valid_topk=100
        | trec_qrels              | 9260   | track=trec,use_qrels_passages=True

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

    DATA_URIS: Dict[Union[Tuple[str], Tuple[str, str]], str] = {
        ("documents",): CODALAB_URI_TEMPLATE.format(bundle="0x50d32fc56ad04dd89510bf86f9c1c9d3"),
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
    DATASET_SEPARATOR: Dict[Union[Tuple[str], Tuple[str, str]], str] = defaultdict(lambda: "\t")
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

    def __init__(self, track: str, valid_topk: Optional[int] = None):
        """The constructor for the MSMARCOScenario.

        Args:
            track: Name of the passage track. Currently, available values are
            as follows:
                    "regular": The regular passage track.
                    "trec": The TREC passage track.
            valid_topk: If set, specifies the number of top documents for which the
                validation instances will be created. Must be in the range
                [self.MIN_TOPK, self.MAX_VALID_TOPK].
        """
        super().__init__()

        # Input validation
        assert track in self.TRACK_NAMES
        self.track: str = track

        self.valid_topk: Optional[int] = valid_topk
        if self.valid_topk is not None:
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
        self.docid_index_dict: Dict[int, int]  # Used to efficiently randomize Document ID lists.

        # Data structures that will be populated once the scenario is run.
        self.document_dict: Dict[int, str] = {}
        self.docids: List[int] = []
        self.query_dicts: Dict[str, Dict[int, str]] = {TRAIN_SPLIT: {}, VALID_SPLIT: {}}
        self.qrels_dicts: Dict[str, Dict[int, Dict[int, int]]] = {TRAIN_SPLIT: {}, VALID_SPLIT: {}}
        self.topk_dicts: Dict[str, Dict[int, Dict[int, int]]] = {TRAIN_SPLIT: {}, VALID_SPLIT: {}}
        self.qids: Dict[str, List[int]] = {TRAIN_SPLIT: [], VALID_SPLIT: []}

    def download_file(self, urlstring: str, target_file_name: str, output_path: str) -> str:
        """Download the resource at urlstring and return the absolute path.

        Downloaded file is saved to a directory named 'data' in
        output_path.
        """
        data_path = os.path.join(output_path, "data")
        ensure_directory_exists(data_path)
        target_file_path = os.path.join(data_path, target_file_name)
        ensure_file_downloaded(source_url=urlstring, target_path=target_file_path)
        return target_file_path

    def download_helper(self, data_key: Union[Tuple[str], Tuple[str, str]], output_path: str) -> str:
        """Call download_file for self.DATA_URIS[data_key] and return the file path to the downloaded file."""
        # Download the file
        urlstring = self.DATA_URIS[data_key]
        target_file_name = f"{'_'.join(data_key)}.tsv"
        file_path = self.download_file(urlstring, target_file_name, output_path)

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
        """Read the provided file as an id to item dictionary.

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
        """Read the provided file as a qrels dictionary.

        Given a file with rows in the following format:
            <qid>   0   <docid>   <rel>

        Return a dictionary of the form:
            {
                <qid>: {
                    <docid>: <rel>,
                    ...
                },
                ...
            }
        """
        qrels_dict: Dict[int, Dict[int, int]] = defaultdict(dict)
        with open(file_path, encoding="utf-8") as f:
            for qid, _, docid, rel in csv.reader(f, delimiter=delimiter):
                qrels_dict[int(qid)][int(docid)] = int(rel)
        qrels_dict = {k: v for k, v in qrels_dict.items()}  # Convert to regular dict
        return qrels_dict

    @staticmethod
    def create_topk_dict(file_path: str, delimiter: str = "\t") -> Dict[int, Dict[int, int]]:
        """Read the provided file as a topk dictionary.

        Given a file with rows in the following format:
            <qid>   <docid>   <rank>

        Return a dictionary of the form:
            {
                <qid>: {
                    <docid>: <rank>,
                    ...
                },
                ...
            }
        """
        topk_dict: Dict[int, Dict[int, int]] = defaultdict(dict)
        with open(file_path, encoding="utf-8") as f:
            for qid, docid, rank in csv.reader(f, delimiter=delimiter):
                topk_dict[int(qid)][int(docid)] = int(rank)
        topk_dict = {k: v for k, v in topk_dict.items()}  # Convert the defaultdict to a regular dict
        return topk_dict

    def prepare_data(self, output_path):
        """Download and load all the data."""
        # Passages
        self.document_dict = self.create_id_item_dict(self.download_helper(("documents",), output_path))
        self.docids = list(self.document_dict.keys())

        # Train queries
        self.query_dicts[TRAIN_SPLIT] = self.create_id_item_dict(
            self.download_helper((TRAIN_SPLIT, "queries"), output_path)
        )
        self.qrels_dicts[TRAIN_SPLIT] = self.create_qrels_dict(
            self.download_helper((TRAIN_SPLIT, "qrels"), output_path)
        )
        self.topk_dicts[TRAIN_SPLIT] = self.create_topk_dict(self.download_helper((TRAIN_SPLIT, "topk"), output_path))
        self.qids[TRAIN_SPLIT] = list(self.query_dicts[TRAIN_SPLIT].keys())

        # Validation queries
        self.query_dicts[VALID_SPLIT] = self.create_id_item_dict(
            self.download_helper((self.track, "queries"), output_path)
        )
        self.qrels_dicts[VALID_SPLIT] = self.create_qrels_dict(self.download_helper((self.track, "qrels"), output_path))
        self.topk_dicts[VALID_SPLIT] = self.create_topk_dict(self.download_helper((self.track, "topk"), output_path))
        self.qids[VALID_SPLIT] = list(self.query_dicts[VALID_SPLIT].keys())

    def shuffle_ids(self):
        """Shuffle Document and Query ID lists.

        This is the only place we perform shuffling throughout the scenario,
        which allows us to make use of caching even if the scenario is run with
        different user parameters.
        """
        self.random.shuffle(self.docids)
        self.docid_index_dict = {docid: ind for ind, docid in enumerate(self.docids)}
        self.random.shuffle(self.qids[TRAIN_SPLIT])
        self.random.shuffle(self.qids[VALID_SPLIT])

    def get_split_variables(self, split):
        """Return variables storing data for the given split."""
        qids = self.qids[split]
        query_dict = self.query_dicts[split]
        qrels_dict = self.qrels_dicts[split]
        topk = self.train_topk if split == TRAIN_SPLIT else self.valid_topk
        topk_dict = self.topk_dicts[split]
        gold_relations = set(self.gold_relations[split])
        return qids, query_dict, qrels_dict, topk, topk_dict, gold_relations

    def filter_qids(self, split: str, check_topk: bool = True) -> List[int]:
        """Return filtered Query IDs for the provided split.

        We filter each query based on the following conditions:
            (1) A query must have a corresponding query relations (qrels)
                dictionary, which specifies contain an answer for a query. We
                need to perform this check because the qrels dictionaries
                provided as part of MS MARCO tasks aren't guaranteed to have an
                entry for each query.
            (2) The qrels dictionary corresponding to the query must identify at
                least 1 gold document. This is so that for each instance we
                create, we ensure that there is at least 1 reference with a
                correct label.
            (3) If check_topk flag is set:
                    (a) We ensure that there is a corresponding topk dictionary
                        for the query. The topk dictionary tells us which
                        documents are most likely candidates for a given query.
                    (b) We ensure that the corresponding topk dictionary ranks
                        at least self.train_topk or self.valid_topk documents
                        depending on the specified split.
        """
        # Retrieve variables for the split.
        qids, _, qrels_dict, topk, topk_dict, gold_relations = self.get_split_variables(split)

        # (1) Ensure that there is a query relations dictionary for each query.
        filtered_qids = [qid for qid in qids if qid in qrels_dict]
        # (2) Ensure that the query relations specified include at least 1 gold document.
        filtered_qids = [qid for qid in filtered_qids if gold_relations.intersection(set(qrels_dict[qid].values()))]
        # (3) Check topk.
        if check_topk:
            # (3a) Ensure that there is a topk dictionary for each query.
            filtered_qids = [qid for qid in filtered_qids if qid in topk_dict]
            # (3b) Ensure that there are at least topk documents in the topk dictionary.
            filtered_qids = [qid for qid in filtered_qids if len(topk_dict[qid]) >= topk]

        return filtered_qids

    def create_reference(self, docid: int, gold: bool, rel: Optional[int], rank: Optional[int]) -> Reference:
        """Create and return a reference made using the provided parameters."""
        # Create tags
        tags = []
        # docid is extra information not needed on the metric sode - we are
        # including it to ensure that MS MARCO documents can be easily located
        # later on.
        tags.append(f"docid={docid}")
        if gold:
            tags.append(CORRECT_TAG)  # Correctness
        if rel:
            tags.append(make_relevance_tag(relevance=rel))  # Relevance
        if rank:
            tags.append(make_rank_tag(rank=rank))  # Top-k rank

        # Create the reference with the document
        document_text: str = self.document_dict[docid]
        reference = Reference(Output(text=document_text), tags=tags)
        return reference

    def create_instance(self, qid: int, split: str, docids: List[int]) -> Instance:
        """Create and return an instance made using the provided parameters."""
        # Retrieve variables for the split.
        _, query_dict, qrels_dict, _, topk_dict, gold_relations = self.get_split_variables(split)

        # Construct references
        references = []
        for docid in docids:
            rel = qrels_dict[qid][docid] if docid in qrels_dict[qid] else None
            gold = rel in gold_relations
            rank = topk_dict[qid].get(docid)  # Get returns None (e.g., for qrels-only docids) if the value is missing.
            reference = self.create_reference(docid, gold, rel, rank)
            references.append(reference)

        return Instance(Input(text=query_dict[qid]), references=references, split=split)

    def get_train_instance(self, qid: int) -> Instance:
        """Create and return a train instance for the given qid.

        References are selected as follows:
            1. We select 1 correct reference, where the documents included
               corresponds to the best document for the given train query.
            2. We create 1 wrong reference, where the document included
               corresponds to a non-gold document for the given train query.
        """
        # Retrieve variables for the split.
        _, _, qrels_dict, topk, topk_dict, gold_relations = self.get_split_variables(TRAIN_SPLIT)

        # Get 1 correct Document ID.
        # - Retrieve the Document IDs relevant for the given query.
        # - Sort the retrieved Document IDs by their relevance, from the
        #   highest to the lowest.
        # - Pick the most relevant Document ID as the correct Document ID.
        qrels_docids = list(qrels_dict[qid].keys())
        qrels_docids = sorted(qrels_docids, key=lambda docid: qrels_dict[qid][docid], reverse=True)
        correct_docid = qrels_docids[0]

        # Get 1 wrong Document ID.
        # - Retrieve all Document IDs in the top-k dictionary for the query.
        # - Filter the Document IDs to only contain those with ranks between
        #   self.min_train_wrong_topk and self.train_topk, inclusive.
        # - Limit the filtered Document IDs to:
        #   * Those that aren't in the qrels dictionary for the query, or;
        #   * Those that are in the qrels dictionary for the query, granted
        #     that their relation to the query is not in the gold_relations
        #     list.
        # - Select the top Document ID from the filtered list as the wrong one.
        #   This ensures that the wrong documents we picked are competitive with
        #   the correct ones.
        filtered_docids = [docid for docid, k in topk_dict[qid].items() if self.min_train_wrong_topk <= k <= topk]
        wrong_docids = [
            docid
            for docid in filtered_docids
            if docid not in qrels_dict[qid] or qrels_dict[qid][docid] not in gold_relations
        ]
        wrong_docid = wrong_docids[0]

        # Combine the selected docids in a list, then sort the list following
        # the shuffled list order, which ensures that the order of the correct
        # and wrong references are varied in the prompts. No further
        # modification is needed on the adapter side.
        docids = [correct_docid, wrong_docid]
        docids = sorted(docids, key=self.docid_index_dict.get)  # type: ignore

        # Create an instance and return.
        instance = self.create_instance(qid, TRAIN_SPLIT, docids)
        return instance

    def get_valid_instance(self, qid) -> Instance:
        """Create and return the validation instance for the given qid.

        By default, we create a reference for each Document ID for which there
        is a judgment with respect to the provided Query ID.

        If self.valid_topk is not None, we ensure that a reference is created
        for all the documents that appear in top self.valid_topk documents for
        the given validation query.
        """
        # Retrieve variables for the split.
        _, _, qrels_dict, topk, topk_dict, _ = self.get_split_variables(VALID_SPLIT)

        # Get Document IDs for which there is a judgment.
        qrels_docids = list(qrels_dict[qid].keys())

        # If self.valid_topk is not None, we retrieve the IDs of the topk
        # most relevant documents for the query.
        topk_docids = [docid for docid, k in topk_dict[qid].items() if k <= topk] if self.valid_topk is not None else []

        # Combine the list of Document IDs, remove duplicates, and sort the list
        # based on the order in the previously shuffled Document ID list.
        docids = qrels_docids + topk_docids
        docids = list(set(docids))
        docids = sorted(docids, key=self.docid_index_dict.get)  # type: ignore

        # Create instance.
        instance = self.create_instance(qid, VALID_SPLIT, docids)

        return instance

    def get_train_instances(self) -> List[Instance]:
        """Get training instances."""
        qids = self.filter_qids(TRAIN_SPLIT, check_topk=True)
        instances = [self.get_train_instance(qid) for qid in qids[: self.num_train_queries]]
        return instances

    def get_valid_instances(self) -> List[Instance]:
        """Get validation instances."""
        qids = self.filter_qids(VALID_SPLIT, check_topk=self.valid_topk is not None)
        instances = [self.get_valid_instance(qid) for qid in qids]
        return instances

    def get_instances(self, output_path: str) -> List[Instance]:
        """Get instances for this scenario.

        Refer to the documentation of the following methods for details on how
        the instances are created:
            * self.get_train_instances
            * self.get_valid_instances
        """
        hlog("Preparing the datasets.")
        self.prepare_data(output_path)

        hlog("Shuffling Document and Query IDs.")
        self.shuffle_ids()

        hlog("Preparing training instances.")
        train_instances = self.get_train_instances()

        hlog("Preparing validation instances.")
        valid_instances = self.get_valid_instances()

        return train_instances + valid_instances
