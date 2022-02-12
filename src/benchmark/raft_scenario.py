import random
import os
import json
import tempfile
import datasets
from pathlib import Path
from common.general import ensure_file_downloaded
from copy import copy
from typing import List, Dict, Literal, Tuple
from common.hierarchical_logger import hlog
from .scenario import Scenario, Instance, Reference, TRAIN_TAG, VALID_TAG, TEST_TAG, CORRECT_TAG

PROMPT_INSTRUCTIONS = dict(
    ade_corpus_v2="""Label the sentence based on whether it is related to an adverse drug effect (ADE). Details are described below:
Drugs: Names of drugs and chemicals that include brand names, trivial names, abbreviations and systematic names were annotated. Mentions of drugs or chemicals should strictly be in a therapeutic context. This category does not include the names of metabolites, reaction byproducts, or hospital chemicals (e.g. surgical equipment disinfectants).
Adverse effect: Mentions of adverse effects include signs, symptoms, diseases, disorders, acquired abnormalities, deficiencies, organ damage or death that strictly occur as a consequence of drug intake.
Possible labels:
1. ADE-related
2. not ADE-related""",
    banking_77="""The following is a banking customer service query. Classify the query into one of the 77 categories available.
Possible labels:
1. Refund_not_showing_up
2. activate_my_card
3. age_limit
4. apple_pay_or_google_pay
5. atm_support
6. automatic_top_up
7. balance_not_updated_after_bank_transfer
8. balance_not_updated_after_cheque_or_cash_deposit
9. beneficiary_not_allowed
10. cancel_transfer
11. card_about_to_expire
12. card_acceptance
13. card_arrival
14. card_delivery_estimate
15. card_linking
16. card_not_working
17. card_payment_fee_charged
18. card_payment_not_recognised
19. card_payment_wrong_exchange_rate
20. card_swallowed
21. cash_withdrawal_charge
22. cash_withdrawal_not_recognised
23. change_pin
24. compromised_card
25. contactless_not_working
26. country_support
27. declined_card_payment
28. declined_cash_withdrawal
29. declined_transfer
30. direct_debit_payment_not_recognised
31. disposable_card_limits
32. edit_personal_details
33. exchange_charge
34. exchange_rate
35. exchange_via_app
36. extra_charge_on_statement
37. failed_transfer
38. fiat_currency_support
39. get_disposable_virtual_card
40. get_physical_card
41. getting_spare_card
42. getting_virtual_card
43. lost_or_stolen_card
44. lost_or_stolen_phone
45. order_physical_card
46. passcode_forgotten
47. pending_card_payment
48. pending_cash_withdrawal
49. pending_top_up
50. pending_transfer
51. pin_blocked
52. receiving_money
53. request_refund
54. reverted_card_payment?
55. supported_cards_and_currencies
56. terminate_account
57. top_up_by_bank_transfer_charge
58. top_up_by_card_charge
59. top_up_by_cash_or_cheque
60. top_up_failed
61. top_up_limits
62. top_up_reverted
63. topping_up_by_card
64. transaction_charged_twice
65. transfer_fee_charged
66. transfer_into_account
67. transfer_not_received_by_recipient
68. transfer_timing
69. unable_to_verify_identity
70. verify_my_identity
71. verify_source_of_funds
72. verify_top_up
73. virtual_card_not_working
74. visa_or_mastercard
75. why_verify_identity
76. wrong_amount_of_cash_received
77. wrong_exchange_rate_for_cash_withdrawal""",
    neurips_impact_statement_risks="""Label the impact statement based on whether it mentions a harmful application of the research done in the paper. Make sure the statement is sufficient to conclude there are harmful applications of the research being done, not a past risk that this research is solving.
Possible labels:
1. doesn't mention a harmful application
2. mentions a harmful application""",
    one_stop_english="""The following is an article sourced from The Guardian newspaper, and rewritten by teachers to suit three levels of adult English as Second Language (ESL) learners: elementary, intermediate, and advanced. Predict the level of the article.
Possible labels:
1. advanced
2. elementary
3. intermediate""",
    overruling="""In law, an overruling sentence is a statement that nullifies a previous case decision as a precedent, by a constitutionally valid statute or a decision by the same or higher ranking court which establishes a different rule on the point of law involved. Label the sentence based on whether it is overruling or not.
Possible labels:
1. not overruling
2. overruling""",
    semiconductor_org_types="""The dataset is a list of institutions that have contributed papers to semiconductor conferences in the last 25 years, as catalogued by IEEE and sampled randomly. The goal is to classify the institutions into one of three categories: "university", "company" or "research institute".
Possible labels:
1. company
2. research institute
3. university""",
    systematic_review_inclusion="""Identify whether this paper should be included in a meta-review which includes the findings of systematic reviews on interventions designed to promote charitable donations.
Included reviews should describe monetary charitable donations, assess any population of participants in any context, and be peer reviewed and written in English.
They should not report new data, be non-systematic reviews, consider cause-related marketing or other kinds of prosocial behaviour.
Possible labels:
1. included
2. not included""",
    tai_safety_research="""Transformative AI (TAI) is defined as AI that precipitates a transition comparable to (or more significant than) the agricultural or industrial revolution. Label a paper as "TAI safety research" if:
1. The contents of the paper are directly motivated by, and substantively inform, the challenge of ensuring good outcomes for TAI,
2. There is substantive content on AI safety, not just AI capabilities,
3. The intended audience is the community of researchers,
4. It meets a subjective threshold of seriousness/quality,
5. Peer review is not required.
Possible labels:
1. TAI safety research
2. not TAI safety research""",
    terms_of_service="""Label the sentence from a Terms of Service based on whether it is potentially unfair. If it seems clearly unfair, mark it as potentially unfair.
According to art. 3 of the Directive 93/13 on Unfair Terms in Consumer Contracts, a contractual term is unfair if: 1) it has not been individually negotiated; and 2) contrary to the requirement of good faith, it causes a significant imbalance in the parties rights and obligations, to the detriment of the consumer.
Details on types of potentially unfair clauses are found below:
The jurisdiction clause stipulates what courts will have the competence to adjudicate disputes under the contract. Jurisdiction clauses giving consumers a right to bring disputes in their place of residence were marked as clearly fair, whereas clauses stating that any judicial proceeding takes a residence away were marked as clearly unfair.
The choice of law clause specifies what law will govern the contract, meaning also what law will be applied in potential adjudication of a dispute arising under the contract. Clauses defining the applicable law as the law of the consumer's country of residence were marked as clearly fair. In every other case, the choice of law clause was considered as potentially unfair.
The limitation of liability clause stipulates that the duty to pay damages is limited or excluded, for certain kind of losses, under certain conditions. Clauses that explicitly affirm non-excludable providers' liabilities were marked as clearly fair. Clauses that reduce, limit, or exclude the liability of the service provider were marked as potentially unfair when concerning broad categories of losses or causes of them.
The unilateral change clause specifies the conditions under which the service provider could amend and modify the terms of service and/or the service itself. Such clause was always considered as potentially unfair.
The unilateral termination clause gives provider the right to suspend and/or terminate the service and/or the contract, and sometimes details the circumstances under which the provider claims to have a right to do so.
The contract by using clause stipulates that the consumer is bound by the terms of use of a specific service, simply by using the service, without even being required to mark that he or she has read and accepted them. We always marked such clauses as potentially unfair.
The content removal gives the provider a right to modify/delete user's content, including in-app purchases, and sometimes specifies the conditions under which the service provider may do so.
The arbitration clause requires or allows the parties to resolve their disputes through an arbitration process, before the case could go to court. Clauses stipulating that the arbitration should take place in a state other then the state of consumer's residence or be based on arbiter's discretion were marked as clearly unfair. Clauses defining arbitration as fully optional were marked as clearly fair.
Possible labels:
1. not potentially unfair
2. potentially unfair""",
    tweet_eval_hate="""Label whether the following tweet contains hate speech against either immigrants or women. Hate Speech (HS) is commonly defined as any communication that disparages a person or a group on the basis of some characteristic such as race, color, ethnicity, gender, sexual orientation, nationality, religion, or other characteristics.
Possible labels:
1. hate speech
2. not hate speech""",
    twitter_complaints="""A complaint presents a state of affairs which breaches the writer’s favorable expectation. Label the tweet text based on whether it contains a complaint.
Possible labels:
1. complaint
2. no complaint""",
)


def get_raft_instructions(subset: str, load_remote=False):
    # Default to local instructions
    instructions = PROMPT_INSTRUCTIONS

    if load_remote:
        tmp_dir = tempfile.gettempdir()
        prompt_construction_settings_path = os.path.join(tmp_dir, "prompt_construction_settings.jsonl")
        ensure_file_downloaded(
            source_url="https://raw.githubusercontent.com/oughtinc/raft-baselines/master/src/raft_baselines/data/prompt_construction_settings.jsonl",
            target_path=prompt_construction_settings_path,
        )
        with open(prompt_construction_settings_path, "r") as f:
            field_ordering, instructions = map(json.loads, f.read().split("\n"))

    assert subset in instructions, "Unknown subset: {}".format(subset)
    return instructions[subset]


class RAFTScenario(Scenario):
    """
    RAFT: A Real-World Few-Shot Text Classification Benchmark
        https://arxiv.org/abs/2109.14076

    Official website for RAFT dataset:
        https://raft.elicit.org/

    Dataset summary:
        https://huggingface.co/datasets/ought/raft/blob/main/README.md

    Prompts are adapted from:
        https://github.com/oughtinc/raft-baselines/tree/master/example_prompts

    Subsets:
        ade_corpus_v2
        banking_77
        neurips_impact_statement_risks
        one_stop_english
        overruling
        semiconductor_org_types
        systematic_review_inclusion
        tai_safety_research
        terms_of_service
        tweet_eval_hate
        twitter_complaints
    """

    name = "raft"
    description = "Real-world Annotated Few-shot Tasks (RAFT)"
    tags = ["text_classification", "robustness"]

    def __init__(self, subset: str, random_seed=42):
        self.subset = subset
        self.random_seed = random_seed
        self.instructions = get_raft_instructions(subset)
        self.fields = None
        random.seed(random_seed)

    def load_prompt_construction_settings(self):
        # Skip if already loaded
        if self.fields:
            return
        # Download prompt construction settings
        prompt_construction_settings_path = str(Path(self.output_path) / "data" / "prompt_construction_settings.jsonl")
        ensure_file_downloaded(
            source_url="https://raw.githubusercontent.com/oughtinc/raft-baselines/master/src/raft_baselines/data/prompt_construction_settings.jsonl",
            target_path=prompt_construction_settings_path,
        )
        with open(prompt_construction_settings_path, "r") as f:
            FIELD_ORDERING, _ = map(json.loads, f.read().split("\n"))
        self.fields = FIELD_ORDERING[self.subset]

    def get_instances(self) -> List[Instance]:
        self.load_prompt_construction_settings()
        cache_dir = str(Path(self.output_path) / "data")
        # Download raw data
        # TODO: Only using labeled instances now. Check if we can get the hidden test set labels.
        all_usable_dataset = datasets.load_dataset("ought/raft", self.subset, cache_dir=cache_dir, split="train")
        assert isinstance(all_usable_dataset, datasets.Dataset)
        dataset = all_usable_dataset.train_test_split(test_size=0.2, seed=self.random_seed)
        train_dataset, test_dataset = dataset["train"], dataset["test"]
        class_label_to_string = train_dataset.features["Label"].int2str

        dataset_splits: Dict[str, datasets.Dataset] = {
            TRAIN_TAG: train_dataset,
            TEST_TAG: test_dataset,
        }

        # # Read all instances
        instances: List[Instance] = []
        for tag, subset in dataset_splits.items():
            for x in subset:
                assert self.fields is not None, "Field ordering not loaded"
                prompt = "\n".join([f"{field}: {x[field]}" for field in self.fields])
                instance = Instance(
                    input=prompt,
                    references=[Reference(output=class_label_to_string(x["Label"]), tags=[CORRECT_TAG])],
                    tags=[tag],
                )
                instances.append(instance)

        return instances
