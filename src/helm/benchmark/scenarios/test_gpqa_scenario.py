import pytest
from tempfile import TemporaryDirectory

from helm.benchmark.scenarios.gpqa_scenario import GPQAScenario
from helm.benchmark.scenarios.scenario import CORRECT_TAG, Input, Output, Reference


@pytest.mark.scenarios
def test_gpqa_scenario():
    with TemporaryDirectory() as tmpdir:
        scenario = GPQAScenario(subset="gpqa_main")
        instances = scenario.get_instances(tmpdir)
        assert len(instances) == 448
        assert instances[0].input == Input(
            text="A large gene has dozens of exons, of which the central ones code for folded triple helical repeats that connect the cytoskeleton with sarcolemma and extracellular space. Each exon usually codes for one folded triple alpha helix. The most common mutations of the gene are central exon deletions that create out-of-frame peptides and progressive degenerative organ waste. A solution is to deliver a Morpholino that recognizes the 5' end of the out-of-frame exon in pre-mRNA. The molecule prevents binding of the spliceosome and creates exon skipping and in-frame joining. Several missing exons are well tolerated by an organism. Which structure below is not involved in the proposed therapy?"
        )
        assert instances[0].references == [
            Reference(output=Output(text="R-loops"), tags=[CORRECT_TAG]),
            Reference(output=Output(text="lariat"), tags=[]),
            Reference(output=Output(text="polyA tail"), tags=[]),
            Reference(output=Output(text="antisense"), tags=[]),
        ]
        assert instances[0].split == "train"

        scenario = GPQAScenario(subset="gpqa_diamond")
        instances = scenario.get_instances(tmpdir)
        assert len(instances) == 198
        assert instances[0].input == Input(
            text="Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?"
        )
        assert instances[0].references == [
            Reference(output=Output(text="10^-4 eV"), tags=[CORRECT_TAG]),
            Reference(output=Output(text="10^-11 eV"), tags=[]),
            Reference(output=Output(text="10^-8 eV"), tags=[]),
            Reference(output=Output(text="10^-9 eV"), tags=[]),
        ]
        assert instances[0].split == "train"

        scenario = GPQAScenario(subset="gpqa_extended")
        instances = scenario.get_instances(tmpdir)
        assert len(instances) == 546
        assert instances[0].input == Input(
            text="A large gene has dozens of exons, of which the central ones code for folded triple helical repeats that connect the cytoskeleton with sarcolemma and extracellular space. Each exon usually codes for one folded triple alpha helix. The most common mutations of the gene are central exon deletions that create out-of-frame peptides and progressive degenerative organ waste. A solution is to deliver a Morpholino that recognizes the 5' end of the out-of-frame exon in pre-mRNA. The molecule prevents binding of the spliceosome and creates exon skipping and in-frame joining. Several missing exons are well tolerated by an organism. Which structure below is not involved in the proposed therapy?"
        )
        assert instances[0].references == [
            Reference(output=Output(text="R-loops"), tags=[CORRECT_TAG]),
            Reference(output=Output(text="lariat"), tags=[]),
            Reference(output=Output(text="polyA tail"), tags=[]),
            Reference(output=Output(text="antisense"), tags=[]),
        ]
        assert instances[0].split == "train"
