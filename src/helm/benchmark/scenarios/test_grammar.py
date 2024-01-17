from helm.benchmark.scenarios.grammar import (
    Grammar,
    GrammarRule,
    Expansion,
    read_grammar,
    ROOT_CATEGORY,
    generate_derivations,
    get_values,
    get_tags,
)


def test_generate():
    grammar = Grammar(
        rules=[
            GrammarRule(
                category=ROOT_CATEGORY,
                expansions=[
                    Expansion(text="hello"),
                    Expansion(text="hi ${Name} and ${Name}"),
                ],
                tags=["t1"],
            ),
            GrammarRule(
                category="Name",
                expansions=[
                    Expansion(text="Alice"),
                    Expansion(text="Bob"),
                    Expansion(text="Carol"),
                ],
                tags=["name"],
            ),
        ]
    )

    derivations = generate_derivations(grammar)
    assert len(derivations) == 1 + 3 * 3

    texts = ["".join(get_values(derivation)) for derivation in derivations]
    assert "hello" in texts
    assert "hi Alice and Bob" in texts

    tags = [get_tags(derivation) for derivation in derivations]
    assert ["t1"] in tags
    assert ["t1", "name", "name"] in tags


def test_read_grammar():
    read_grammar("src/helm/benchmark/scenarios/best_chatgpt_prompts.yaml")
    read_grammar("src/helm/benchmark/scenarios/compositional_instructions.yaml")
