from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import List, Optional
from helm.common.hierarchical_logger import hlog

import dacite
import re
import yaml


ROOT_CATEGORY = "Root"


@dataclass
class Expansion:
    """
    Example: text = "this is a ${Topic}"
    """
    text: str
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.children = re.split(r"(\${\w+})", self.text)
        self.categories = [x for x in [get_category(child) for child in self.children] if x is not None]


@dataclass(frozen=True)
class GrammarRule:
    """
    Represents a set of grammar rules:

            <category> -> expansions[0]
            <category> -> expansions[1]
            ...
    """
    category: str
    expansions: List[Expansion]
    tags: List[str] = field(default_factory=list)


def get_category(text: str) -> Optional[str]:
    """
    Example: "${Root}" => "Root"
    """
    if text.startswith("${") and text.endswith("}"):
        return text[2:-1]
    return None



@dataclass
class Grammar:
    rules: List[GrammarRule]

    def __post_init__(self):
        self.category_to_rules = defaultdict(list)
        for rule in self.rules:
            self.category_to_rules[rule.category].append(rule)


@dataclass(frozen=True)
class Derivation:
    # Exactly one of value of children is active
    value: Optional[str]
    children: Optional[List['Derivation']]
    tags: List[str]

    @property
    def is_leaf(self):
        return self.value is not None


def validate_grammar(grammar: Grammar):
    for rule in grammar.rules:
        for expansion in rule.expansions:
            # Make sure all categories are defined
            for category in expansion.categories:
                if category not in grammar.category_to_rules:
                    hlog(f"Category {category} is not defined")


def read_grammar(path: str) -> Grammar:
    with open(path) as f:
        raw = yaml.safe_load(f)
    grammar = dacite.from_dict(Grammar, raw)
    validate_grammar(grammar)
    return grammar


def generate_derivations(grammar: Grammar) -> List[Derivation]:
    def expand_rule_expansion(rule: GrammarRule, expansion: Expansion) -> List[Derivation]:
        results: List[Derivation] = [Derivation(value=None, children=[], tags=rule.tags + expansion.tags)]

        # Go through each item of the RHS of the rule and expand it, forming
        # the cross product as we go.
        for item in expansion.children:
            # Get list of candidate children
            candidates: List[Derivation] = []
            category = get_category(item)
            if category is None:
                # Terminal
                candidates = [Derivation(value=item, children=None, tags=[])]
            else:
                # Non-terminal
                candidates = expand_category(category)

            # Extend each derivation with each candidate children
            new_results: List[Derivation] = []
            for derivation in results:
                for child in candidates:
                    new_derivation = replace(derivation, children=derivation.children + [child])
                    new_results.append(new_derivation)
            results = new_results

        return results


    def expand_category(category: str) -> List[Derivation]:
        results: List[Derivation] = []
        for rule in grammar.category_to_rules[category]:
            for expansion in rule.expansions:
                results.extend(expand_rule_expansion(rule, expansion))
        return results

    return expand_category(ROOT_CATEGORY)


def get_values(derivation: Derivation) -> List[str]:
    """Return all the `values` that are collected recursively."""
    if derivation.is_leaf:
        return [derivation.value]
    values: List[str] = []
    for child in derivation.children:
        values.extend(get_values(child))
    return values


def get_tags(derivation: Derivation) -> List[str]:
    """Return all the `tags` that are collected recursively."""
    tags: List[str] = []
    tags.extend(derivation.tags)
    if derivation.children is not None:
        for child in derivation.children:
            tags.extend(get_tags(child))
    return tags
