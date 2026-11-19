"""Association rules A -> B from frequent itemsets.

    support(A -> B)    = support(A ∪ B)
    confidence(A -> B) = support(A ∪ B) / support(A)        P(B | A)
    lift(A -> B)       = confidence(A -> B) / support(B)    P(B | A) / P(B); 1 means independent
    leverage(A -> B)   = support(A ∪ B) − support(A) support(B)
    conviction(A -> B) = (1 − support(B)) / (1 − confidence)  ∞ when the rule never fails

Every subset of a frequent itemset is frequent (Apriori property), so support(A) and
support(B) are always in the mined dictionary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Dict, FrozenSet, List


@dataclass(frozen=True)
class Rule:
    antecedent: FrozenSet[int]
    consequent: FrozenSet[int]
    support: float
    confidence: float
    lift: float
    leverage: float
    conviction: float

    def to_dict(self) -> dict:
        d = asdict(self)
        d["antecedent"], d["consequent"] = sorted(self.antecedent), sorted(self.consequent)
        return d


def generate_rules(supports: Dict[FrozenSet[int], float], min_confidence: float = 0.1, min_lift: float = 0.0,
                   max_consequent: int = 1) -> List[Rule]:
    rules = []
    for items, s in supports.items():
        if len(items) < 2:
            continue
        for k in range(1, min(max_consequent, len(items) - 1) + 1):
            for cons in map(frozenset, combinations(sorted(items), k)):
                ante = items - cons
                conf = s / supports[ante]
                if conf < min_confidence:
                    continue
                lift = conf / supports[cons]
                if lift < min_lift:
                    continue
                conv = float("inf") if conf >= 1 else (1 - supports[cons]) / (1 - conf)
                rules.append(Rule(ante, cons, s, conf, lift, s - supports[ante] * supports[cons], conv))
    return sorted(rules, key=lambda r: (-r.lift, -r.confidence, -r.support))
