from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import yaml

@dataclass
class RuleMatchResult:
    rule_name: str
    action: Dict[str, Any]

class RuleEngine:
    def __init__(self, rules_path: str) -> None:
        with open(rules_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self.rules = data.get("rules", [])

    def _text_contains_any(self, text: str, patterns: List[str]) -> bool:
        text_lower = text.lower()
        return any(p.lower() in text_lower for p in patterns)

    def evaluate(self, user_text: str) -> Optional[RuleMatchResult]:
        for rule in self.rules:
            cond = rule.get("conditions", {})
            contains_any = cond.get("contains_any", [])

            if contains_any and not self._text_contains_any(user_text, contains_any):
                continue

            # You can extend with more condition types later (regex, entities, user state, etc.)
            return RuleMatchResult(rule_name=rule.get("name", ""), action=rule.get("action", {}))

        return None
