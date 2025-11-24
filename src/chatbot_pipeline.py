from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from .ml_intent_classifier import IntentClassifier, IntentPrediction
from .rules_engine import RuleEngine, RuleMatchResult
from .config import ChatbotConfig

@dataclass
class ChatbotResponse:
    text: str
    intent: Optional[str] = None
    confidence: Optional[float] = None
    source: str = "ml"  # "rule" | "ml" | "fallback"

class Chatbot:
    def __init__(self, config: ChatbotConfig, rules_path: str, training_data_path: str) -> None:
        self.config = config
        self.rules = RuleEngine(rules_path)
        self.intent_clf = IntentClassifier()
        self.intent_clf.train_from_csv(training_data_path)

    def handle_message(self, user_text: str) -> ChatbotResponse:
        # 1. Apply rules first
        rule_result: RuleMatchResult | None = self.rules.evaluate(user_text)
        if rule_result:
            action_type = rule_result.action.get("type")

            # Action: direct reply from rule
            if action_type == "reply":
                return ChatbotResponse(
                    text=rule_result.action.get("response", ""),
                    source="rule"
                )

            # Action: set intent and let ML-style response layer handle it
            if action_type == "set_intent":
                intent = rule_result.action.get("intent")
                return self._respond_by_intent(
                    intent=intent,
                    confidence=None,
                    source="rule"
                )

        # 2. ML intent classification
        pred: IntentPrediction = self.intent_clf.predict(user_text)

        if pred.confidence < self.config.ml.min_confidence:
            # Low confidence fallback
            return ChatbotResponse(
                text="I'm not fully sure I understood. Can you rephrase?",
                intent=None,
                confidence=pred.confidence,
                source="fallback"
            )

        # 3. Response based on ML intent
        return self._respond_by_intent(
            intent=pred.intent,
            confidence=pred.confidence,
            source="ml"
        )

    def _respond_by_intent(
        self,
        intent: Optional[str],
        confidence: Optional[float],
        source: str
    ) -> ChatbotResponse:
        if intent == "greet":
            text = "Hi there! How can I help you today?"
        elif intent == "goodbye":
            text = "Goodbye! Have a great day."
        elif intent == "ask_bot_name":
            text = "I'm your Rules + ML assistant chatbot."
        elif intent == "account_help":
            text = "Sure, I can help with account issues. Can you describe the problem in more detail?"
        elif intent == "billing_issue":
            text = "Let me help with that billing issue. What exactly went wrong with the payment?"
        else:
            text = "I detected your question, but I’m still learning how to handle that topic."

        return ChatbotResponse(
            text=text,
            intent=intent,
            confidence=confidence,
            source=source
        )
