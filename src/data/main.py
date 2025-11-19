from __future__ import annotations
from .config import ChatbotConfig
from .chatbot_pipeline import Chatbot

def main() -> None:
    config = ChatbotConfig()
    bot = Chatbot(
        config=config,
        rules_path="src/data/rules.yml",
        training_data_path="src/data/training_data.csv"
    )

    print("Chatbot with Rules + ML (type 'quit' to exit)")
    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in {"quit", "exit"}:
            print("Bot: Goodbye!")
            break

        response = bot.handle_message(user_text)
        meta = f"[intent={response.intent}, conf={response.confidence}, source={response.source}]"
        print(f"Bot: {response.text} {meta}")

if __name__ == "__main__":
    main()
