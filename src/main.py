"""RAG Chatbot - Main entry point."""

try:
    from src import config
    from src.chatbot import chat
except ModuleNotFoundError:
    import config
    from chatbot import chat


def main():
    """Run the RAG chatbot."""
    persist_dir = config.CHROMA_PERSIST_DIR
    chat_history = []

    print("=" * 50)
    print("RAG Chatbot - Welcome!")
    print("=" * 50)
    print(f"Vector store: {persist_dir}")
    print("\nType your question and press Enter.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ["quit", "exit", "q"]:
                print("\nChatbot: Goodbye!")
                break

            if not user_input:
                print("\nChatbot: Please enter a question.")
                continue

            result = chat(user_input, persist_dir, chat_history=chat_history)
            chat_history = result["chat_history"]

            print(f"\nChatbot: {result['response']}")
            if result["sources"]:
                print(f"Sources: {', '.join(result['sources'])}")
            print()

        except Exception as error:
            print(f"\nChatbot error: {error}")

        except KeyboardInterrupt:
            print("\n\nChatbot: Goodbye!")
            break


if __name__ == "__main__":
    main()
