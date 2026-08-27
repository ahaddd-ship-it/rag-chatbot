"""RAG Chatbot - Main entry point."""

import os
from src.chatbot import chat


def main():
    """Run the RAG chatbot."""
    persist_dir = os.path.join(os.path.dirname(__file__), "..", "chroma_db")

    print("=" * 50)
    print("RAG Chatbot - Welcome!")
    print("=" * 50)
    print(f"Vector store: {persist_dir}")
    print("\nType your question and press Enter.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nChatbot: Goodbye!")
                break

            # Generate response
            response = chat(user_input, persist_dir)

            print(f"\nChatbot: {response}")
            print()

        except KeyboardInterrupt:
            print("\n\nChatbot: Goodbye!")
            break


if __name__ == "__main__":
    main()
