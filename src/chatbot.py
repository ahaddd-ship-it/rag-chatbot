"""RAG Chatbot - Query vector store and generate responses."""

import os
from typing import Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def get_vector_store(persist_dir: str) -> Chroma:
    """Load or create the vector store.

    Args:
        persist_dir: Directory where the vector store is persisted.

    Returns:
        Chroma vector store instance.
    """
    # Create embeddings for consistency
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    # Load existing vector store or create new one
    try:
        print(f"Loading vector store from {persist_dir}...")
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        print("Vector store loaded successfully.")
    except Exception as e:
        print(f"Could not load existing vector store: {e}")
        print("Creating new vector store...")
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            persistence="memory",  # Don't persist during creation
        )
        print("New vector store created.")

    return vectorstore


def retrieve_context(query: str, vectorstore: Chroma, top_k: int = 3) -> list:
    """Retrieve relevant context from the vector store.

    Args:
        query: User's search query.
        vectorstore: Chroma vector store instance.
        top_k: Number of chunks to retrieve.

    Returns:
        List of retrieved document chunks.
    """
    print(f"\nSearching for: '{query}'")

    # Perform similarity search
    docs = vectorstore.similarity_search(query, k=top_k)

    # Format context for LLM
    context_text = "\n\n".join([doc.page_content for doc in docs])

    if context_text:
        print(f"Found {len(docs)} relevant chunk(s).")
        print(f"\nRetrieved context:\n{context_text}")
    else:
        print("No relevant context found.")

    return docs


def generate_response(query: str, context: str, system_prompt: str) -> str:
    """Generate a response using an LLM.

    Args:
        query: User's question.
        context: Retrieved context from vector store.
        system_prompt: System prompt for the LLM.

    Returns:
        Generated response text.
    """
    # Check if we have API credentials configured
    import os

    huggingface_key = os.getenv("HUGGING_FACE_API_KEY", "")

    if not huggingface_key:
        print("No Hugging Face API key found. Using local context only.")
        response = f"""I couldn't connect to an external LLM service without an API key. Here's the information I found from your documents:

{context}

To use this chatbot fully, you'll need to set up an API key for either:
1. Hugging Face Inference API (free tier available at https://huggingface.co/settings/tokens)
2. Ollama with a local model (e.g., Mistral-7B-Instruct)

You can configure your API key by setting the HUGGING_FACE_API_KEY environment variable."""
        return response

    # Use LangChain's LLM with the retrieved context
    try:
        from langchain_huggingface import HuggingFaceEndpointLLM

        llm = HuggingFaceEndpointLLM(
            model="mistralai/Mistral-7B-Instruct-foundation",
            huggingfacehub=True,
            max_length=512,
            top_p=0.95,
            typical_p=0.95,
        )

        prompt = f"""You are a helpful assistant that answers questions based on the provided context.
If the context doesn't contain relevant information, state that clearly.

Context:
{context}

Question: {query}

Answer:"""

        response = llm.invoke(prompt)
    except Exception as e:
        print(f"Error using HuggingFaceEndpoint: {e}")
        # Fallback to a simpler approach - just return the context with a note
        response = f"""I couldn't connect to an external LLM service. Here's the information I found from your documents:

{context}

To use this chatbot fully, you'll need to set up an API key for either:
1. Hugging Face Inference API (free tier available at https://huggingface.co/settings/tokens)
2. Ollama with a local model (e.g., Mistral-7B-Instruct)

You can configure your API key by setting the HUGGING_FACE_API_KEY environment variable."""

    return response


def chat(query: str, persist_dir: str, system_prompt: str = "You are a helpful assistant.") -> str:
    """Main chat function.

    Args:
        query: User's question.
        persist_dir: Directory where the vector store is persisted.
        system_prompt: System prompt for the LLM.

    Returns:
        Generated response text.
    """
    # Get vector store
    vectorstore = get_vector_store(persist_dir)

    # Retrieve relevant context
    docs = retrieve_context(query, vectorstore)

    # Generate response
    print("\nGenerating response...")
    response = generate_response(query, "\n\n".join([doc.page_content for doc in docs]), system_prompt)

    return response


if __name__ == "__main__":
    # Default persist directory
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
