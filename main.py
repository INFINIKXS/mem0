# File: C:\mem0ai\mem0\run_memo_gemini.py

import os
from dotenv import load_dotenv
from mem0 import Memory
from gemini_wrapper import GeminiLLM

# --- Configuration ---
# Load environment variables from .env file
# This will load GOOGLE_API_KEY if it's in your .env file
load_dotenv()

# --- Gemini config for both LLM and embeddings ---
GEMINI_MODEL_NAME = "gemini-pro"  # or your preferred Gemini model
GEMINI_EMBED_MODEL = "embedding-001"  # replace with the correct Gemini embedding model if needed

config = {
    "llm": {
        "provider": "google",
        "config": {
            "model": GEMINI_MODEL_NAME,
            "api_key": os.getenv("GOOGLE_API_KEY"),
        },
    },
    "embedder": {
        "provider": "google",
        "config": {
            "model": GEMINI_EMBED_MODEL,
            "api_key": os.getenv("GOOGLE_API_KEY"),
        },
    },
}

# --- Initialization ---
def initialize_memory_system():
    """Initializes and returns the Mem0 system with Gemini for both LLM and embeddings."""
    print("Initializing Mem0 system with Gemini for LLM and embeddings...")
    memory_system = Memory.from_config(config_dict=config)
    print("Mem0 system initialized successfully.")
    return memory_system

# --- Main Application Logic ---
def main():
    """Main function to demonstrate Mem0 with Gemini wrapper."""
    # Initialize Gemini LLM
    gemini_llm = GeminiLLM()
    print("\n--- Gemini LLM: Text Generation Example ---")
    prompt = "Tell me a joke about memory."
    print(f"Prompt: {prompt}")
    print("Gemini Response:", gemini_llm.generate(prompt))

    print("\n--- Gemini LLM: Chat Example ---")
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "And what is it famous for?"}
    ]
    print("Gemini Chat Response:", gemini_llm.chat(messages))

    print("\n--- Adding Memories ---")
    memory_system = initialize_memory_system()
    memory_system.add("My favorite color is blue.", user_id="user123", session_id="session_alpha")
    memory_system.add("I live in a city called Techville.", user_id="user123", session_id="session_alpha")
    memory_system.add("My cat's name is Whiskers and he is 3 years old.", user_id="user456", metadata={"pet_type": "cat"})
    memory_system.add("I enjoy hiking on weekends.", user_id="user123", session_id="session_beta", metadata={"activity_type": "outdoor"})
    memory_system.add("Last weekend, I went hiking to Eagle Peak.", user_id="user123", session_id="session_beta")
    print("Memories added.")

    print("\n--- Searching Memories ---")
    query1 = "What is my favorite color?"
    print(f"\nSearching for (user123, session_alpha): '{query1}'")
    results_color = memory_system.search(query=query1, user_id="user123", session_id="session_alpha")
    if results_color:
        for res in results_color:
            print(f"  - Memory: '{res['memory']}' (Score: {res['score']:.4f}, ID: {res['id']})")
    else:
        print("  No relevant memories found.")

    query2 = "Tell me about my pet."
    print(f"\nSearching for (user456): '{query2}'")
    results_pet = memory_system.search(query=query2, user_id="user456")
    if results_pet:
        for res in results_pet:
            print(f"  - Memory: '{res['memory']}' (Score: {res['score']:.4f}, ID: {res['id']})")
    else:
        print("  No relevant memories found.")
        
    query3 = "What did I do last weekend?"
    print(f"\nSearching for (user123, session_beta): '{query3}'")
    results_activity = memory_system.search(query=query3, user_id="user123", session_id="session_beta")
    if results_activity:
        for res in results_activity:
            print(f"  - Memory: '{res['memory']}' (Score: {res['score']:.4f}, ID: {res['id']})")
    else:
        print("  No relevant memories found.")

    print("\n--- Getting All Memories (for a specific user) ---")
    all_mems_user123 = memory_system.get_all(user_id="user123")
    print(f"All memories for user123 (Total: {len(all_mems_user123)}):")
    for mem in all_mems_user123:
        print(f"  - ID: {mem['id']}, Memory: '{mem['memory']}', Session: {mem.get('session_id', 'N/A')}, Meta: {mem.get('metadata', {})}")

    print("\n--- Getting All Memories (all users) ---")
    all_mems = memory_system.get_all()
    print(f"All memories in the system (Total: {len(all_mems)}):")
    for mem in all_mems:
        print(f"  - ID: {mem['id']}, User: {mem.get('user_id', 'N/A')}, Memory: '{mem['memory']}', Session: {mem.get('session_id', 'N/A')}, Meta: {mem.get('metadata', {})}")

    print("\n--- Demo Finished ---")

if __name__ == "__main__":
    main()