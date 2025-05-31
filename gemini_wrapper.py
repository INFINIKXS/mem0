# gemini_wrapper.py
"""
A simple wrapper for Google Gemini (via google-generativeai) for text generation.
Install: pip install google-generativeai
"""
import os
import google.generativeai as genai

class GeminiLLM:
    def __init__(self, api_key=None, model="models/gemini-2.5-flash-preview-05-20"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY must be set in environment or passed explicitly.")
        self.model = model
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model)

    def generate(self, prompt, **kwargs):
        # prompt: str
        response = self.client.generate_content(prompt, **kwargs)
        # response.text contains the generated text
        return response.text

    def chat(self, messages, **kwargs):
        # messages: list of dicts with 'role' and 'content'
        # We'll concatenate for a simple chat prompt
        prompt = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages])
        return self.generate(prompt, **kwargs)

# Example usage (uncomment to test):
# if __name__ == "__main__":
#     llm = GeminiLLM()
#     print(llm.generate("Tell me a joke about memory."))
