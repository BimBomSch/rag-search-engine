import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
base_url = os.environ.get("BASE_URL")
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key and not base_url:
    raise RuntimeError("OPENAI_API_KEY, BASE_URL environment variables not set")


def main():
    client = OpenAI(base_url=base_url, api_key=api_key,)
    model = "google/gemma-3-27b"
    prompt = "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."

    response = client.responses.create(model=model, input=prompt,)
    assert response.usage is not None

    print(f"Prompt tokens: {response.usage.input_tokens}")
    print(f"Response tokens: {response.usage.output_tokens}")
    print(f"Total tokens used: {response.usage.total_tokens}")
    print(response.output_text)



if __name__ == "__main__":
    main()









