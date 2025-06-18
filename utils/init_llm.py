from langchain_ollama import ChatOllama
from ollama import chat
from pydantic import BaseModel

def get_llm(model: str="qwen2.5:3b", temperature: float=0.7) -> ChatOllama:
    return ChatOllama(
        model=model,
        temperature=temperature,
        num_ctx=100000
    )

def parse_to_json(text: str, response_schema: BaseModel) -> dict:
    response = chat(
    messages=[
        {
            "role": "system",
            "content": f"You are a helpful assistant that understands and translates text to JSON format according to the following schema. {response_schema.model_json_schema()}"
        },
        {
        'role': 'user',
        'content': text,
        }
    ],
    model='Osmosis/Osmosis-Structure-0.6B',
    format=response_schema.model_json_schema(),
    )

    answer = response_schema.model_validate_json(response.message.content)
    return answer