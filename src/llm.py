import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from src.config import MODEL_NAME

load_dotenv()


model = ChatOllama(
    model=MODEL_NAME,
    temperature=0,
)


def ask_llm(question: str) -> str:
    response = model.invoke(question)
    return response.content.strip()