import os
import time

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from src.config import MODEL_NAME, TEMPERATURE

load_dotenv()


# Inisialisasi model
model = ChatOllama(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
)


def ask_llm(question: str) -> str:
    response = model.invoke(question)
    return response.content