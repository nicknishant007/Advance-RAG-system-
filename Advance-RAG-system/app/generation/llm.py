from dotenv import load_dotenv
load_dotenv()
import os
from langchain.chat_models import init_chat_model

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "Advance RAG System"

class LLM:

    _llm = None

    @classmethod
    def load_model(cls):

        if cls._llm is None:

            cls._llm = init_chat_model(
                model="mistaral-large-latest",
                model_provider="mistaralai",
                temperature=0.4,
                max_tokens=2048,
                max_retries=3,
                timeout=30,
                streaming=True
            )

    @classmethod
    def get_llm(cls):

        cls.load_model()

        return cls._llm