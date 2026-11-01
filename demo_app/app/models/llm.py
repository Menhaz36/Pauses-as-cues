import logging
from app.config import LLM_MODEL_ID,LLM_TEMPERATURE,groq_api_key
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)


def get_llm(model:str=LLM_MODEL_ID,temperature:float=LLM_TEMPERATURE)-> ChatGroq:
      """
            Returns the initialized LLM instance (gpt-oss-120b through the Groq API).
            """
      logger.info(f"Loading LLM: {model} with temperature={temperature}")
      if not groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set: add it to .env (see .env.example)")
      try:
            llm = ChatGroq(
            model=model,
            temperature=temperature,
            api_key=groq_api_key,#type:ignore
            )
            return llm
      except Exception:
            logger.exception("Failed to initialize LLM")
            raise
