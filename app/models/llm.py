import logging
import os
from pydantic import SecretStr
from app.config import LLM_MODEL_ID,LLM_TEMPERATURE,cf_api_token,cf_account_id
from langchain_cloudflare.chat_models import ChatCloudflareWorkersAI

logger = logging.getLogger(__name__)


def get_llm(model:str=LLM_MODEL_ID,temperature:float=LLM_TEMPERATURE)-> ChatCloudflareWorkersAI:
      """
            Returns the initialized LLM instance.
            """
      logger.info("Loading IBM LLM:{model} with temperature={temperature}")
      try:
            llm = ChatCloudflareWorkersAI(
            model=model,#type:ignore 
            temperature=temperature,
            api_token=cf_api_token,#type:ignore
            account_id=cf_account_id,#type:ignore
            )
            return llm
      except Exception:
            logger.exception("Failed to initialize LLM")
            raise



