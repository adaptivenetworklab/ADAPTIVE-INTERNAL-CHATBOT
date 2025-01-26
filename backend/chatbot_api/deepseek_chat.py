from typing import Any, Dict, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from openai import OpenAI

class DeepSeekChat(BaseChatModel):
    api_key: str
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"
    temperature: float = 0.2
    max_tokens: Optional[int] = None

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AIMessage:
        # Initialize the OpenAI client
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        # Format messages for DeepSeek-Chat API
        formatted_messages = [
            {"role": self._get_role(message), "content": message.content}
            for message in messages
        ]

        # Call DeepSeek-Chat API
        response = client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            stream=False,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop,
            **kwargs,
        )

        # Extract the response content
        content = response.choices[0].message.content
        return AIMessage(content=content)

    def _get_role(self, message: BaseMessage) -> str:
        if isinstance(message, HumanMessage):
            return "user"
        elif isinstance(message, AIMessage):
            return "assistant"
        elif isinstance(message, SystemMessage):
            return "system"
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")

    @property
    def _llm_type(self) -> str:
        return "deepseek-chat"
    

class DeepSeekEmbeddings(Embeddings):
    api_key: str
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-embeddings"

    def __init__(self, api_key: str, **kwargs: Any):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Call DeepSeek Embeddings API for documents
        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
        )
        return [embedding.embedding for embedding in response.data]

    def embed_query(self, text: str) -> List[float]:
        # Call DeepSeek Embeddings API for a single query
        response = self.client.embeddings.create(
            input=[text],
            model=self.model,
        )
        return response.data[0].embedding