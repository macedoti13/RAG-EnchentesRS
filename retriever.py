from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from typing import Optional
import pickle

class Retriever:
    
    @staticmethod
    def create_retriever_from_db(db: Chroma, model: str = "gpt-3.5-turbo", top_k: int = 10) -> MultiQueryRetriever:
        
        llm = ChatOpenAI(model=model, temperature=0)
        retriever = MultiQueryRetriever.from_llm(retriever=db.as_retriever(top_k=top_k), llm=llm)
        
        return retriever
