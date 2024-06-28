from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from typing import Optional

class Retriever:
    """
    A class used for creating a retriever from a Chroma vector database using a specified model.

    Methods
    -------
    create_retriever_from_db(db: Chroma, model: Optional[str] = "gpt-3.5-turbo", top_k: Optional[int] = 10) -> MultiQueryRetriever
        Creates a MultiQueryRetriever from a Chroma vector database.
    """

    @staticmethod
    def create_retriever_from_db(db: Chroma, model: Optional[str] = "gpt-3.5-turbo", top_k: Optional[int] = 10) -> MultiQueryRetriever:
        """
        Creates a MultiQueryRetriever from a Chroma vector database.

        Parameters
        ----------
        db : Chroma
            The Chroma vector database to use as the retriever.
        model : Optional[str], optional
            The name of the model to use for the retriever (default is "gpt-3.5-turbo").
        top_k : Optional[int], optional
            The number of top documents to retrieve (default is 10).

        Returns
        -------
        MultiQueryRetriever
            The created MultiQueryRetriever.
        """
        llm = ChatOpenAI(model=model, temperature=0)
        retriever = MultiQueryRetriever.from_llm(retriever=db.as_retriever(top_k=top_k), llm=llm)
        
        return retriever
