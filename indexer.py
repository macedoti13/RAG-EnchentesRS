from langchain_core.documents.base import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing import List, Optional

class Indexer:
    """
    A class used for creating and managing a Chroma vector database using documents and embeddings.

    Methods
    -------
    create_new_db(documents: List[Document], embedding_model: str = "text-embedding-3-small", persist_directory: str = "./vector_db") -> Chroma
        Creates a new Chroma vector database from the provided documents.
        
    add_documents_to_db(documents: List[Document], vector_db: Optional[Chroma] = None, path: Optional[str] = None) -> Chroma
        Adds documents to an existing Chroma vector database.
        
    load_db(path: str, embedding_model: str = "text-embedding-3-small") -> Chroma
        Loads a Chroma vector database from the specified path.
    """

    @staticmethod
    def create_new_db(documents: List[Document], embedding_model: str = "text-embedding-3-small", persist_directory: str = "./vector_db") -> Chroma:
        """
        Creates a new Chroma vector database from the provided documents.

        Parameters
        ----------
        documents : List[Document]
            A list of Document objects to be added to the database.
        embedding_model : str, optional
            The name of the embedding model to use (default is "text-embedding-3-small").
        persist_directory : str, optional
            The directory to save the persisted vector database (default is "./vector_db").

        Returns
        -------
        Chroma
            The created Chroma vector database.
        """
        embeddings = OpenAIEmbeddings(model=embedding_model)
        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory if persist_directory is not None else None,
        )
        
        return vector_db
    
    @staticmethod
    def add_documents_to_db(documents: List[Document], vector_db: Optional[Chroma] = None, path: Optional[str] = None) -> Chroma:
        """
        Adds documents to an existing Chroma vector database.

        Parameters
        ----------
        documents : List[Document]
            A list of Document objects to be added.
        vector_db : Optional[Chroma], optional
            An existing Chroma vector database instance (default is None).
        path : Optional[str], optional
            The path to load the Chroma vector database from if vector_db is not provided (default is None).

        Returns
        -------
        Chroma
            The updated Chroma vector database with the new documents added.

        Raises
        ------
        ValueError
            If neither vector_db nor path are provided.
        """
        if vector_db is None and path is not None:
            vector_db = Indexer.load_db(path)
        elif vector_db is None and path is None:
            raise ValueError("Either vector_db or path must be provided")
        
        vector_db.add_documents(documents)
        return vector_db
    
    @staticmethod
    def load_db(path: str, embedding_model: str = "text-embedding-3-small") -> Chroma:
        """
        Loads a Chroma vector database from the specified path.

        Parameters
        ----------
        path : str
            The directory to load the persisted vector database from.
        embedding_model : str, optional
            The name of the embedding model to use (default is "text-embedding-3-small").

        Returns
        -------
        Chroma
            The loaded Chroma vector database.
        """
        return Chroma(persist_directory=path, embedding_function=OpenAIEmbeddings(model=embedding_model))
