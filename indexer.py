from langchain_core.documents.base import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing import List, Optional

class DocumentIndexer:
    
    @staticmethod
    def create_new_db(documents: List[Document], embedding_model: str = "text-embedding-3-small", persist_directory: str = "./vector_db") -> Chroma:
        embeddings = OpenAIEmbeddings(model=embedding_model)
        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory if persist_directory is not None else None,
        )
        
        return vector_db
    
    @staticmethod
    def add_documents_to_db(documents: List[Document], vector_db: Optional[Chroma] = None, path: Optional[str] = None) -> Chroma:
        if vector_db is None and path is not None:
            vector_db = DocumentIndexer.load_db(path)
        elif vector_db is None and path is None:
            raise ValueError("Either vector_db or path must be provided")
        
        vector_db.add_documents(documents)
        return vector_db
    
    @staticmethod
    def load_db(path: str, embedding_model: str = "text-embedding-3-small") -> Chroma:
        return Chroma(persist_directory=path, embedding_function=OpenAIEmbeddings(model=embedding_model))
