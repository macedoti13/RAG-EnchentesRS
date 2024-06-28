from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain_core.documents.base import Document
from typing import Union, List
import tiktoken
import pickle
import os

class Chunker:
    
    @staticmethod
    def chunk(documents: List[Document], model_name: str = "gpt-3.5-turbo", chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda text: Chunker._tiktoken_len(text, model_name),
            add_start_index=True
        )
        chunks = text_splitter.split_documents(documents)
            
        return chunks
    
    @staticmethod
    def _tiktoken_len(text: str, model_name: str) -> int:
        return len(tiktoken.encoding_for_model(model_name).encode(text))


class Loader:
    
    @staticmethod
    def load_documents(
            urls: Union[str, List[str]], 
            save_path: str = None, 
            chunk: bool = True, 
            chunk_model_name: str = "gpt-3.5-turbo", 
            chunk_size: int = 500, 
            chunk_overlap: int = 50, 
            chunk_save_path: str = None
        ) -> List[Document]:
        
        loader = WebBaseLoader(web_paths=[urls] if isinstance(urls, str) else urls)
        documents = loader.load()
        
        if chunk:
            documents = Chunker.chunk(documents, model_name=chunk_model_name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        if save_path is not None:
            Loader._save_documents(documents, save_path)
        
        return documents

    @staticmethod
    def _save_documents(documents: List[Document], save_path: str):
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                existing_documents = pickle.load(f)
            existing_documents.extend(documents)
        else:
            existing_documents = documents
        
        with open(save_path, 'wb') as f:
            pickle.dump(existing_documents, f)

    @staticmethod
    def load_from_file(save_path: str) -> List[Document]:
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"No file found at {save_path}")
        
        with open(save_path, 'rb') as f:
            documents = pickle.load(f)
        
        return documents

