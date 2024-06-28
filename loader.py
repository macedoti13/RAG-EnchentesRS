from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
from langchain_core.documents.base import Document
from typing import Union, List
import tiktoken
import pickle
import os

class Chunker:
    """
    A class used for splitting documents into smaller chunks based on a specified model's token length.

    Methods
    -------
    chunk(documents: List[Document], model_name: str = "gpt-3.5-turbo", chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]
        Splits the input documents into smaller chunks.
        
    _tiktoken_len(text: str, model_name: str) -> int
        Returns the token length of the input text for the specified model.
    """

    @staticmethod
    def chunk(documents: List[Document], model_name: str = "gpt-3.5-turbo", chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
        """
        Splits the input documents into smaller chunks.

        Parameters
        ----------
        documents : List[Document]
            A list of Document objects to be split.
        model_name : str, optional
            The name of the model used to determine token length (default is "gpt-3.5-turbo").
        chunk_size : int, optional
            The maximum number of tokens per chunk (default is 500).
        chunk_overlap : int, optional
            The number of tokens to overlap between chunks (default is 50).

        Returns
        -------
        List[Document]
            A list of Document objects split into smaller chunks.
        """
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
        """
        Returns the token length of the input text for the specified model.

        Parameters
        ----------
        text : str
            The text to be tokenized.
        model_name : str
            The name of the model used to determine token length.

        Returns
        -------
        int
            The token length of the input text.
        """
        return len(tiktoken.encoding_for_model(model_name).encode(text))


class Loader:
    """
    A class used for loading documents from URLs and optionally splitting them into smaller chunks.

    Methods
    -------
    load_documents(urls: Union[str, List[str]], save_path: str = None, chunk: bool = True, chunk_model_name: str = "gpt-3.5-turbo", chunk_size: int = 500, chunk_overlap: int = 50, chunk_save_path: str = None) -> List[Document]
        Loads documents from the specified URLs and optionally splits them into smaller chunks.
        
    _save_documents(documents: List[Document], save_path: str)
        Saves the documents to the specified path.
        
    load_from_file(save_path: str) -> List[Document]
        Loads documents from a file at the specified path.
    """

    @staticmethod
    def load_documents(
            urls: Union[str, List[str]], 
            save_path: str = None, 
            chunk: bool = True, 
            chunk_model_name: str = "gpt-3.5-turbo", 
            chunk_size: int = 500, 
            chunk_overlap: int = 50
        ) -> List[Document]:
        """
        Loads documents from the specified URLs and optionally splits them into smaller chunks.

        Parameters
        ----------
        urls : Union[str, List[str]]
            A single URL or a list of URLs to load documents from.
        save_path : str, optional
            The path to save the loaded documents (default is None).
        chunk : bool, optional
            Whether to split the loaded documents into smaller chunks (default is True).
        chunk_model_name : str, optional
            The name of the model used to determine token length for chunking (default is "gpt-3.5-turbo").
        chunk_size : int, optional
            The maximum number of tokens per chunk (default is 500).
        chunk_overlap : int, optional
            The number of tokens to overlap between chunks (default is 50).
        chunk_save_path : str, optional
            The path to save the chunked documents (default is None).

        Returns
        -------
        List[Document]
            A list of loaded and optionally chunked Document objects.
        """
        loader = WebBaseLoader(web_paths=[urls] if isinstance(urls, str) else urls)
        documents = loader.load()
        
        if chunk:
            documents = Chunker.chunk(documents, model_name=chunk_model_name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        if save_path is not None:
            Loader._save_documents(documents, save_path)
        
        return documents

    @staticmethod
    def _save_documents(documents: List[Document], save_path: str):
        """
        Saves the documents to the specified path.

        Parameters
        ----------
        documents : List[Document]
            A list of Document objects to be saved.
        save_path : str
            The path to save the documents.
        """
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
        """
        Loads documents from a file at the specified path.

        Parameters
        ----------
        save_path : str
            The path to load the documents from.

        Returns
        -------
        List[Document]
            A list of Document objects loaded from the file.

        Raises
        ------
        FileNotFoundError
            If no file is found at the specified path.
        """
        if not os.path.exists(save_path):
            raise FileNotFoundError(f"No file found at {save_path}")
        
        with open(save_path, 'rb') as f:
            documents = pickle.load(f)
        
        return documents
