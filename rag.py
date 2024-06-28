from langchain_core.documents.base import Document
from retriever import Retriever
from indexer import Indexer
from loader import Loader
from langchain_chroma import Chroma
from typing import List, Union, Optional, Dict
from langchain.prompts.chat import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

class RAG:
    
    
    def __init__(
            self, 
            completion_model: Optional[str] = "gpt-3.5-turbo", 
            embedding_model: Optional[str] = "text-embedding-3-small", 
            db: Optional[Indexer] = None, 
            top_k: Optional[int] = 10
        ) -> None:
        
        self.completion_model: str = completion_model
        self.embedding_model: str = embedding_model
        self.db: Indexer = db
        self.top_k = top_k
        self.retrieved_contexts: Dict[str, List[Document]] = {}
        
    
    def query(self, question: str) -> str:
            
        if self.rag_chain is None:
            raise ValueError("No documents have been added to the RAG. Please add documents before querying.")
        
        response = self.rag_chain.invoke({"input": question})
        self.retrieved_contexts[question] = response["context"]
        return response["answer"]
        
        
    def add_documents(self, urls: Union[str, List[str]]) -> None:
        
        # Load documents from the web
        documents = Loader.load_documents(urls, chunk_model_name=self.completion_model)
        
        # Create a new db or add documents to an existing db
        self._add_documents_to_db(documents)
        
        
    def _add_documents_to_db(self, documents: List[Document]) -> None:
        
        if self.db is not None:
            # Add documents to existing db
            Indexer.add_documents_to_db(documents, self.db)
        else:
            # Create a new db if one does not exist
            self.db = Indexer.create_new_db(documents, embedding_model=self.embedding_model, persist_directory="./db")
        
        # Create (or recreate) the retriever with the new db 
        self._create_retriever(self.db, top_k=self.top_k)
        self._create_rag_chain()
        
            
    def _create_retriever(self, db: Indexer, top_k: int = 10) -> Retriever:
        self.retriever = Retriever.create_retriever_from_db(db, model=self.completion_model, top_k=top_k)
    
    
    @staticmethod
    def from_db(db_path: str, completion_model: Optional[str] = "gpt-3.5-turbo", embedding_model: Optional[str] = "text-embedding-3-small") -> "RAG":
        db = Indexer.load_db(db_path, embedding_model=embedding_model)
        return RAG(db=db, completion_model=completion_model, embedding_model=embedding_model)
    
    
    def _create_rag_chain(self) -> None:
        system_prompt: str = """
        Você é um assistente de perguntas e respostas inserido em uma RAG (Retrieval-Augmented Generation). \
        Você receberá um contexto extraído de documentos e uma pergunta feita pelo usuário. Sua tarefa é \
        responder à pergunta usando exclusivamente o contexto fornecido. Se o contexto não for fornecido ou \
        não for suficiente para responder à pergunta, responda com "Eu não sei". Utilize apenas as informações \
        contidas no contexto fornecido.

        Contexto fornecido: {context}
        """

        prompt = ChatPromptTemplate.from_messages([("system", system_prompt),("human", "{input}")])
        question_answer_chain = create_stuff_documents_chain(ChatOpenAI(model=self.completion_model), prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)
