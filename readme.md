# RAG Application for Document Retrieval and QA

This project provides a comprehensive setup for a Retrieval-Augmented Generation (RAG) application focused on documents related to floods in Rio Grande do Sul, Brazil. The application is built using Python and leverages multiple libraries including `langchain`, `gradio`, and `OpenAI`.

## Table of Contents
- [RAG Application for Document Retrieval and QA](#rag-application-for-document-retrieval-and-qa)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Components](#components)
    - [Chunker](#chunker)
    - [Loader](#loader)
    - [Indexer](#indexer)
    - [Retriever](#retriever)
    - [RAG](#rag)
    - [Gradio UI](#gradio-ui)

## Overview

The application allows users to:
- Load documents from various URLs.
- Split the documents into smaller chunks based on token length.
- Create and manage a vector database using embeddings.
- Perform queries using a retrieval-augmented generation approach.
- Interact with the system via a user-friendly Gradio interface.

## Components

### Chunker
The `Chunker` class splits documents into smaller chunks based on a specified model's token length.

**Methods:**
- `chunk(documents, model_name="gpt-3.5-turbo", chunk_size=500, chunk_overlap=50)`: Splits documents into chunks.
- `_tiktoken_len(text, model_name)`: Returns the token length of the input text for the specified model.

### Loader
The `Loader` class loads documents from URLs and optionally splits them into smaller chunks.

**Methods:**
- `load_documents(urls, save_path=None, chunk=True, chunk_model_name="gpt-3.5-turbo", chunk_size=500, chunk_overlap=50)`: Loads and optionally splits documents.
- `_save_documents(documents, save_path)`: Saves the documents to a specified path.
- `load_from_file(save_path)`: Loads documents from a file.

### Indexer
The `Indexer` class creates and manages a Chroma vector database using documents and embeddings.

**Methods:**
- `create_new_db(documents, embedding_model="text-embedding-3-small", persist_directory="./vector_db")`: Creates a new Chroma vector database.
- `add_documents_to_db(documents, vector_db=None, path=None)`: Adds documents to an existing Chroma vector database.
- `load_db(path, embedding_model="text-embedding-3-small")`: Loads a Chroma vector database from a specified path.

### Retriever
The `Retriever` class creates a retriever from a Chroma vector database using a specified model.

**Methods:**
- `create_retriever_from_db(db, model="gpt-3.5-turbo", top_k=10)`: Creates a `MultiQueryRetriever` from a Chroma vector database.

### RAG
The `RAG` class handles Retrieval-Augmented Generation by combining document retrieval and language model generation.

**Attributes:**
- `completion_model`: The model to use for language generation.
- `embedding_model`: The model to use for generating embeddings.
- `db`: The database indexer for document storage and retrieval.
- `top_k`: The number of top documents to retrieve.
- `retrieved_contexts`: A dictionary storing the contexts retrieved for each query.

**Methods:**
- `query(question)`: Queries the RAG chain with the given question and returns the response.
- `add_documents(urls)`: Adds documents from the specified URLs to the RAG.
- `_add_documents_to_db(documents)`: Adds documents to the database and updates the retriever.
- `_create_retriever(db, top_k=10)`: Creates a retriever from the database.
- `from_db(db_path, completion_model="gpt-3.5-turbo", embedding_model="text-embedding-3-small")`: Creates a RAG instance from an existing database.
- `_create_rag_chain()`: Creates the RAG chain combining document retrieval and language generation.

### Gradio UI
A user-friendly interface built with Gradio that allows users to interact with the RAG system.

**Features:**
- Select completion and embedding models.
- Initialize models.
- Add document links.
- Ask questions and get responses.
- Check the context of the responses.
