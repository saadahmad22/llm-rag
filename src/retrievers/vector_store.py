from langchain_core.documents.base import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from src.config.settings import DATA_DIR
from src.utils.document_loader import TextLoader
from src.utils.text_splitter import TextSplitter
from pathlib import Path

class VectorStore:
    '''A class to represent a vector store for retrieving documents'''

    def __init__(self, documents: list[Document]=None, chunk_size: int = 500, chunk_overlap: int = 50):
        '''Initialise the vector store

        Args:
            documents (list[Document]): The list of documents to create the vector store from
                If None, will pull from the default directory (default data/documents)
        '''
        
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")        
        # LlamaCppEmbeddings(
        #     model_path="/Users/saad/Documents/Work/RaWee/poc/python/model/llama-3.2-3b-instruct-q8_0.gguf",
        #     n_ctx=512
        #     )
        self.store = self.create_from_documents(documents)        
        
    def create_from_documents(self, documents: list[Document]=None) -> FAISS:
        '''Create a vector store from a list of documents or he default directory

        Args:
            documents (list[Document]): The list of documents to create the vector store from

        Returns:
            FAISS: The vector store created from the documents
        '''

        if documents is None:
            text_splitter = TextSplitter()
            documents = [doc for file_path in Path(DATA_DIR).rglob("*.txt") for doc in TextLoader(file_path).load()]
            documents = text_splitter.split_documents(documents)
        print(f"Creating vector store from {len(documents)} documents")

        documents = [doc.page_content for doc in documents]
        text_embeddings = self.embeddings.embed_documents(documents)
        # text_embeddings = [emb[0] if isinstance(emb, list) else emb for emb in text_embeddings]
        text_embedding_pairs = zip(documents, text_embeddings)
        return FAISS.from_embeddings(text_embedding_pairs, self.embeddings)
        # return FAISS.from_documents(
        #     documents, 
        #     self.embeddings)
        # return FAISS.from_texts(
        #     documents,
        #     self.embeddings,
        # )
    
    def retrieve(self, query: str, k_documents:int = 10) -> list[Document]:
        '''Retrieve relevant documents from the vector store
        
        Args:
            query (str): The query to retrieve documents for
            k_documents (int): The number of documents to retrieve (default 10)
            
        Returns:
            list[Document]: The list of documents retrieved
        '''

        retriever = self.store.as_retriever()
        return retriever.invoke(query, search_kwargs={"k": k_documents})