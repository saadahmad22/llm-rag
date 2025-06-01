from langchain_core.documents.base import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import LlamaCppEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.document_loader import TextLoader
from src.utils.text_splitter import TextSplitter
from pathlib import Path

DATA_DIR = "./data/documents"

class VectorStore:
    '''A class to represent a vector store for retrieving documents'''

    def __init__(self, documents: list[Document]=None, chunk_size: int = 500, chunk_overlap: int = 50):
        '''Initialise the vector store

        Args:
            documents (list[Document]): The list of documents to create the vector store from
                If None, will pull from the default directory (default data/documents)
        '''
        
        self.embeddings = LlamaCppEmbeddings(model_path="./model/Llama-3.2-3B-Instruct-Q6_K.gguf", n_ctx=20_000, n_batch=512, n_threads=8, verbose=False)
        # self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")        
        self.store = self.create_from_documents(documents)
        
    def create_from_documents(self, raw_documents: list[Document]=None) -> FAISS:
        '''Create a vector store from a list of documents or he default directory

        Args:
            documents (list[Document]): The list of documents to create the vector store from

        Returns:
            FAISS: The vector store created from the documents
        '''

        if raw_documents is None:
            text_splitter = TextSplitter()
            raw_documents: list[Document] = [doc for file_path in Path(DATA_DIR).rglob("*.txt") for doc in TextLoader(file_path).load()]
            # split. Will still be a list of Document objects, but now each Document object will have a smaller chunk of text
            split_documents: list[Document] = text_splitter.split_documents(raw_documents)        

        # extract text from the split Document objects
        documents_strings: list[str] = [doc.page_content for doc in split_documents]
        # create embeddings for the documents
        text_embeddings = self.embeddings.embed_documents(documents_strings) 

        # create pairs of (text, embedding) to pass to the FAISS vector store
        # i.e., (string object, vector)
        # does basically a zipper merge, which pairs each text with its corresponding embedding
        text_embedding_pairs = zip(documents_strings, text_embeddings)
        # print(f"Creating vector store from {len(split_documents)} documents")
        # print(f"Example text embedding pair: {next(text_embedding_pairs)}")

        return FAISS.from_embeddings(text_embedding_pairs, self.embeddings)
    
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
