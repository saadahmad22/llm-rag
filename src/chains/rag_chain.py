from langchain_core.documents.base import Document
from langchain_core.chat_history import InMemoryChatMessageHistory

from models.llama import LlamaModel
from src.retrievers.vector_store import VectorStore

class RAGChain:
    '''A class to represent a simple Retrieval-Augmented Generation (RAG) chain.'''

    def __init__(self, vector_store: VectorStore, model: LlamaModel) -> None:
        '''Initializes the RAGChain with a vector store and a language model.
        
        Args:
            vector_store (VectorStore): The vector store to retrieve documents from.
            model (LLM): The language model to generate responses.
        '''

        self.store: VectorStore = vector_store
        self.llm: LlamaModel = model

    def run(self, query: str, history: InMemoryChatMessageHistory) -> str:
        '''Runs the RAG chain to retrieve relevant documents and generate a response.
        
        Args:
            query (str): The query to search for relevant documents.
            history (InMemoryChatMessageHistory): The chat history for the user.
        Returns:
            str: The generated response based on the retrieved documents and the query.
        '''

        # Retrieve relevant documents
        documents: list[Document] = self.store.retrieve(query, k_documents=10)
        
        # Combine documents into a single context        
        context: str = " ".join((doc.page_content for doc in documents))
        
        
        # Generate response using the Llama model
        # This gives the model both the question and the context
        # Conduct prompt engineering to ensure the model understands the context
        # and can generate a relevant response        
        response = self.llm.generate_response(f"Question: {query}\n\nContext: {context}", history)
        '''A better prompt might be:
        response = self.llm.generate_response(
            f"You are an LLM that is an expert in ___use-case___. You have been asked the following question\n"
            f"Make a step-by-step plan to answer the question, and then answer the question based relevant information on the context provided.\n"
            f"Make sure to answer the question in a concise, informative, and user-friendly manner, and to NOT hallucinate information.\n\n"
            f"The question you are to answer is as follows: {query}\n\n"
            f"Some POSSIBLY helpful context is as follows: {context}\n\n"
            "Please provide a concise and informative answer based on the context provided."
        )
        '''
        
        return response
    
    def query_llm(self, query: str, history: InMemoryChatMessageHistory) -> str:
        '''Queries the LLM directly with a question and user history.
        
        Args:
            query (str): The question to ask the LLM.
            history (InMemoryChatMessageHistory): The chat history for the user.
        Returns:
            str: The generated response from the LLM.
        '''

        # A better prompt might be:
        # return self.llm.generate_response(
        #     f"You are an LLM that is an expert in ___use-case___. You have been asked the following question\n"
        #     f"Make a step-by-step plan to answer the question.\n"
        #     f"Make sure to answer the question in a concise, informative, and user-friendly manner, and to NOT hallucinate information.\n\n"
        #     f"The question you are to answer is as follows: {query}\n\n"
        #     "Please provide a concise and informative answer based on the instructions provided."
        # )
        return self.llm.generate_response(f"Question: {query}", history)
    
    def embed(self, query: str) -> list[float]:
        '''Embeds a query using the vector store's embeddings.
        
        Args:
            query (str): The query to embed.
        Returns:
            list[float]: The embedded representation of the query.
        '''

        # this embeds a list of strings, but we only have one query (string),
        # so we can just return the first element
        return self.store.embeddings.embed_documents([query])[0]
    
    def get_user_history(self, user_id: str) -> InMemoryChatMessageHistory:
        '''Retrieves the chat history for a specific user.
        
        Args:
            user_id (str): The ID of the user whose history is to be retrieved.
        Returns:
            InMemoryChatMessageHistory: The chat history for the user.
        '''

        return self.llm.get_user_history(user_id)