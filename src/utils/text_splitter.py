from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document

class TextSplitter:
    def __init__(self, chunk_size:int = 500, chunk_overlap:int = 50) -> None:
        """Initializes the TextSplitter class.
        
        Keyword arguments:
        chunk_size -- the size of each chunk (default 1000)
        chunk_overlap -- the overlap between chunks (default 100)
        
        Return: None
        """
        
        # RecursiveTextSplitter is a bit more advanced than the CharacterTextSplitter
        # as it can handle nested structures and is more efficient
        # However, this makes it a bit slower than the CharacterTextSplitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Splits the documents into chunks.

        Keyword arguments:
        documents -- the documents to split

        Return: the chunks
        """

        return self.text_splitter.split_documents(documents)