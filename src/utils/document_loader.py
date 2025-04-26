from langchain_core.documents.base import Document

from langchain_community.document_loaders import DirectoryLoader, TextLoader

class DocumentLoader:
    def __init__(self, directory_path="./"):
        """Initializes the DocumentLoader class.

        Keyword arguments:
        directory_path -- the path to the directory containing the documents

        Return: None
        """
        self.directory_path = directory_path
        
    def load(self) -> list[Document]:
        """Loads the documents from the directory.

        Return: the documents
        """

        loader = DirectoryLoader(
            self.directory_path,
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        return loader.load()