from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

class SimpleLlamaIndexRag:
    """
    Simple LlamaIndex RAG with an index as a VectorStore
    """
    def __init__(self, directory) -> None:
        self.directory = directory
        self.index = None

    def indexation(self):
        """
        Index the documents in the directory
        """
        documents = SimpleDirectoryReader(self.directory).load_data()
        self.index = VectorStoreIndex.from_documents(documents)

    def query(self, query, streaming=False):
        """
        Query the index with a query
        """
        if not self.index:
            self.indexation()

        return self.index.as_query_engine(streaming=streaming).query(query)

    def retrieve(self, query):
        """
        Simple retrieval of the documents from an index
        """
        if not self.index:
            self.indexation()

        return self.index.as_retriever().retrieve(query)

if __name__ == "__main__":
    load_dotenv()
    index = SimpleLlamaIndexRag("data")
    index.indexation()
    print(index.query("What is on the menu?"))
    print(index.query("Can I have a Margherita Pizza?"))
    print(index.query("Can I have a Vegetarian Pizza?, What is the price ?"))
    print(index.query("Given that a Vegetarian Pizza is on the menu, can I have a Vegetarian Pizza?"))

    retrieved_docs = index.retrieve("Can I have a Vegetarian Pizza?, What is the price ?")
    for doc in retrieved_docs:
        print(doc)
