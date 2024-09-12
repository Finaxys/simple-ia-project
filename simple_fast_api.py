from fastapi import FastAPI
from dotenv import load_dotenv
from simple_model_api import SimpleModelAPI
from simple_llamaindex_rag import SimpleLlamaIndexRag

app = FastAPI()
load_dotenv()

@app.post("/query")
def query_model_api(query: str):
    """
    Query the model API with a query
    """
    model_api = SimpleModelAPI("gpt-3.5-turbo")
    response = model_api.call(query)
    return response

@app.post("/query_index")
def query_index_api(query: str):
    """
    Query the index API with a query
    """
    index = SimpleLlamaIndexRag("data")
    response = index.query(query)
    return response

@app.post("/retrieve_index")
def retrieve_index_api(query: str):
    """
    Retrieve the index API with a query
    """
    index = SimpleLlamaIndexRag("data")
    retrieved_docs = index.retrieve(query)
    return retrieved_docs
