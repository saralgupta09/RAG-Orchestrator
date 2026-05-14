import os

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import (
    create_stuff_documents_chain
)

load_dotenv()

llm = ChatOpenAI(
    model="meta/llama-3.1-70b-instruct",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0
)

prompt = PromptTemplate.from_template(
    """
You are an intelligent AI assistant.

Answer the user's question ONLY using the provided context.

If the answer is not available in the context,
say:
"I could not find the answer in the uploaded documents."

Context:
{context}

Question:
{input}

Answer:
"""
)

document_chain = create_stuff_documents_chain(
    llm,
    prompt
)


def generate_answer(retrieved_docs, query):

    response = document_chain.invoke({
        "context": retrieved_docs,
        "input": query
    })

    return response