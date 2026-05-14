from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()


class GradeDocuments(BaseModel):

    binary_score: str = Field(
        description="Relevance score: yes or no"
    )


llm = ChatOpenAI(
    model="meta/llama-3.1-70b-instruct",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0
)

structured_llm = llm.with_structured_output(
    GradeDocuments
)

prompt = PromptTemplate(
    template="""
You are a grader assessing relevance of a retrieved document.

If the document contains information relevant to the user question,
grade it as relevant.

Give a binary score:
yes or no.

Retrieved document:
{document}

User question:
{question}
""",
    input_variables=["document", "question"]
)

document_grader = prompt | structured_llm