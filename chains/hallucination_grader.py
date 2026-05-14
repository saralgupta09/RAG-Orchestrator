from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()


class GradeHallucinations(BaseModel):

    binary_score: str = Field(
        description="Answer grounded in facts: yes or no"
    )


llm = ChatOpenAI(
    model="meta/llama-3.1-70b-instruct",
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0
)

structured_llm = llm.with_structured_output(
    GradeHallucinations
)

prompt = PromptTemplate(
    template="""
You are a grader assessing whether an answer is grounded in
the retrieved documents.

Give a binary score:
yes or no.

Retrieved documents:
{documents}

Generated answer:
{generation}
""",
    input_variables=["documents", "generation"]
)

hallucination_grader = prompt | structured_llm