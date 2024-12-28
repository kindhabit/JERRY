from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import Dict, List

class AnalysisChain:
    @staticmethod
    def create_supplement_chain(llm, parser):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a supplement analysis expert."),
            ("user", "{text}")
        ])
        return prompt | llm | parser

    @staticmethod
    def create_interaction_chain(llm, parser):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an interaction analysis expert."),
            ("user", "{text}")
        ])
        return prompt | llm | parser 