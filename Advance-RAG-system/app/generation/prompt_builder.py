from langchain.prompts import (ChatPromptTemplate)


class PromptBuilder:

    @staticmethod
    def build():

        return ChatPromptTemplate.from_template(
            """
You are an intelligent AI assistant.

Answer the user's question ONLY
using the provided context.

If the answer is not found in
the context, say:

"I could not find the answer
inside the provided documents."

================================
CONTEXT
================================

{context}

================================
QUESTION
================================

{question}

================================
ANSWER
================================
"""
        )