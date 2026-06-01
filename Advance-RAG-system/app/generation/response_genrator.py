from app.generation.llm import (LLM)

from app.generation.prompt_builder import (PromptBuilder)


class ResponseGenerator:

    @staticmethod
    def generate_response(
        query,
        retrieved_chunks
    ):

        context = "\n\n".join([

            chunk.page_content

            for chunk in retrieved_chunks

        ])

        prompt = PromptBuilder.build()

        llm = LLM.get_llm()

        chain = prompt | llm

        for chunk in chain.invoke({

            "context": context,

            "question": query

        }):

             yield chunk.content