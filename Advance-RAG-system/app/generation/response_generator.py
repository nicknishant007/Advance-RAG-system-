from app.generation.llm import LLM
from app.generation.prompt_builder import PromptBuilder


class ResponseGenerator:

    @staticmethod
    def stream_response(query, retrieved_chunks):

        print("\n" + "=" * 120)
        print("🔍 RESPONSE GENERATOR")
        print("=" * 120)

        print(f"\nQuestion: {query}")

        print(f"\nRetrieved Chunks: {len(retrieved_chunks)}")

        if not retrieved_chunks:
            print("\n❌ No chunks received!")
        else:
            for i, chunk in enumerate(retrieved_chunks, start=1):

                print("\n" + "-" * 100)
                print(f"CHUNK #{i}")

                print("\nMetadata:")
                print(chunk.metadata)

                print("\nLength:", len(chunk.page_content))

                print("\nContent:")
                print(chunk.page_content)

        context = "\n\n".join(
            chunk.page_content
            for chunk in retrieved_chunks
        )

        print("\n" + "=" * 120)
        print("FINAL CONTEXT")
        print("=" * 120)

        print(context)

        print("\nContext Length:", len(context))

        print("=" * 120 + "\n")

        prompt = PromptBuilder.build()

        llm = LLM.get_llm()

        chain = prompt | llm

        print("🚀 Sending prompt to Gemini...\n")

        first_token = True

        for token in chain.stream(
            {
                "context": context,
                "question": query,
            }
        ):

            if token.content:

                if first_token:
                    print("✅ Gemini started streaming.\n")
                    first_token = False

                print(token.content, end="", flush=True)

                yield token.content

        print("\n\n" + "=" * 120)
        print("✅ STREAM FINISHED")
        print("=" * 120)