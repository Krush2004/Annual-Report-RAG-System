from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.config import OPENROUTER_API_KEY


def build_rag_components(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

    llm = ChatOpenAI(
        model="google/gemini-2.5-flash-lite",
        temperature=0,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )

    prompt = ChatPromptTemplate.from_template(
        """
        You are a meticulous Financial & Business Analyst. Use the provided context to answer the question with absolute precision.

        CORE DIRECTIVES:
        1. **CONCISENESS**: Provide a direct, 1-2 sentence answer. Do NOT add filler or repeat the question.
        2. **NO CONFLATION**: Verify details belong specifically to the entity asked about. Do NOT mix up similar brands.
        3. **STRICT GROUNDING**: Use only the provided context. If a detail is not explicitly linked, do not include it.
        4. **MISSING INFO**: If details are missing, say exactly: "The annual report does not provide these specific details."

        Context:
        {context}

        Question: {question}

        Answer:
        """
    )

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever
