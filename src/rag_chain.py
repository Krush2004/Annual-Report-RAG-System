from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.config import OPENROUTER_API_KEY


def build_rag_components(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model="mistralai/mistral-7b-instruct:free",
        temperature=0,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )

    prompt = ChatPromptTemplate.from_template(
        """
        You are an AI assistant answering questions STRICTLY using the provided context 
        from the Annual Report.

        RULES (MANDATORY):
        1. Use ONLY the information explicitly stated in the context.
        2. Do NOT infer, estimate, or calculate values that are not present.
        3. If a question asks for a comparison (e.g., year-over-year, increase/decrease, 
        growth rate) AND the context does NOT provide explicit numerical comparison, 
        respond with:
        "The annual report does not provide a specific comparative figure for this question."
        4. You may mention qualitative statements (e.g., “increase”, “improvement”) ONLY 
        if they are explicitly stated in the context.
        5. If the answer is completely missing, respond exactly with:
        "I could not find this information in the Annual Report."

        FORMAT YOUR ANSWER AS:
        - A clear, concise paragraph
        - No bullet points unless the context itself is bulleted
        - No external knowledge

        Context:
        {context}

        Question:
        {question}

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
