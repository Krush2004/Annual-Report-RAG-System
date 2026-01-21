import streamlit as st
import tempfile
import re
from collections import defaultdict

from src.pdf_loader import load_pdf
from src.chunking import split_documents
from src.embeddings import get_embeddings
from src.pinecone_store import get_vectorstore
from src.rag_chain import build_rag_components


# -------------------- Page Config --------------------

st.set_page_config(page_title="Annual Report RAG", layout="wide")
st.title("📘 Annual Report – AI Q&A")
st.caption("Answers strictly based on the uploaded PDF")


# -------------------- Helper Functions --------------------

FINANCIAL_DRIVER_KEYWORDS = {
    "commission", "advertising", "user fee", "ticket",
    "top-line", "revenue", "income"
}

def detect_query_type(query: str) -> str:
    return "financial" if any(k in query.lower() for k in FINANCIAL_DRIVER_KEYWORDS) else "general"


def is_ambiguous_query(query: str) -> bool:
    ambiguous_starts = ("how is", "tell me about", "what about", "explain", "describe")
    return query.lower().startswith(ambiguous_starts)

def is_not_found_answer(answer: str) -> bool:
    strict_triggers = [
        "does not provide any information",
        "could not find any information",
        "not mentioned in the report",
        "no information available in the report"
    ]
    return any(t in answer.lower() for t in strict_triggers)

def prioritize_financial_drivers(docs):
    """Boost chunks that explicitly mention revenue drivers"""
    prioritized = []
    others = []

    for doc in docs:
        text = doc.page_content.lower()
        if any(k in text for k in FINANCIAL_DRIVER_KEYWORDS):
            prioritized.append(doc)
        else:
            others.append(doc)

    return prioritized + others

def post_answer_completeness_check(answer: str, docs):
    """
    Warn if financial drivers exist in docs but not reflected in answer
    """
    answer_lower = answer.lower()
    missing = []

    for k in FINANCIAL_DRIVER_KEYWORDS:
        if any(k in d.page_content.lower() for d in docs) and k not in answer_lower:
            missing.append(k)

    return missing


def extract_page_citations(docs):
    citations = set()
    for doc in docs:
        page = doc.metadata.get("page")
        section = doc.metadata.get("section", "FY section")
        if page is not None:
            citations.add(f"Page {page} – {section}")
    return ", ".join(sorted(citations))


def highlight_sentences(text, query):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    query_words = set(query.lower().split())

    output = []
    for sent in sentences:
        overlap = sum(word in sent.lower() for word in query_words)
        if overlap >= 2:
            output.append(f"**🟡 {sent}**")
        else:
            output.append(sent)

    return " ".join(output)


def compute_confidence(query, docs):
    if not docs:
        return 0.0

    query_words = set(query.lower().split())
    scores = []

    for doc in docs:
        overlap = sum(w in doc.page_content.lower() for w in query_words)
        scores.append(overlap / max(len(query_words), 1))

    return round(min(1.0, sum(scores) / len(scores)) * 100, 1)


def confidence_label(confidence: float) -> str:
    if confidence >= 70:
        return "High"
    elif confidence >= 40:
        return "Medium"
    return "Low"


def merge_evidence_by_section(docs):
    """
    Merge multi-page evidence and remove duplicate chunks
    """
    merged = {}
    seen_texts = set()

    for doc in docs:
        section = doc.metadata.get("section", "General")
        text_key = doc.page_content.strip()[:300]

        if text_key in seen_texts:
            continue

        seen_texts.add(text_key)

        if section not in merged:
            merged[section] = []

        merged[section].append(doc)

    return merged


# -------------------- PDF Upload --------------------

uploaded_pdf = st.file_uploader("Upload Any Annual Report (PDF)", type=["pdf"])


@st.cache_resource
def initialize_rag(pdf_path):
    docs = load_pdf(pdf_path)
    chunks = split_documents(docs)
    embeddings = get_embeddings()
    vectorstore = get_vectorstore(chunks, embeddings)
    return build_rag_components(vectorstore)


# -------------------- Main App --------------------

if uploaded_pdf:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        pdf_path = tmp.name

    rag_chain, retriever = initialize_rag(pdf_path)

    query = st.text_input("Ask a question about the report")

    if query:
        with st.spinner("Searching the report..."):
            answer = rag_chain.invoke(query)
            retrieved_docs = retriever.invoke(query) or []

        # Ambiguous question handling
        if is_ambiguous_query(query) and not retrieved_docs:
            st.subheader("✅ Answer")
            st.markdown(
                "The question is ambiguous and the annual report does not provide "
                "specific information to clearly address it."
            )
            st.metric("Answer Confidence", "20% (Low)")
            st.stop()

        # Financial-driver prioritization
        source_docs = prioritize_financial_drivers(retrieved_docs)

        hide_evidence = is_not_found_answer(answer)

        # Confidence
        confidence = compute_confidence(query, source_docs)
        if hide_evidence:
            confidence = min(confidence, 25.0)

        label = confidence_label(confidence)

        # Citations
        citations = extract_page_citations(source_docs) if not hide_evidence else "Not explicitly mentioned"

        # Output
        st.subheader("✅ Answer")
        st.markdown(f"{answer}\n\n**Sources:** ({citations})")

        st.metric(
            label="Answer Confidence",
            value=f"{confidence}% ({label})",
            help="Based on overlap between the question and retrieved evidence"
        )

        # Post-answer completeness check
        missing = post_answer_completeness_check(answer, source_docs)
        if missing:
            st.warning(
                "⚠️ The report mentions additional revenue drivers "
                f"not fully reflected in the answer: {', '.join(sorted(set(missing)))}."
            )

        # Evidence display
        if not hide_evidence:
            merged = merge_evidence_by_section(source_docs)

            with st.expander("📄 Supporting Context (Merged Evidence)"):
                for section, docs in merged.items():
                    st.markdown(f"### {section}")
                    for doc in docs[:3]:
                        page = doc.metadata.get("page", "N/A")
                        highlighted = highlight_sentences(doc.page_content, query)
                        st.markdown(f"**Page {page}**")
                        st.markdown(highlighted)
                        st.divider()
