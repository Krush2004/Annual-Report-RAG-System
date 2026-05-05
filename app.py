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

# -------------------- Page Config & Premium Styling --------------------

st.set_page_config(page_title="Annual Report RAG", layout="wide", initial_sidebar_state="expanded")

def inject_premium_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        /* Global Base */
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
            background-color: #0b0f19;
            color: #e2e8f0;
        }

        /* Sidebar: Pro Glassmorphism */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111827 0%, #0b0f19 100%) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 10px 0 40px rgba(0, 0, 0, 0.4);
        }
        
        /* Hide Streamlit default elements but keep sidebar toggle */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Header Layout */
        .header-container {
            margin-top: -60px;
            margin-bottom: 40px;
            animation: fadeIn 1s ease-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .premium-header {
            font-size: 3.5rem;
            font-weight: 700;
            letter-spacing: -0.05rem;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
        }

        .header-text {
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .header-logo {
            font-size: 3rem;
            margin-right: 20px;
            animation: pulse 2.5s infinite alternate;
            filter: drop-shadow(0 0 10px rgba(96, 165, 250, 0.5));
            color: #60a5fa; /* Ensure it has a base color */
        }

        @keyframes pulse {
            0% { transform: scale(1); filter: drop-shadow(0 0 5px #60a5fa); }
            100% { transform: scale(1.1); filter: drop-shadow(0 0 15px #a78bfa); }
        }
        
        .premium-caption {
            color: #94a3b8;
            font-size: 1.1rem;
            margin-bottom: 15px;
            font-weight: 300;
        }

        .header-separator {
            height: 1px;
            background: linear-gradient(90deg, rgba(96, 165, 250, 0), rgba(96, 165, 250, 0.5), rgba(167, 139, 250, 0.5), rgba(167, 139, 250, 0));
            margin-bottom: 30px;
        }

        /* Metric Cards */
        div.stMetric {
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 24px !important;
            border-radius: 20px;
            backdrop-filter: blur(12px);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
        }
        div.stMetric:hover {
            transform: translateY(-5px);
            border-color: rgba(96, 165, 250, 0.3);
        }
        [data-testid="stMetricValue"] {
            font-size: 2.5rem !important;
            color: #60a5fa !important;
        }

        /* Chat Layout Improvements */
        .stChatMessage {
            background: rgba(15, 23, 42, 0.4) !important;
            border-radius: 24px !important;
            border: 1px solid rgba(255, 255, 255, 0.03) !important;
            padding: 1.5rem !important;
            margin-bottom: 2rem !important;
        }

        /* Input Experience */
        .stChatInputContainer {
            border-radius: 30px !important;
            background: #1e293b !important;
            border: 1px solid #334155 !important;
            padding: 0.8rem 1.2rem !important;
        }
        
        /* Citations Bullet Points */
        .citation-list {
            margin-top: 10px;
            padding-left: 5px;
            font-size: 0.9rem;
            color: #94a3b8;
        }
        .citation-item {
            list-style: none;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
        }
        .citation-item::before {
            content: "•";
            color: #60a5fa;
            font-size: 1.5rem;
            margin-right: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

inject_premium_css()

st.markdown('''
    <div class="header-container">
        <h1 class="premium-header">
            <span class="header-logo">🛡️</span>
            <span class="header-text">Annual Report AI</span>
        </h1>
        <p class="premium-caption">Advanced RAG Analysis strictly grounded in your documents.</p>
        <div class="header-separator"></div>
    </div>
''', unsafe_allow_html=True)


# -------------------- Helper Functions --------------------

FINANCIAL_KEYWORDS = {
    "revenue", "profit", "loss", "ebitda", "margin", "income", "expense", "summary"
}
BUSINESS_METRIC_KEYWORDS = {
    "how many", "count", "number of", "stores", "cities", "partners", "skus"
}
LEADERSHIP_KEYWORDS = {
    "chairman", "ceo", "cfo", "management", "director", "founder", "head of"
}
STRATEGY_KEYWORDS = {
    "acquisition", "acquired", "merger", "integrated", "synergy", "impact of", "pilot"
}

def detect_query_type(query: str) -> str:
    query_lower = query.lower()
    if any(k in query_lower for k in FINANCIAL_KEYWORDS):
        return "financial"
    if any(k in query_lower for k in BUSINESS_METRIC_KEYWORDS):
        return "metric"
    if any(k in query_lower for k in LEADERSHIP_KEYWORDS):
        return "leadership"
    if any(k in query_lower for k in STRATEGY_KEYWORDS):
        return "strategy"
    return "general"


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

def prioritize_docs(docs, query_type):
    """Sort chunks based on query intent"""
    if query_type == "financial":
        return sorted(docs, key=lambda d: any(k in d.page_content.lower() for k in FINANCIAL_KEYWORDS), reverse=True)
    if query_type == "metric":
        # Boost chunks with digits and metric keywords
        return sorted(docs, key=lambda d: (any(c.isdigit() for c in d.page_content), any(k in d.page_content.lower() for k in BUSINESS_METRIC_KEYWORDS)), reverse=True)
    if query_type == "leadership":
        # Boost chunks that mention leadership roles
        return sorted(docs, key=lambda d: any(k in d.page_content.lower() for k in LEADERSHIP_KEYWORDS), reverse=True)
    if query_type == "strategy":
        # Boost chunks that mention acquisitions and impact
        return sorted(docs, key=lambda d: any(k in d.page_content.lower() for k in STRATEGY_KEYWORDS), reverse=True)
    return docs

def compute_confidence(query, docs):
    if not docs:
        return 0.0
    
    query_words = set(re.findall(r'\w+', query.lower()))
    # Remove common stop words to focus on real facts
    stop_words = {"what", "was", "the", "of", "in", "and", "to", "for", "is", "on", "that", "how", "impact"}
    query_facts = {w for w in query_words if w not in stop_words and len(w) > 2}
    
    if not query_facts:
        query_facts = query_words

    scores = []
    for doc in docs:
        content_lower = doc.page_content.lower()
        
        # Word Overlap Score
        overlap = sum(1 for w in query_facts if w in content_lower)
        base_score = overlap / len(query_facts)
        
        # Proper Noun Bonus (Case sensitive check)
        # Proper nouns in query are usually capitalized (e.g., 'Dineout')
        proper_nouns = [w for w in query.split() if w[0].isupper() and w.lower() not in stop_words]
        noun_bonus = 0.0
        if proper_nouns:
            found_nouns = sum(1 for n in proper_nouns if n.lower() in content_lower)
            noun_bonus = (found_nouns / len(proper_nouns)) * 0.3
            
        # Financial Context Bonus
        fin_bonus = 0.0
        if any(k in content_lower for k in FINANCIAL_KEYWORDS):
            fin_bonus = 0.1
            
        scores.append(base_score + noun_bonus + fin_bonus)

    # Use a more generous average but clamp at 100
    avg_score = sum(scores) / len(scores)
    final_score = min(1.0, avg_score * 1.8) # Boost scaling for better representation
    
    return round(final_score * 100, 1)

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


def extract_page_citations(docs, offset=0):
    if not docs:
        return "None"
        
    relevant_docs = docs[:5]
    unique_citations = {}

    for doc in relevant_docs:
        page_val = doc.metadata.get("page")
        section = doc.metadata.get("section", "General")
        
        if page_val is not None:
            try:
                # Apply the user-defined offset
                page = int(float(page_val)) + 1 + offset
                if page not in unique_citations:
                    unique_citations[page] = section
            except (ValueError, TypeError):
                if page_val not in unique_citations:
                    unique_citations[page_val] = section

    sorted_pages = sorted(unique_citations.keys())
    
    html_list = '<ul class="citation-list">'
    for p in sorted_pages:
        html_list += f'<li class="citation-item">Page {p} ({unique_citations[p]})</li>'
    html_list += '</ul>'
    return html_list


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


def confidence_label(score):
    if score >= 85:
        return "🛡️ High Accuracy", "#22c55e" # Green
    if score >= 60:
        return "⚖️ Moderate Confidence", "#eab308" # Yellow/Orange
    return "⚠️ Low - Verify Proof", "#ef4444" # Red


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


# -------------------- Sidebar --------------------

with st.sidebar:
    st.header("📄 Configuration")
    uploaded_pdf = st.file_uploader("Upload Annual Report (PDF)", type=["pdf"])
    
    st.divider()
    st.subheader("⚙️ Control Panel")
    
    # Actions
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    
    # System Status
    if uploaded_pdf:
        st.success("✅ Document Loaded")
    else:
        st.info("💡 Waiting for PDF upload...")


def initialize_rag(pdf_path):
    docs = load_pdf(pdf_path)
    chunks = split_documents(docs)
    embeddings = get_embeddings()
    vectorstore = get_vectorstore(chunks, embeddings)
    return build_rag_components(vectorstore)


# -------------------- Main App --------------------

if uploaded_pdf:
    # Use session state to avoid re-processing on every query
    if "processed_filename" not in st.session_state or st.session_state.processed_filename != uploaded_pdf.name:
        with st.status("📁 Processing Document...", expanded=True) as status:
            st.write("Extracting text from PDF...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.read())
                pdf_path = tmp.name

            st.write("Chunking and generating embeddings...")
            rag_chain, retriever = initialize_rag(pdf_path)
            
            st.write("Syncing with Pinecone Vector Store...")
            st.session_state.rag_chain = rag_chain
            st.session_state.retriever = retriever
            st.session_state.processed_filename = uploaded_pdf.name
            
            status.update(label="✅ Document Ready!", state="complete", expanded=False)

    rag_chain = st.session_state.rag_chain
    retriever = st.session_state.retriever

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show Supporting Evidence if it exists in history
            if "evidence" in message and message["evidence"]:
                with st.expander("📄 Supporting Evidence (Proof)"):
                    st.markdown(message["evidence"], unsafe_allow_html=True)
            
            if "stats" in message:
                with st.expander("📊 Analysis Details"):
                    st.markdown(message["stats"], unsafe_allow_html=True)

    # Use chat input for better UX
    query = st.chat_input("Ask a question about the report...")

    if query:
        # Display user query
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        with st.status("🔍 Analyzing Report...", expanded=True) as status:
            st.write("Fetching relevant document sections...")
            retrieved_docs = retriever.invoke(query) or []
            
            st.write("Synthesizing evidence-based answer...")
            answer = rag_chain.invoke(query)
            
            status.update(label="✅ Analysis Complete", state="complete", expanded=False)

        # Generate Evidence HTML for persistence (with HYPER-STRICT filtering)
        evidence_html = ""
        q_type = detect_query_type(query)
        source_docs = prioritize_docs(retrieved_docs, q_type)
        
        SMART_OFFSET = -2

        if not is_not_found_answer(answer):
            merged = merge_evidence_by_section(source_docs)
            section_count = 0
            
            # Identify Proper Nouns in query (e.g., 'Dineout')
            stop_words = {"what", "was", "the", "of", "in", "and", "to", "for", "is", "on", "that", "how", "impact"}
            proper_nouns_query = [w.lower() for w in query.split() if w[0].isupper() and w.lower() not in stop_words]

            for section, docs in merged.items():
                if section_count >= 3: break 
                
                filtered_docs = []
                for d in docs:
                    d_content = d.page_content.lower()
                    
                    # HYPER-STRICT: If we have proper nouns in query, they MUST be in the doc
                    if proper_nouns_query:
                        if not any(pn in d_content for pn in proper_nouns_query):
                            continue # Skip noise that doesn't mention the core subject
                    
                    filtered_docs.append(d)
                
                if not filtered_docs:
                    continue
                    
                evidence_html += f"<div style='border-left: 3px solid #60a5fa; padding-left: 15px; margin-bottom: 20px;'>"
                evidence_html += f"<b style='color: #60a5fa; font-size: 1.1rem;'>Section: {section}</b><br>"
                
                for doc in filtered_docs[:2]:
                    page_val = doc.metadata.get("page", "N/A")
                    page = int(float(page_val)) + 1 + SMART_OFFSET if str(page_val).replace('.','',1).isdigit() else page_val
                    evidence_html += f"<i style='color: #94a3b8;'>Printed Page {page}</i><br>"
                    content = doc.page_content.replace("\n", " ").strip()
                    highlighted = highlight_sentences(content, query)
                    evidence_html += f"<div style='background: rgba(30, 41, 59, 0.3); padding: 12px; border-radius: 8px; margin: 8px 0; font-size: 0.95rem; line-height: 1.6;'>{highlighted}</div>"
                
                evidence_html += "</div>"
                section_count += 1

        # Confidence & Citations (Using Smart Offset)
        use_source_docs = source_docs if not is_not_found_answer(answer) else []
        confidence = compute_confidence(query, use_source_docs)
        if is_not_found_answer(answer): confidence = min(confidence, 25.0)
        citations_html = extract_page_citations(source_docs, offset=SMART_OFFSET) if source_docs else "None"

        # Display AI response
        with st.chat_message("assistant"):
            st.markdown(answer)
            if evidence_html:
                with st.expander("📄 Supporting Evidence (Proof)", expanded=True):
                    st.markdown(evidence_html, unsafe_allow_html=True)
            
            with st.expander("📊 Analysis Details"):
                col1, col2 = st.columns(2)
                label, color = confidence_label(confidence)
                with col1:
                    st.markdown(f"<h3 style='color: {color}; margin-bottom: 0;'>{label}</h3>", unsafe_allow_html=True)
                    st.metric("Score", f"{confidence}%")
                with col2:
                    st.markdown("**Verified Pages:**")
                    st.markdown(citations_html, unsafe_allow_html=True)

        # Save assistant response to history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "evidence": evidence_html,
            "stats": f"<div style='color: {color}; font-size: 1.2rem; font-weight: bold;'>{label}</div><b>Score:</b> {confidence}%<br><br><b>Verified Pages:</b>{citations_html}"
        })
        st.rerun()

else:
    st.info("Please upload an Annual Report PDF in the sidebar to begin.")

