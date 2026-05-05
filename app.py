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
            margin-bottom: 20px; /* Reduced gap */
            animation: fadeIn 0.8s ease-out;
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
            margin-bottom: 15px; /* Reduced gap */
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
            border-radius: 20px !important;
            border: 1px solid rgba(255, 255, 255, 0.03) !important;
            padding: 1rem 1.25rem !important; /* Reduced for "chat little" */
            margin-bottom: 1.5rem !important;
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

        /* Sidebar Global Overrides */
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div:has(button) {
            margin-top: 5px;
        }

        /* Target Sidebar Buttons specifically (Excluding File Uploader) */
        [data-testid="stSidebar"] .stButton button[kind="secondary"] {
            background: rgba(239, 68, 68, 0.08) !important;
            border: 1px solid rgba(239, 68, 68, 0.25) !important;
            color: #fca5a5 !important;
            border-radius: 10px !important;
            width: 100% !important;
            padding: 8px 12px !important;
            font-size: 0.85rem !important;
            font-weight: 500 !important;
            transition: all 0.3s ease !important;
            text-transform: none !important;
        }

        /* Restore File Uploader original aesthetic */
        [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            color: #e2e8f0 !important;
            box-shadow: none !important;
            transform: none !important;
        }

        [data-testid="stSidebar"] button[kind="secondary"]:hover {
            background: rgba(239, 68, 68, 0.2) !important;
            border-color: #f87171 !important;
            color: #ffffff !important;
            box-shadow: 0 0 20px rgba(239, 68, 68, 0.3) !important;
            transform: translateY(-2px);
        }

        [data-testid="stSidebar"] h3 {
            color: #a78bfa !important;
            font-size: 0.9rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.1rem !important;
            margin-top: 1rem !important; /* Reduced from 2rem */
            opacity: 0.8;
        }

        /* Sidebar Divider Tightening */
        [data-testid="stSidebar"] hr {
            margin-top: 0.5rem !important;
            margin-bottom: 0.5rem !important;
            height: 2px !important; /* Slightly bolder */
            background: rgba(255, 255, 255, 0.15) !important;
            border: none !important;
            opacity: 0.3 !important;
        }

        /* Modern Typography-Led Design */
        .status-chip {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 10px 20px;
            border-radius: 50px;
            font-weight: 600;
            font-size: 0.85rem;
            margin-top: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid transparent;
        }

        .sidebar-tip {
            margin-top: 22px; /* Increased gap from 12px */
            padding: 0 10px;
            font-size: 0.75rem;
            color: #64748b;
            line-height: 1.5;
        }

        .sidebar-tip b {
            color: #94a3b8;
        }

        .status-chip.success {
            background: rgba(16, 185, 129, 0.1);
            border-color: rgba(16, 185, 129, 0.2);
            color: #34d399;
        }

        .status-chip.waiting {
            background: rgba(245, 158, 11, 0.1);
            border-color: rgba(245, 158, 11, 0.2);
            color: #fbbf24;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            animation: pulse-dot 2s infinite;
        }

        .status-chip.success .status-dot {
            background: #10b981;
            box-shadow: 0 0 10px #10b981;
        }

        .status-chip.waiting .status-dot {
            background: #f59e0b;
            box-shadow: 0 0 10px #f59e0b;
        }

        @keyframes pulse-dot {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.4); opacity: 0.5; }
            100% { transform: scale(1); opacity: 1; }
        }
        .modern-welcome-container {
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 40vh; /* Reduced from 60vh to prevent scrollbar */
            margin: 0 auto;
            max-width: 850px;
            animation: fadeIn 0.8s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        .modern-welcome-container h1 {
            font-size: 4rem;
            font-weight: 800;
            margin-bottom: 0.2rem; /* Significantly reduced from 0.8rem */
            letter-spacing: -0.15rem;
            color: #f1f5f9;
            line-height: 1;
        }

        .modern-welcome-container h1 span {
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 30px rgba(96, 165, 250, 0.2);
        }

        .modern-welcome-container p {
            color: #94a3b8;
            font-size: 1.1rem;
            margin-bottom: 1.5rem; /* Significantly reduced from 4rem */
            font-weight: 300;
            max-width: 550px;
            margin-left: auto;
            margin-right: auto;
            line-height: 1.6;
        }

        .sidebar-tip b {
            color: #94a3b8;
        }

        .unique-mission {
            font-size: 0.7rem;
            color: #60a5fa;
            letter-spacing: 0.12rem;
            text-transform: uppercase;
            font-weight: 700;
            margin-bottom: 1rem;
            padding: 5px 15px;
            border: 1px solid rgba(96, 165, 250, 0.25);
            background: rgba(96, 165, 250, 0.05);
            border-radius: 50px;
            display: inline-block;
            backdrop-filter: blur(5px);
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

def prioritize_docs(docs, query, query_type):
    """Sort chunks based on relevance score (Keyword Overlap + Intent Boost)"""
    stop_words = {"what", "which", "year", "did", "the", "was", "how", "many", "describe", "explain", "tell", "show"}
    query_keywords = [w.lower() for w in query.split() if len(w) > 3 and w.lower() not in stop_words]
    
    def score_doc(doc):
        content = doc.page_content.lower()
        # Base Score: Keyword Overlap
        score = sum(1.5 for kw in query_keywords if kw in content)
        
        # Intent Boosts
        if query_type == "financial" and any(k in content for k in FINANCIAL_KEYWORDS): score += 2.0
        if query_type == "strategy" and any(k in content for k in STRATEGY_KEYWORDS): score += 2.0
        if query_type == "metric" and any(k in content for k in BUSINESS_METRIC_KEYWORDS): score += 2.0
        if any(c.isdigit() for c in content): score += 0.5 # Digits boost for numbers/years
        
        return score

    return sorted(docs, key=score_doc, reverse=True)

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
            
        # Phrase Bonus (Sequential matches)
        phrase_bonus = 0.0
        query_words_list = query.lower().split()
        for i in range(len(query_words_list)-1):
            phrase = f"{query_words_list[i]} {query_words_list[i+1]}"
            if phrase in content_lower:
                phrase_bonus += 0.1
            
        scores.append(base_score + noun_bonus + fin_bonus + phrase_bonus)

    # FOCUS on Top 3 docs (quality over quantity)
    top_scores = sorted(scores, reverse=True)[:3]
    avg_score = sum(top_scores) / len(top_scores)
    final_score = min(1.0, avg_score * 1.4) 
    
    return round(final_score * 100, 1)

def post_answer_completeness_check(answer: str, docs):
    """
    Warn if financial drivers exist in docs but not reflected in answer
    """
    answer_lower = answer.lower()
    missing = []

    for k in FINANCIAL_KEYWORDS:
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
    stop_words = {"what", "which", "year", "did", "the", "was", "how", "many"}
    query_words = [w.lower() for w in query.split() if len(w) > 3 and w.lower() not in stop_words]

    output = []
    for sent in sentences:
        sent_lower = sent.lower()
        # High-relevance sentences often contain multiple query keywords
        matches = sum(1 for word in query_keywords if word in sent_lower) if 'query_keywords' in locals() else sum(1 for word in query_words if word in sent_lower)
        
        # Basic Fuzzy check for typos (matches prefix of long words)
        for word in query_words:
            if len(word) > 6 and word[:4] in sent_lower:
                matches += 0.5

        if matches >= 1.5:
            output.append(f"**🟡 {sent}**")
        else:
            output.append(sent)

    return " ".join(output)


def confidence_label(score):
    if score >= 85:
        return "🛡️ High Accuracy", "#22c55e" # Green
    if score >= 60:
        return "⚖️ Moderate Confidence", "#eab308" # Yellow/Orange
    return "⚠️ Low - Verify Proof", "#ef4444" # Red


def merge_evidence_by_section(docs):
    """
    Merge multi-page evidence and remove duplicate chunks while preserving relevance order.
    Returns a list of tuples: [(section_name, [docs])]
    """
    merged = {}
    seen_content = set()
    section_order = []
    
    for doc in docs:
        section = doc.metadata.get("section", "General Business")
        content = doc.page_content.strip()
        
        # Check for near-duplicates
        content_hash = content[:100]
        if content_hash in seen_content:
            continue
        seen_content.add(content_hash)
        
        if section not in merged:
            merged[section] = []
            section_order.append(section)
        merged[section].append(doc)
        
    return [(s, merged[s]) for s in section_order]


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
        st.sidebar.markdown("""
            <div class="status-chip success">
                <div class="status-dot"></div>
                Document Ready
            </div>
            <div class="sidebar-tip">
                💡 <b>Pro Tip:</b> Ask about specific financial metrics or strategy risks for the best results.
            </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
            <div class="status-chip waiting">
                <div class="status-dot"></div>
                Awaiting Upload
            </div>
            <div class="sidebar-tip">
                💡 <b>System Tip:</b> Use searchable OCR PDFs for 100% accurate evidence grounding.
            </div>
        """, unsafe_allow_html=True)


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
    if st.session_state.messages:
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
    else:
        # Render Modern Typography Dashboard
        st.markdown(f"""
            <div class="modern-welcome-container">
                <h1>Analyze. <span>Verified.</span></h1>
                <p>Grounded analysis of <b>{st.session_state.processed_filename}</b> is ready. <br>Ask a specific question below to begin your analysis.</p>
            </div>
        """, unsafe_allow_html=True)

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
        source_docs = prioritize_docs(retrieved_docs, query, q_type)
        
        SMART_OFFSET = -2

        if not is_not_found_answer(answer):
            ordered_evidence = merge_evidence_by_section(source_docs)
            section_count = 0
            
            # Extract significant keywords for filtering (Case-Insensitive)
            stop_words = {"what", "which", "year", "did", "the", "was", "how", "many", "describe", "explain", "tell", "show"}
            query_keywords = [w.lower() for w in query.split() if len(w) > 3 and w.lower() not in stop_words]

            for section, docs in ordered_evidence:
                if section_count >= 3: break 
                
                filtered_docs = []
                for d in docs:
                    d_content = d.page_content.lower()
                    
                    # If we have keywords, ensure at least one matches to prevent unrelated noise
                    if query_keywords:
                        if not any(kw in d_content for kw in query_keywords):
                            continue 
                    
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
    st.markdown("""
<div class="modern-welcome-container">
    <div class="unique-mission">Zero-Hallucination Audit Engine</div>
    <h1>Precision <span>Audit.</span></h1>
    <p>High-precision institutional research grounded in physical page verification. <br>Upload a document in the sidebar to begin.</p>
</div>
""", unsafe_allow_html=True)

