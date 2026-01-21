from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    for doc in chunks:
        text = doc.page_content.lower()

        # -------- Financial Performance --------
        if any(k in text for k in [
            "revenue", "top-line", "bottom-line", "profit",
            "loss", "ebitda", "commission", "advertising",
            "user fee", "ticket sales", "margin", "income",
            "expense", "cash flow", "financial performance"
        ]):
            doc.metadata["section"] = "Financial Performance"

        # -------- Corporate Governance --------
        elif any(k in text for k in [
            "board", "director", "governance",
            "committee", "compliance", "audit",
            "corporate governance", "risk management", "independent director"
        ]):
            doc.metadata["section"] = "Corporate Governance"

        # -------- Business & Strategy --------
        elif any(k in text for k in [
            "strategy", "growth", "business overview",
            "operations", "market", "expansion",
            "scale-up", "demand", "supply"
        ]):
            doc.metadata["section"] = "Business & Strategy"

        # -------- Fallback --------
        else:
            doc.metadata["section"] = "General"

        # -------- Ensure page metadata exists --------
        if "page" not in doc.metadata:
            doc.metadata["page"] = "N/A"

    return chunks
