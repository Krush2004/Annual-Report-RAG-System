from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    for doc in chunks:
        text = doc.page_content.lower()

        # -------- Quick Commerce & Logistics --------
        if any(k in text for k in [
            "instamart", "dark store", "quick commerce", "skus", 
            "delivery partner", "logistics", "mall", "insanelygood", "handpicked"
        ]):
            doc.metadata["section"] = "Quick Commerce & Logistics"

        # -------- Food Delivery --------
        elif any(k in text for k in [
            "food delivery", "restaurant partner", "dining out", 
            "dineout", "out-of-home", "steppinout", "transacting users"
        ]):
            doc.metadata["section"] = "Food Delivery"

        # -------- Financial Performance --------
        elif any(k in text for k in [
            "revenue", "profit", "loss", "ebitda", "commission", 
            "advertising", "margin", "income", "expense", "financial summary"
        ]):
            doc.metadata["section"] = "Financial Performance"

        # -------- Corporate Governance --------
        elif any(k in text for k in [
            "board", "director", "governance", "committee", 
            "compliance", "audit", "share capital", "kmp"
        ]):
            doc.metadata["section"] = "Corporate Governance"

        # -------- Strategy & Acquisitions --------
        elif any(k in text for k in [
            "acquisition", "acquired", "merger", "integrated", 
            "synergy", "expansion", "piloted", "venture", "partnership"
        ]):
            doc.metadata["section"] = "Strategy & Acquisitions"

        # -------- Fallback --------
        else:
            doc.metadata["section"] = "General Business"

        # -------- Ensure page metadata exists --------
        if "page" not in doc.metadata:
            doc.metadata["page"] = "N/A"

    return chunks
