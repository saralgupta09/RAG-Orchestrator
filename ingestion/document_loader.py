from langchain_community.document_loaders import PyPDFLoader


def load_pdf_documents(file_path: str):

    try:
        loader = PyPDFLoader(file_path)

        documents = loader.load()

        documents = [
            doc for doc in documents
            if doc.page_content.strip()
        ]

        return documents

    except Exception as e:
        print(f"PDF loading error: {e}")
        return []