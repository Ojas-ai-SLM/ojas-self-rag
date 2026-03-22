from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

DOCS_DIR = Path("docs")

# Two separate locations:
# - BAKED_INDEX: inside the image (committed to git), read-only reference
# - VOLUME_INDEX: on the Railway volume, persists across restarts
BAKED_INDEX = Path("faiss_index_baked")
VOLUME_INDEX = Path("faiss_index")


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def build_index(embeddings, save_path: Path):
    """Load all PDFs from docs/ and create a fresh FAISS index."""
    docs = []
    for pdf in DOCS_DIR.glob("**/*.pdf"):
        print(f"  Loading: {pdf.name}")
        try:
            docs.extend(PyPDFLoader(str(pdf)).load())
        except Exception as e:
            print(f"  Error loading {pdf.name}: {e}")

    if not docs:
        raise ValueError("No PDFs found in docs/ folder.")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    ).split_documents(docs)

    print(f"  Created {len(chunks)} chunks from {len(docs)} pages")

    store = FAISS.from_documents(chunks, embeddings)
    save_path.mkdir(exist_ok=True)
    store.save_local(str(save_path))
    print(f"  Index saved to {save_path}/")
    return store


def _load_from(path: Path, embeddings):
    return FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True
    )


def load_or_build_index(embeddings):
    """
    Priority order:
    1. Load from volume (persisted index with any uploaded docs)
    2. Load from baked index (committed to git, inside image)
    3. Build fresh from PDFs
    """
    volume_index_file = VOLUME_INDEX / "index.faiss"
    baked_index_file = BAKED_INDEX / "index.faiss"

    if volume_index_file.exists():
        print(f"  Loading index from volume ({VOLUME_INDEX}/)")
        return _load_from(VOLUME_INDEX, embeddings)

    if baked_index_file.exists():
        print(f"  Loading baked index from image ({BAKED_INDEX}/)")
        store = _load_from(BAKED_INDEX, embeddings)
        # Copy to volume so future uploads persist correctly
        print(f"  Copying baked index to volume for persistence...")
        store.save_local(str(VOLUME_INDEX))
        return store

    print("  No index found — building from docs/")
    return build_index(embeddings, VOLUME_INDEX)


def add_documents_to_index(pdf_path: str, store: FAISS, embeddings):
    """Add a new PDF to the existing FAISS index and persist to volume."""
    docs = PyPDFLoader(pdf_path).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    ).split_documents(docs)
    store.add_documents(chunks)
    store.save_local(str(VOLUME_INDEX))
    return len(chunks)