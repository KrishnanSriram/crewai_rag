import os
import json
import psycopg2
from psycopg2.extras import execute_values
from pathlib import Path
from typing import List, Dict, Any

# Document processing
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import PGVector


CONNECTION_STRING = "postgresql://krishnansriram:1234postgres$@localhost:5432/homework"
COLLECTION_NAME = "documents"

# Configuration - simple dictionary
DB_CONFIG = {
    'host': 'localhost',
    'port': '5432',
    'database': 'homework',
    'user': 'krishnansriram',
    'password': '1234postgres$'
}

OLLAMA_CONFIG = {
    'base_url': 'http://localhost:11434',
    'model': 'nomic-embed-text'
}

# Text splitting settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def setup_database():
    """Setup PostgreSQL database with pgvector extension and documents table."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        # Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Create documents table
        cur.execute("""
                    CREATE TABLE IF NOT EXISTS documents
                    (
                        id
                        SERIAL
                        PRIMARY
                        KEY,
                        doc_id
                        TEXT,
                        chunk_index
                        INT,
                        content
                        TEXT,
                        metadata
                        JSONB,
                        embedding
                        VECTOR
                    (
                        768
                    )
                        );
                    """)

        # Basic index for doc_id
        cur.execute("CREATE INDEX IF NOT EXISTS idx_documents_doc_id ON documents (doc_id);")

        conn.commit()
        cur.close()
        conn.close()
        print("✅ Database setup complete")

    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        raise


def load_pdf(pdf_path: str) -> List[Dict]:
    """Load and split a single PDF into chunks."""
    try:
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(pages)

        # Convert to simple dictionaries
        doc_chunks = []
        doc_id = Path(pdf_path).stem

        for i, chunk in enumerate(chunks):
            doc_chunks.append({
                'doc_id': doc_id,
                'chunk_index': i,
                'content': chunk.page_content,
                'metadata': {
                    'source': pdf_path,
                    'page': chunk.metadata.get('page', 0),
                    'doc_id': doc_id,
                    'chunk_index': i
                }
            })

        print(f"✅ Loaded {pdf_path}: {len(doc_chunks)} chunks")
        return doc_chunks

    except Exception as e:
        print(f"❌ Failed to load {pdf_path}: {e}")
        return []


def load_pdfs(pdf_paths: List[str]) -> List[Dict]:
    """Load multiple PDFs and return all chunks."""
    all_chunks = []

    for pdf_path in pdf_paths:
        if os.path.exists(pdf_path):
            chunks = load_pdf(pdf_path)
            all_chunks.extend(chunks)
        else:
            print(f"⚠️  File not found: {pdf_path}")

    print(f"📄 Total chunks loaded: {len(all_chunks)}")
    return all_chunks


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using Ollama."""
    print(f"Generating embeddings for {len(texts)} documents")
    try:
        embeddings_model = OllamaEmbeddings(
            model=OLLAMA_CONFIG['model'],
            base_url=OLLAMA_CONFIG['base_url']
        )

        print(f"🔄 Generating embeddings for {len(texts)} texts...")
        embeddings = embeddings_model.embed_documents(texts)
        print("✅ Embeddings generated")
        return embeddings

    except Exception as e:
        print(f"❌ Failed to generate embeddings: {e}")
        raise


def insert_chunks_to_db(chunks: List[Dict]):
    """Insert document chunks with embeddings into PostgreSQL."""
    if not chunks:
        print("⚠️  No chunks to insert")
        return

    try:
        print(f"We need to insert chunks - {len(chunks)} into PostgreSQL")
        # Get embeddings for all chunks
        texts = [chunk['content'] for chunk in chunks]
        embeddings = get_embeddings(texts)

        # Prepare data for insertion
        insert_data = []
        for chunk, embedding in zip(chunks, embeddings):
            insert_data.append((
                chunk['doc_id'],
                chunk['chunk_index'],
                chunk['content'],
                json.dumps(chunk['metadata']),
                embedding
            ))

        # Insert into database
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        insert_sql = """
                     INSERT INTO documents (doc_id, chunk_index, content, metadata, embedding)
                     VALUES %s \
                     """

        execute_values(cur, insert_sql, insert_data, page_size=100)
        conn.commit()
        cur.close()
        conn.close()

        print(f"✅ Inserted {len(insert_data)} chunks into database")

    except Exception as e:
        print(f"❌ Failed to insert chunks: {e}")
        raise


def build_connection_string() -> str:
    """Build PostgreSQL connection string."""
    return (
        f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )


def search_similar_chunks(query: str, k: int = 5) -> List[Dict]:
    """Search for similar chunks using direct database query."""
    try:
        # Get query embedding
        embeddings_model = OllamaEmbeddings(
            model=OLLAMA_CONFIG['model'],
            base_url=OLLAMA_CONFIG['base_url']
        )

        query_embedding = embeddings_model.embed_query(query)

        # Search in database
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        search_sql = """
                     SELECT doc_id, \
                            chunk_index, \
                            content, \
                            metadata,
                            1 - (embedding <=> %s::vector) as similarity
                     FROM documents
                     ORDER BY embedding <=> %s::vector
                         LIMIT %s \
                     """

        cur.execute(search_sql, (query_embedding, query_embedding, k))
        results = cur.fetchall()

        cur.close()
        conn.close()

        # Format results
        search_results = []
        for row in results:
            search_results.append({
                'doc_id': row[0],
                'chunk_index': row[1],
                'content': row[2],
                'metadata': row[3],
                'similarity': float(row[4])
            })

        return search_results

    except Exception as e:
        print(f"❌ Search failed: {e}")
        return []


# Main pipeline function
def process_pdfs_to_pgvector(pdf_paths: List[str], use_langchain_method: bool = False, setup_database: bool = False):
    """Main function to process PDFs and store in PGVector."""
    print("🚀 Starting PDF to PGVector pipeline...")

    # Step 1: Setup database
    if setup_database == True:
        setup_database()

    # Step 2: Load and chunk PDFs
    chunks = load_pdfs(pdf_paths)

    if not chunks:
        print("❌ No chunks to process!")
        return

    # Step 3: Store in database
    insert_chunks_to_db(chunks)

    print("✅ Pipeline completed!")
    return chunks


# Simple test/demo functions
def demo_search(query: str = "machine learning"):
    """Demo search functionality."""
    print(f"\n🔍 Searching for: '{query}'")
    results = search_similar_chunks(query, k=3)
    print(f"Length of results: {len(results)}")
    print(results)

    for i, result in enumerate(results, 1):
        print(f"\n📄 Result {i} (Similarity: {result['similarity']:.3f})")
        print(f"   Doc: {result['doc_id']}")
        print(f"   Content: {result['content'][:100]}...")


def list_documents_in_db():
    """List all documents in the database."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        cur.execute("SELECT doc_id, COUNT(*) as chunk_count FROM documents GROUP BY doc_id")
        results = cur.fetchall()

        print("\n📚 Documents in database:")
        for doc_id, count in results:
            print(f"   {doc_id}: {count} chunks")

        cur.close()
        conn.close()

    except Exception as e:
        print(f"❌ Failed to list documents: {e}")


def build_pgvector_with_knowledge():
    """Build PostgreSQL connection string."""
    # List of PDF files to process
    pdf_files = [
        "./knowledge/history.pdf",
        "./knowledge/location_working_hours.pdf",
        "./knowledge/pizza_menu.pdf",
        "./knowledge/sandwiches_menu.pdf"
        # Add your PDF paths here
    ]

    # Process PDFs (choose one method)
    chunks = process_pdfs_to_pgvector(pdf_files, use_langchain_method=False)

if __name__ == "__main__":
    # Test search
    demo_search("Can you give me the location of La Famiglia Cucina in Chicago and working hours?")

    # List what's in the database
    list_documents_in_db()