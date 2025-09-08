import os
import json
from typing import List, Dict, Any
from crewai import Agent, Task, Crew, LLM
from crewai_tools import RagTool
from langchain.document_loaders import PyPDFLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_ollama import OllamaEmbeddings


class MultiFormatRAGProcessor:
    """Handles processing of multiple file formats for RAG"""

    def __init__(self, embeddings_model=None):
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434", temperature=0.0)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def load_pdf_documents(self, pdf_paths: List[str]) -> List[Document]:
        """Load and process PDF documents"""
        documents = []
        for pdf_path in pdf_paths:
            if os.path.exists(pdf_path):
                loader = PyPDFLoader(pdf_path)
                docs = loader.load()
                # Add metadata
                for doc in docs:
                    doc.metadata['source_type'] = 'pdf'
                    doc.metadata['file_name'] = os.path.basename(pdf_path)
                documents.extend(docs)
        return documents

    def load_json_documents(self, json_paths: List[str]) -> List[Document]:
        """Load and process JSON documents"""
        documents = []
        for json_path in json_paths:
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Handle different JSON structures
                if isinstance(data, list):
                    # Array of objects
                    for i, item in enumerate(data):
                        content = json.dumps(item, indent=2)
                        doc = Document(
                            page_content=content,
                            metadata={
                                'source': json_path,
                                'source_type': 'json',
                                'file_name': os.path.basename(json_path),
                                'item_index': i
                            }
                        )
                        documents.append(doc)
                else:
                    # Single object
                    content = json.dumps(data, indent=2)
                    doc = Document(
                        page_content=content,
                        metadata={
                            'source': json_path,
                            'source_type': 'json',
                            'file_name': os.path.basename(json_path)
                        }
                    )
                    documents.append(doc)
        return documents

    def create_vector_store(self, documents: List[Document], persist_directory: str = "./vector_db"):
        """Create and persist vector store from documents"""
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)

        # Create vector store
        vectorstore = FAISS.from_documents(chunks, self.embeddings)

        # Save vector store locally
        vectorstore.save_local(persist_directory)

        return vectorstore


class CrewAIRAGSystem:
    """Main RAG system using CrewAI"""

    def __init__(self, vector_db_path: str = "./vector_db"):
        self.llm = LLM(model="ollama/mistral:7b", temperature=0.2, base_url="http://localhost:11434")
        self.vector_db_path = vector_db_path
        self.rag_tool = None
        self.setup_rag_tool()

    def setup_rag_tool(self):
        """Initialize RagTool with the vector database"""
        if os.path.exists(self.vector_db_path):
            self.rag_tool = RagTool(
                name="Knowledge Base Search",
                description="Search through the knowledge base of PDF and JSON documents to find relevant information",
                vectorstore_path=self.vector_db_path,
            )
        else:
            print(f"Vector database not found at {self.vector_db_path}. Please create it first.")

    def create_agents(self):
        # Research Agent
        researcher = Agent(
            role='Knowledge Researcher',
            goal='Find and extract relevant information from the knowledge base',
            backstory="""You are an expert researcher with access to a comprehensive 
            knowledge base containing PDF documents and structured JSON data. Your job 
            is to search through this information efficiently and find the most relevant 
            content to answer user queries.""",
            tools=[self.rag_tool] if self.rag_tool else [],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

        # Analysis Agent
        analyzer = Agent(
            role='Information Analyst',
            goal='Analyze and synthesize information from multiple sources',
            backstory="""You are a skilled analyst who takes raw information from 
            various sources and synthesizes it into coherent, comprehensive answers. 
            You excel at connecting dots between different pieces of information and 
            providing insights.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )

        return researcher, analyzer

    def create_tasks(self, query: str, researcher: Agent, analyzer: Agent):
        """Create tasks for processing the query"""

        # Research Task
        research_task = Task(
            description=f"""
            Search the knowledge base for information related to: {query}

            Instructions:
            1. Use the Knowledge Base Search tool to find relevant documents
            2. Look for information in both PDF and JSON sources
            3. Extract key facts, data points, and relevant context
            4. Provide source references for all information found

            Query: {query}
            """,
            agent=researcher,
            expected_output="A comprehensive collection of relevant information with source references"
        )

        # Analysis Task
        analysis_task = Task(
            description=f"""
            Analyze the research findings and provide a comprehensive answer to the user's query.

            Instructions:
            1. Review all the information gathered by the researcher
            2. Synthesize the information into a coherent response
            3. Highlight key insights and important details
            4. Maintain source attributions
            5. If information is conflicting, note the discrepancies

            Original Query: {query}
            """,
            agent=analyzer,
            expected_output="A well-structured, comprehensive answer with clear source attributions"
        )

        return [research_task, analysis_task]

    def query(self, question: str):
        """Process a query using the RAG system"""
        if not self.rag_tool:
            return "RAG tool not initialized. Please ensure vector database exists."

        # Create agents
        researcher, analyzer = self.create_agents()

        # Create tasks
        tasks = self.create_tasks(question, researcher, analyzer)

        # Create and run crew
        crew = Crew(
            agents=[researcher, analyzer],
            tasks=tasks,
            verbose=True
        )

        result = crew.kickoff()
        return result


def setup_rag_system(pdf_files: List[str], json_files: List[str], vector_db_path: str = "./vector_db"):
    """Setup the complete RAG system with your documents"""

    print("Setting up RAG system...")

    # Initialize processor
    processor = MultiFormatRAGProcessor()

    # Load documents
    print("Loading PDF documents...")
    pdf_docs = processor.load_pdf_documents(pdf_files)
    print(f"Loaded {len(pdf_docs)} PDF documents")

    print("Loading JSON documents...")
    json_docs = processor.load_json_documents(json_files)
    print(f"Loaded {len(json_docs)} JSON documents")

    # Combine all documents
    all_documents = pdf_docs + json_docs
    print(f"Total documents: {len(all_documents)}")

    # Create vector store
    print("Creating vector database...")
    processor.create_vector_store(all_documents, vector_db_path)
    print("Vector database created successfully!")

    # Initialize RAG system
    rag_system = CrewAIRAGSystem(vector_db_path)

    return rag_system


# Example usage
if __name__ == "__main__":
    # Set your OpenAI API key
    # os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

    # Define your file paths
    pdf_files = [
        "./knowledge/rd-gsg.pdf",
    ]

    json_files = [
        "./knowledge/restaurant.json",
    ]

    # Setup the RAG system
    rag_system = setup_rag_system(pdf_files, json_files)

    # Example queries
    queries = [
        "Do you carry any pasta items? If you do, what are they?"
    ]

    # Process queries
    for query in queries:
        print(f"\n{'=' * 50}")
        print(f"Query: {query}")
        print(f"{'=' * 50}")

        response = rag_system.query(query)
        print(response)
        print(f"{'=' * 50}\n")