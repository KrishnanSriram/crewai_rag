import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import RagTool
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# Make sure you have the required libraries installed:
# pip install crewai crewai_tools langchain-community langchain-openai chromadb pypdf

llm = LLM(model="ollama/mistral:7b", base_url="http://localhost:11434", temperature=0)
# Set your OpenAI API key as an environment variable
# os.environ = "YOUR_API_KEY"

# --- Step 1: Ingest and Index the PDF into a Vector Store ---
# This process loads the document, splits it into chunks, and creates a vector store.

# Load documents from a specific directory containing your PDF file.
# Make sure your PDF is in a folder named 'knowledge_base'
loader = PyPDFDirectoryLoader("./knowledge")
documents = loader.load()

# Split the documents into smaller, manageable chunks for better retrieval.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
splits = text_splitter.split_documents(documents)

# Create a vector store from the document chunks using OpenAI embeddings.
# This will handle creating the numerical vectors and storing them.
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

print("PDF knowledge base has been indexed successfully.")

# --- Step 2: Create the CrewAI RAG Tool ---
# The RAGTool is the "hand" that your agent will use to search the vector store.
# The config dictionary is passed to Embedchain, which powers the RagTool.
rag_config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "mistral:7b"
        }
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text"
        }
    }
}
# Wrap the vector store's retriever in a CrewAI RAGTool.
# This tool allows the agent to perform similarity searches on the vector data.
rag_tool = RagTool(
    name="PDF Knowledge Base Search Tool",
    description="A tool for searching and retrieving information from a PDF document knowledge base.",
    rag_retriever=vectorstore.as_retriever(),
    config=rag_config
)

# --- Step 3: Define Agents, Tasks, and Crew ---
# Now we set up the agent and task to use the newly created tool.

# Define the Agent that will use the RAG Tool
researcher_agent = Agent(
    role='Document Researcher',
    goal='Provide accurate and concise answers based on the provided PDF documents.',
    backstory='You are an expert researcher. Your sole purpose is to find and synthesize information from the document knowledge base.',
    tools=[rag_tool],
    llm=llm,
    verbose=True
)

# Define the Task for the agent. The description guides it to use the tool.
research_task = Task(
    description="""Answer the following customer question with data from your datastore  {question}'
                Your response should be based entirely on the retrieved content.""",
    expected_output='Specific answer summarized for easier understanding. Provide name of the file for reference in your final result',
    agent=researcher_agent
)

# Form the Crew with the agent and task.
research_crew = Crew(
    agents=[researcher_agent],
    tasks=[research_task],
    process=Process.sequential,
    verbose=True
)

# --- Step 4: Run the Crew ---
# Execute the task, which will cause the agent to use the RAGTool to search the PDF.
print("\n\nStarting the CrewAI process...\n")
query = {"question": "How can I add read replicas?"}
result = research_crew.kickoff(query)

print("\n\n########################")
print("## Final Agent Output ##")
print("########################\n")
print(result)