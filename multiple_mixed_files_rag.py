from crewai import Crew, Task, Agent, LLM
from crewai_tools import RagTool

llm = LLM(model="ollama/gemma3:12b", base_url="http://localhost:11434", temperature=0.0)
config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "gemma3:12b",
        }
    },
    "embedding_model": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text"
        }
    }
}

rag_tool = RagTool(config=config)
RagTool()
rag_tool.add("./knowledge/rds-gsg.pdf", data_type="pdf_file")
rag_tool.add("./knowledge/history.pdf", data_type="pdf_file")
rag_tool.add("./knowledge/location_working_hours.pdf", data_type="pdf_file")
agent = Agent(
    role="Document Analysis Specialist",
    goal="Extract and analyze information from documents to answer user questions accurately",
    backstory="You are an expert document analyst designed to help users find specific information within documents and provide comprehensive answers based on the content",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=[rag_tool],
    max_retry_limit=5
)

task1 = Task(
    description="""Answer the following question based STRICTLY on the information available in the provided document: {query}

    INSTRUCTIONS:
    1. Use ONLY the RAG tool to search and retrieve information from the provided documents
    2. Do NOT use any external knowledge or general information not present in the documents
    3. If the information is not found in the documents, respond with: "I cannot find information about this topic in the provided document(s). Please ensure your question relates to the content of the uploaded document."
    4. Always cite specific sections or pages from the document when providing answers
    5. Be clear about what information is available and what is not available in the documents""",
    expected_output="""A response that either:
    - Provides a comprehensive answer based solely on document content with specific citations, OR
    - Clearly states that the requested information is not available in the provided documents
    - Give the name of the file referenced and page number, if available""",
    agent=agent
)
crew = Crew(agents=[agent], tasks=[task1], verbose=True)


# query = "How do I setup public access for RDS instance?"
# query = "Can you give me the location of La Famiglia Cucina in Chicago and working hours?"
# query = "How can I scale my RDS instance vertically?"
query = "Can you tell me about cosmos DB?"
task_output = crew.kickoff(inputs={"query": query})
