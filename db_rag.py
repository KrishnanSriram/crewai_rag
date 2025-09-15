import os
import json
import psycopg2
from typing import List, Dict, Any
from langchain.embeddings import OllamaEmbeddings
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool


llm = LLM(model="ollama/gemma3:12b", temperature=0.0, base_url="http://localhost:11434")
# Your existing database config
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


class DocumentSearchTool(BaseTool):
    name: str = "document_search"
    description: str = (
        "Search for relevant documents based on a query. "
        "Use this tool to find specific information from the document database. "
        "Input should be a search query string."
    )

    def _run(self, query: str) -> str:
        try:
            # Get embeddings model
            embeddings_model = OllamaEmbeddings(
                model=OLLAMA_CONFIG['model'],
                base_url=OLLAMA_CONFIG['base_url']
            )

            # Get query embedding
            query_embedding = embeddings_model.embed_query(query)

            # Search database
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()

            search_sql = """
                         SELECT doc_id, \
                                chunk_index, \
                                content, \
                                metadata,
                                1 - (embedding <=> %s::vector) as similarity
                         FROM documents
                         WHERE 1 - (embedding <=> %s::vector) > 0.3
                         ORDER BY embedding <=> %s::vector
                             LIMIT 10 \
                         """

            cur.execute(search_sql, (query_embedding, query_embedding, query_embedding))
            results = cur.fetchall()

            cur.close()
            conn.close()

            if not results:
                return f"No relevant documents found for query: '{query}'"

            # Format results for the agent
            formatted_results = []
            for i, row in enumerate(results, 1):
                doc_id, chunk_index, content, metadata, similarity = row
                formatted_results.append(
                    f"Document {i}:\n"
                    f"Source: {doc_id} (chunk {chunk_index})\n"
                    f"Relevance: {similarity:.3f}\n"
                    f"Content: {content[:500]}...\n"
                    f"---"
                )

            return "\n\n".join(formatted_results)

        except Exception as e:
            return f"Error searching documents: {str(e)}"


class DocumentListTool(BaseTool):
    name: str = "list_documents"
    description: str = (
        "Get a list of all available documents in the database. "
        "Use this to understand what documents are available to search through."
    )

    def _run(self, query: str = "") -> str:
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()

            cur.execute("""
                        SELECT doc_id,
                               COUNT(*)             as chunk_count,
                               MIN(LENGTH(content)) as min_chunk_size,
                               MAX(LENGTH(content)) as max_chunk_size
                        FROM documents
                        GROUP BY doc_id
                        ORDER BY doc_id
                        """)
            results = cur.fetchall()

            cur.close()
            conn.close()

            if not results:
                return "No documents found in the database."

            doc_list = ["Available Documents:"]
            for doc_id, count, min_size, max_size in results:
                doc_list.append(
                    f"📄 {doc_id}: {count} chunks "
                    f"(size range: {min_size}-{max_size} chars)"
                )

            return "\n".join(doc_list)

        except Exception as e:
            return f"Error listing documents: {str(e)}"


def create_search_agent():
    return Agent(
        role='Document Research Specialist',
        goal='Find and analyze relevant information from the document database',
        backstory=(
            "You are an expert research specialist who excels at finding "
            "relevant information from document databases. You understand "
            "how to interpret search results, synthesize information from "
            "multiple sources, and provide comprehensive answers."
        ),
        tools=[DocumentSearchTool(), DocumentListTool()],
        verbose=True,
        allow_delegation=False,
        llm=llm,
        max_retry_limit=5,
        max_iter=5
    )


def create_analyst_agent():
    return Agent(
        role='Information Analyst',
        goal='Analyze and synthesize information to provide clear, actionable insights',
        backstory=(
            "You are a skilled analyst who takes raw research data and "
            "transforms it into clear, well-structured insights. You excel "
            "at identifying key themes, summarizing complex information, "
            "and presenting findings in an organized manner."
        ),
        tools=[],
        verbose=True,
        allow_delegation=False,
        llm = llm,
        max_retry_limit=5,
        max_iter=5,
    )


def create_search_task(query: str, context: str = ""):
    return Task(
        description=(
            f"Search the document database for information related to: '{query}'"
            f"{f' Context: {context}' if context else ''}"
            f"\n\nSteps to complete:"
            f"\n1. Use the document_search tool to find relevant documents"
            f"\n2. If needed, use list_documents to understand available sources"
            f"\n3. Analyze the search results for relevance and quality"
            f"\n4. Identify the most important findings"
        ),
        expected_output=(
            "A comprehensive summary of search results including:"
            "\n- Key findings related to the query"
            "\n- Source documents and their relevance scores"
            "\n- Any important quotes or specific information found"
            "\n- Assessment of information quality and completeness"
        ),
        agent=create_search_agent()
    )


def create_analysis_task(query: str):
    return Task(
        description=(
            f"Analyze the research findings and provide a clear, structured response to: '{query}'"
            f"\n\nYour analysis should:"
            f"\n1. Synthesize information from multiple sources"
            f"\n2. Identify key themes and insights"
            f"\n3. Present information in a clear, organized format"
            f"\n4. Include relevant quotes and citations"
            f"\n5. Note any gaps or limitations in the available information"
        ),
        expected_output=(
            "A well-structured analytical report that includes:"
            "\n- Executive summary of findings"
            "\n- Detailed analysis with supporting evidence"
            "\n- Key quotes and citations from source documents"
            "\n- Conclusions and recommendations (if applicable)"
            "\n- Notes on information quality and any limitations"
        ),
        agent=create_analyst_agent(),
        context=[create_search_task(query)]  # This task depends on search results
    )


def search_documents_with_crewai(query: str, context: str = "") -> str:
    print(f"🔍 Starting CrewAI search for: '{query}'")

    # Create agents
    researcher = create_search_agent()
    analyst = create_analyst_agent()

    # Create tasks
    search_task = create_search_task(query, context)
    analysis_task = create_analysis_task(query)

    # Create crew
    crew = Crew(
        agents=[researcher], #, analyst],
        tasks=[search_task], #, analysis_task],
        process=Process.sequential,
        verbose=True
    )

    # Execute the crew
    try:
        result = crew.kickoff()
        return result
    except Exception as e:
        return f"CrewAI search failed: {str(e)}"


# Specific use case functions
def research_topic(topic: str, aspects: List[str] = None) -> str:
    if aspects:
        context = f"Focus on these aspects: {', '.join(aspects)}"
    else:
        context = ""

    return search_documents_with_crewai(topic, context)


# Example usage functions
def demo_searches():
    print("🎯 Running demo searches...")

    # Example 1: General topic search
    print("\n1. Can you give me the location of La Famiglia Cucina in Chicago and working hours?'...")
    result1 = research_topic("Can you give me the location of La Famiglia Cucina in Chicago and working hours?")


if __name__ == "__main__":
    demo_searches()