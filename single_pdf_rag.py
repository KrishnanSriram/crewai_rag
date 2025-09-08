import os
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import PDFSearchTool

llm = LLM(model="ollama/mistral:7b", base_url="http://localhost:11434", temperature=0.3)

pdf_search_tool = PDFSearchTool(
    pdf="./knowledge/pizza_menu.pdf",
    config=dict(
        llm=dict(provider="ollama", config=dict(model="llama3.2")),
        embedder=dict(provider="ollama", config=dict(model="nomic-embed-text")),
    ),
)

# --- Agents ---
research_agent = Agent(
    role="Research Agent",
    goal="Search through the PDF to find relevant answers",
    allow_delegation=False,
    verbose=True,
    max_iter=5,
    max_retry_limit=5,
    backstory=(
        """
        You are an restaurant research agent is adept at searching and 
        extracting knowledge from documents, ensuring accurate and prompt responses.
        """
    ),
    llm=llm,
    tools=[pdf_search_tool],

)

# --- Tasks ---
answer_customer_question_task = Task(
    description=(
        """
        Answer the customer's questions based on the pizza menu PDF.
        The research agent will search through the PDF to find the relevant answers.
        Your final answer MUST be clear and accurate, based on the content of the pizza menu PDF.
        If you are not sure about the answer say Don't know or out of context question. Do not
        make up your own answer or search web for answers.

        Here is the customer's question:
        {customer_question}
        """
    ),
    expected_output="""
        Provide clear and accurate answers to the customer's questions based on 
        the content of the pizza menu PDF.
        """,
    tools=[pdf_search_tool],
    agent=research_agent,
)

# --- Crew ---
crew = Crew(
    agents=[research_agent],
    tasks=[answer_customer_question_task],
    process=Process.sequential,
)

customer_question = "What are the ingredients of Mediterranean Veggie?\n"

result = crew.kickoff(inputs={"customer_question": customer_question})
print(result)