from crewai import Agent, Task, Crew, Process, LLM
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
import json


# Simple menu knowledge for Giuseppe's Italian Kitchen

llm = LLM(model="ollama/mistral:7b", base_url="http://localhost:11434", temperature=0)

# Create JSON knowledge source
pdf_knowledge_source = PDFKnowledgeSource(
    file_paths= ["history.pdf", "location_working_hours.pdf", "pizza_menu.pdf", "sandwiches_menu.pdf"],
    metadata={"source": "restaurant_menu", "type": "menu_data"}
)

# Create menu assistant agent
menu_assistant = Agent(
    role="Restaurant Menu Assistant",
    goal="Help customers with menu questions, prices, and restaurant information",
    backstory="""You are a friendly server at Giuseppe's Italian Kitchen. 
    You help customers with menu questions, pricing, dietary restrictions, and restaurant details.
    You always provide accurate information from the official store database only.""",
    knowledge_sources=[pdf_knowledge_source],
    embedder={  # Agent can have its own embedder
        "provider": "ollama",
        "config": {"model": "nomic-embed-text"}
    },
    verbose=True,
    llm=llm
)

# Create task for handling menu questions
menu_question_task = Task(
    description="""Answer the customer's question about Giuseppe's Italian Kitchen:
    '{question}'

    Use the menu database to provide accurate information about items, prices, and details.""",
    expected_output="""A helpful answer with specific menu details and prices when relevant""",
    agent=menu_assistant
)

# Create the crew
menu_crew = Crew(
    agents=[menu_assistant],
    tasks=[menu_question_task],
    process=Process.sequential,
    # knowledge_sources=[json_knowledge_source],
    verbose=True
)

if __name__ == "__main__":
    # Sample questions to test
    sample_questions = [
        "Where's La Famiglia Cucina restaurant located in Chicago?"
    ]

    print("=== La Famiglia Cucina KITCHEN MENU ASSISTANT ===\n")

    # Test each question
    for question in sample_questions:
        print(f"\n{'=' * 60}")
        print(f"CUSTOMER: {question}")
        print('=' * 60)

        # Get answer from the crew
        result = menu_crew.kickoff(inputs={"question": question})
        print(f"\nSERVER: {result}")
        print("\n" + "-" * 60)

    # print(f"\n{'=' * 60}")
    # print("You can also ask custom questions:")
    # print('=' * 60)
    #
    # # Example of asking a custom question
    # custom_question = "What's your most expensive dish?"
    # result = menu_crew.kickoff(inputs={"question": custom_question})
    # print(f"Q: {custom_question}")
    # print(f"A: {result}")