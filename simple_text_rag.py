from crewai import Agent, Task, Crew, Process, LLM
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

# Create a knowledge source with pizza store information
pizza_store_info = """
MARIO'S PIZZA PALACE - STORE INFORMATION:

STORE HOURS:
- Monday-Thursday: 11:00 AM - 10:00 PM
- Friday-Saturday: 11:00 AM - 11:00 PM  
- Sunday: 12:00 PM - 9:00 PM

CONTACT DETAILS:
- Phone: (555) 123-PIZZA
- Address: 123 Main Street, Downtown City, NY 12345
- Email: info@mariospizzapalace.com
- Website: www.mariospizzapalace.com

DELIVERY & PICKUP:
- Free delivery within 5 miles
- Pickup orders ready in 15-20 minutes
- Delivery time: 30-45 minutes
- Minimum order for delivery: $15

SPECIAL HOURS:
- New Year's Day: Closed
- Christmas Day: Closed  
- Thanksgiving: 2:00 PM - 8:00 PM
- All other holidays: Regular hours
"""

llm = LLM(model="ollama/mistral:7b", base_url="http://localhost:11434", temperature=0)

# Create knowledge source
knowledge_source = StringKnowledgeSource(
    content=pizza_store_info,
    metadata={"source": "store_info", "type": "business_details"}
)

# Define an agent that can answer store inquiries
store_assistant = Agent(
    role="Store Information Assistant",
    goal="Provide accurate information about Mario's Pizza Palace hours, contact details, and services",
    backstory="""You are a friendly customer service assistant for Mario's Pizza Palace. 
    You help customers with questions about store hours, contact information, delivery details,
    and other store-related inquiries. You always provide accurate information from the 
    official store database.""",
    llm=llm,
    verbose=True,
)

# Define a task that handles customer inquiries
customer_inquiry_task = Task(
    description="""Answer the following customer question about Mario's Pizza Palace:
    '{question}'

    Use the store information database to provide accurate and helpful details.""",
    expected_output="""A friendly, clear answer that provides the requested information
    about store hours, contact details, or services""",
    agent=store_assistant
)

# Create the crew
pizza_store_crew = Crew(
    agents=[store_assistant],
    tasks=[customer_inquiry_task],
    process=Process.sequential,
    knowledge_sources=[knowledge_source],
    verbose=True
)

if __name__ == "__main__":
    # Example customer questions to test
    sample_questions = [
        # "What time do you close on weekends?",
        "What's your phone number and address?",
        # "Do you deliver and what's the minimum order?",
        # "Are you open on Thanksgiving?",
        # "What are your hours on Sunday?",
        # "How long does pickup take?"
    ]

    # Execute with different customer questions
    for question in sample_questions:
        print(f"\n{'=' * 60}")
        print(f"CUSTOMER QUESTION: {question}")
        print('=' * 60)

        # Pass the question as input to kickoff
        result = pizza_store_crew.kickoff(inputs={"question": question})
        print(f"\nSTORE ASSISTANT: {result}")
        print("\n" + "-" * 60)