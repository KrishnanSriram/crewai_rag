from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerpApiGoogleSearchTool
from dotenv import load_dotenv

load_dotenv()
llm = LLM(model="ollama/llama3.2", temperature=0.3, base_url="http://localhost:11434")
tool = SerpApiGoogleSearchTool()

tool = SerpApiGoogleSearchTool()

agent = Agent(
    role="Researcher",
    goal="Answer questions using Google search",
    backstory="Search specialist",
    tools=[tool],
    verbose=True,
    llm=llm
)

task = Task(
    description="Search for the latest CrewAI releases",
    expected_output="A concise list of relevant results with titles and links",
    agent=agent,
)

crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()
print(result)