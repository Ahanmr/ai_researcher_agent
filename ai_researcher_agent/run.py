#!/usr/bin/env python
from dotenv import load_dotenv
import os

load_dotenv()

print("OpenAI Key loaded:", bool(os.getenv("OPENAI_API_KEY")))
print("Serper Key loaded:", bool(os.getenv("SERPER_API_KEY")))

from naptha_sdk.schemas import AgentRunInput
from naptha_sdk.utils import get_logger
from schemas import InputSchema, ResearchTopic
from typing import Dict, Any, List

from crewai import Agent, Task, Crew, Process
from textwrap import dedent
from langchain_openai import ChatOpenAI
import datetime
from crewai_tools import (
    SerperDevTool,
    FileReadTool,
    FileWriteTool
)
from langchain.tools import Tool

logger = get_logger(__name__)

class ResearcherAgent:
    def __init__(self, module_run: AgentRunInput):
        self.module_run = module_run
        self.llm_config = module_run.agent_deployment.agent_config.llm_config
        self.setup_tools()
        self.setup_agents()

    def setup_tools(self):
        """Initialize research tools"""
        self.search_tool = SerperDevTool()
        self.file_read = FileReadTool()
        self.file_write = FileWriteTool()

    def setup_agents(self):
        """Initialize the research agents with CrewAI"""
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
        
        self.keyword_researcher = Agent(
            role="Expert Keyword Research Specialist",
            goal="Analyze topics and identify the most effective search terms and phrases",
            backstory="""Expert keyword research specialist with deep understanding of search behavior 
                        and semantic analysis. Skilled at breaking down topics into relevant search terms.""",
            tools=[self.search_tool],
            llm=llm,
            verbose=True
        )

        self.search_analyst = Agent(
            role="Search Results Analyst",
            goal="Evaluate and refine search queries based on initial results",
            backstory="""Analytical expert specializing in evaluating search results and identifying 
                        patterns in successful queries.""",
            tools=[self.search_tool],
            llm=llm,
            verbose=True
        )

        self.insights_compiler = Agent(
            role="Research Insights Compiler",
            goal="Synthesize findings and create structured recommendations",
            backstory="""Research synthesis specialist who excels at organizing insights into clear, 
                        actionable recommendations.""",
            tools=[self.search_tool],
            llm=llm,
            verbose=True
        )

    def create_tasks(self, research_topic: ResearchTopic) -> List[Task]:
        """Create research tasks for the crew"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        task_1 = Task(
            description=f"""
                Analyze the following topic and identify potential keywords and search phrases:
                TOPIC: "{research_topic.topic}"
                RESEARCH OBJECTIVE: "{research_topic.research_objective}"
                SPECIFIC FOCUS: "{research_topic.specific_focus}"
                
                1. Break down the topic into main concepts
                2. Generate primary keywords for each concept
                3. Identify related terms and synonyms
                4. Consider different search intents
                5. List at least 5 potential search phrases
            """,
            agent=self.keyword_researcher,
            output_file=f'output-files/keyword_analysis_{timestamp}.md'
        )

        task_2 = Task(
            description="""
                Evaluate the effectiveness of the proposed keywords by analyzing search results:
                1. Test each main keyword/phrase identified
                2. Analyze the relevance of returned results
                3. Identify gaps or irrelevant results
                4. Suggest modifications to improve search accuracy
            """,
            agent=self.search_analyst,
            context=[task_1],
            output_file=f'output-files/search_analysis_{timestamp}.md'
        )

        task_3 = Task(
            description="""
                Compile final keyword recommendations and search strategy:
                1. Synthesize findings from keyword research and search analysis
                2. Create a prioritized list of recommended search terms
                3. Group related terms and phrases
                4. Provide specific recommendations for different search objectives
            """,
            agent=self.insights_compiler,
            context=[task_2],
            output_file=f'output-files/final_recommendations_{timestamp}.md'
        )

        return [task_1, task_2, task_3]

    def research(self, research_topic: ResearchTopic) -> Dict[str, Any]:
        """Conduct research using the CrewAI workflow"""
        
        tasks = self.create_tasks(research_topic)
        
        crew = Crew(
            agents=[self.keyword_researcher, self.search_analyst, self.insights_compiler],
            tasks=tasks,
            verbose=True,
            process=Process.sequential
        )

        result = crew.kickoff()
        
        try:
            research_results = {
                "keyword_analysis": result.get("task_1", {}),
                "search_analysis": result.get("task_2", {}),
                "final_recommendations": result.get("task_3", {})
            }
            return research_results
        except Exception as e:
            logger.error(f"Failed to parse research results: {str(e)}")
            return {
                "keyword_analysis": {},
                "search_analysis": {},
                "final_recommendations": {}
            }

def run(module_run: AgentRunInput):
    """Main entry point for the researcher agent"""
    researcher = ResearcherAgent(module_run)
    
    if isinstance(module_run.inputs, dict):
        input_params = InputSchema(**module_run.inputs)
    else:
        input_params = module_run.inputs
        
    research_results = researcher.research(input_params.research_topic)
    return research_results

if __name__ == "__main__":
    from naptha_sdk.client.naptha import Naptha
    from naptha_sdk.configs import load_agent_deployments

    naptha = Naptha()

    # Test input parameters
    input_params = InputSchema(
        research_topic=ResearchTopic(
            topic="Latest developments in AI agents and autonomous systems",
            context="Focus on practical applications and recent breakthroughs",
            depth="comprehensive"
        ),
        max_sources=5,
        include_citations=True
    )

    # Load agent deployment configuration
    agent_deployments = load_agent_deployments(
        "configs/agent_deployments.json", 
        load_persona_data=False, 
        load_persona_schema=False
    )

    # Create agent run input
    agent_run = AgentRunInput(
        inputs=input_params,
        agent_deployment=agent_deployments[0],
        consumer_id=naptha.user.id,
    )

    # Execute the agent
    response = run(agent_run)
    print("Research Results:", response)