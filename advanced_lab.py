"""
Virtual Lab API Implementation - Based on the Nature Paper Approach
Programmatic agent orchestration for APOE analysis
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import anthropic
import openai
from datetime import datetime
import requests
from urllib.parse import quote

@dataclass
class Agent:
    name: str
    title: str
    expertise: str
    goal: str
    role: str
    prompt_template: str

@dataclass
class Meeting:
    meeting_type: str  # "team" or "individual"
    agenda: str
    participants: List[str]
    rounds: int = 3
    temperature: float = 0.7
    max_tokens: int = 4000

class VirtualLab:
    """
    Virtual Lab implementation based on the Nature paper methodology
    Supports both Claude (Anthropic) and GPT-4 (OpenAI) APIs
    """
    
    def __init__(self, 
                 api_provider: str = "anthropic",  # or "openai"
                 api_key: str = None,
                 project_dir: str = None):
        
        self.api_provider = api_provider
        
        # Create timestamped project directory
        if project_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            project_dir = f"./apoe_virtual_lab_{timestamp}"
        
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organized output
        (self.project_dir / "meetings").mkdir(exist_ok=True)
        (self.project_dir / "reports").mkdir(exist_ok=True)
        (self.project_dir / "json_data").mkdir(exist_ok=True)
        (self.project_dir / "code_implementations").mkdir(exist_ok=True)
        
        # Initialize API client
        if api_provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = "claude-3-5-sonnet-20241022"
        elif api_provider == "openai":
            self.client = openai.OpenAI(api_key=api_key)
            self.model = "gpt-4o"  # Same as Virtual Lab paper
        else:
            raise ValueError("api_provider must be 'anthropic' or 'openai'")
            
        self.agents = {}
        self.meeting_history = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.project_dir / 'virtual_lab.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def create_agent(self, 
                     name: str, 
                     title: str, 
                     expertise: str, 
                     goal: str, 
                     role: str) -> Agent:
        """Create a specialized agent with defined expertise"""
        
        # Base prompt template following Virtual Lab paper structure
        prompt_template = f"""
You are {name}, a {title} in a Virtual Lab studying the APOE region for Alzheimer's disease.

EXPERTISE: {expertise}
GOAL: {goal}
ROLE: {role}

PROJECT CONTEXT:
- Analyzing APOE region (chr19:44-46Mb) for Alzheimer's disease
- Have GWAS summary statistics (~500k samples) and fine-mapped molecular QTL data
- No individual-level GWAS data, but have individual-level molecular QTL genotype/phenotype data
- MAJOR CHALLENGES:
  1. APOE E2/E3/E4 signals too strong, especially E4 - overshadows other signals when conditioning
  2. LD reference panel mismatches create spurious independent signals
  3. Multiple xQTL colocalizations may be LD artifacts from E4 dominance
  4. xQTL data includes effects on nearby genes beyond APOE - need to distinguish direct vs indirect effects
- Goal: Find independent AD-predisposing variants beyond E2/E3/E4 with robust methodology

CAPABILITIES:
- You have access to internet search to look up latest methods, software, and research
- If uncertain about specific approaches, you can search for current best practices
- Always ground recommendations in the most recent and robust methodologies

INSTRUCTIONS:
- Provide expert analysis within your area of expertise
- Be specific about methods, software, and parameters
- Suggest quality control steps and validation approaches
- Identify potential issues and limitations
- If you need to verify or find information about methods/tools, use web search
- Consider the challenge of analyzing ~300 genes in the region - suggest prioritization strategies

When participating in meetings:
- Contribute your specialized perspective
- Build on other agents' ideas constructively
- Ask clarifying questions when needed
- Provide concrete, actionable recommendations
- Search for information if you're uncertain about specific technical details
"""

        agent = Agent(
            name=name,
            title=title,
            expertise=expertise,
            goal=goal,
            role=role,
            prompt_template=prompt_template
        )
        
        self.agents[name] = agent
        
        # Save agent configuration
        agent_file = self.project_dir / f"agent_{name.lower().replace(' ', '_')}.json"
        with open(agent_file, 'w') as f:
            json.dump({
                'name': name,
                'title': title,
                'expertise': expertise,
                'goal': goal,
                'role': role,
                'created': datetime.now().isoformat()
            }, f, indent=2)
            
        self.logger.info(f"Created agent: {name}")
        return agent

    def web_search(self, query: str, num_results: int = 3) -> str:
        """Perform web search for latest information"""
        try:
            # Using DuckDuckGo search API (no key required)
            url = f"https://api.duckduckgo.com/?q={quote(query)}&format=json&no_html=1&skip_disambig=1"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                # Get abstract if available
                if data.get('Abstract'):
                    results.append(f"Summary: {data['Abstract']}")
                
                # Get related topics
                for topic in data.get('RelatedTopics', [])[:num_results]:
                    if isinstance(topic, dict) and topic.get('Text'):
                        results.append(f"• {topic['Text']}")
                
                return "\n".join(results) if results else "No relevant results found."
            
            return "Search temporarily unavailable."
            
        except Exception as e:
            self.logger.warning(f"Web search failed: {e}")
            return "Web search unavailable - proceeding with existing knowledge."

    async def call_llm(self, 
                       prompt: str, 
                       temperature: float = 0.7,
                       max_tokens: int = 4000,
                       enable_search: bool = True) -> str:
        """Make API call to LLM with optional web search capability"""
        
        # Add search capability to prompt if enabled
        if enable_search:
            search_instructions = """
            
WEB SEARCH CAPABILITY:
If you need to verify information or find the latest methods/tools, you can request a web search by including in your response:
[SEARCH: your search query]

I will then provide you with current information to inform your recommendations.
"""
            prompt += search_instructions
        
        try:
            if self.api_provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.content[0].text
                
            elif self.api_provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                response_text = response.choices[0].message.content
            
            # Process search requests if present
            if enable_search and "[SEARCH:" in response_text:
                import re
                search_pattern = r'\[SEARCH:\s*([^\]]+)\]'
                searches = re.findall(search_pattern, response_text)
                
                for search_query in searches:
                    search_results = self.web_search(search_query.strip())
                    response_text = response_text.replace(f"[SEARCH: {search_query}]", 
                                                        f"\n\n**Web Search Results for '{search_query}':**\n{search_results}\n")
            
            return response_text
                
        except Exception as e:
            self.logger.error(f"LLM API call failed: {e}")
            raise

    async def run_individual_meeting(self, 
                                   agent_name: str, 
                                   task: str,
                                   context: str = "",
                                   rounds: int = 1,
                                   temperature: float = 0.7) -> Dict[str, Any]:
        """
        Run individual meeting with specific agent
        Based on Virtual Lab paper individual meeting structure
        """
        
        if agent_name not in self.agents:
            raise ValueError(f"Agent {agent_name} not found")
            
        agent = self.agents[agent_name]
        
        # Construct meeting prompt
        meeting_prompt = f"""
{agent.prompt_template}

INDIVIDUAL MEETING TASK:
{task}

ADDITIONAL CONTEXT:
{context}

Please provide:
1. Your analysis approach for this specific task
2. Detailed methodology recommendations
3. Expected outcomes and validation steps
4. Potential limitations and how to address them
5. Next steps and follow-up analyses

Be specific about software, parameters, and implementation details.
"""

        self.logger.info(f"Running individual meeting with {agent_name}")
        
        # Multiple rounds if specified (like paper's iterative approach)
        responses = []
        for round_num in range(rounds):
            if round_num > 0:
                meeting_prompt += f"\n\nPREVIOUS ANALYSIS:\n{responses[-1]}\n\nPlease refine and improve your recommendations:"
                
            response = await self.call_llm(meeting_prompt, temperature=temperature)
            responses.append(response)
        
        # Save meeting results
        meeting_result = {
            'type': 'individual',
            'agent': agent_name,
            'task': task,
            'context': context,
            'rounds': rounds,
            'responses': responses,
            'final_response': responses[-1],
            'timestamp': datetime.now().isoformat()
        }
        
        self.meeting_history.append(meeting_result)
        
        # Save JSON data
        json_file = self.project_dir / "json_data" / f"individual_meeting_{len(self.meeting_history)}.json"
        with open(json_file, 'w') as f:
            json.dump(meeting_result, f, indent=2)
        
        # Save readable text report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        readable_file = self.project_dir / "meetings" / f"{agent_name.lower().replace(' ', '_')}_{timestamp}.txt"
        
        with open(readable_file, 'w') as f:
            f.write(f"INDIVIDUAL CONSULTATION: {agent_name}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {meeting_result['timestamp']}\n")
            f.write(f"Agent: {agent_name}\n")
            f.write(f"Rounds: {rounds}\n\n")
            f.write("TASK:\n")
            f.write("-" * 10 + "\n")
            f.write(f"{task}\n\n")
            if context:
                f.write("CONTEXT:\n")
                f.write("-" * 10 + "\n")
                f.write(f"{context}\n\n")
            f.write("RESPONSE:\n")
            f.write("-" * 10 + "\n")
            f.write(responses[-1])
            
        self.logger.info(f"✅ Individual meeting saved: {readable_file}")
            
        return meeting_result

    async def run_team_meeting(self, 
                             agenda: str,
                             participants: List[str],
                             rounds: int = 3,
                             temperature: float = 0.7) -> Dict[str, Any]:
        """
        Run team meeting with multiple agents
        Based on Virtual Lab paper team meeting structure
        """
        
        # Validate participants
        for participant in participants:
            if participant not in self.agents:
                raise ValueError(f"Agent {participant} not found")
        
        # Construct team meeting prompt
        team_prompt = f"""
VIRTUAL LAB TEAM MEETING

AGENDA: {agenda}

PARTICIPANTS:
"""
        
        for participant in participants:
            agent = self.agents[participant]
            team_prompt += f"- {agent.name} ({agent.title}): {agent.expertise[:100]}...\n"
            
        team_prompt += f"""

PROJECT CONTEXT:
- Analyzing APOE region (chr19:44-46Mb) for Alzheimer's disease independent signals
- Have GWAS summary statistics (~500k samples) and fine-mapped molecular QTL data
- No individual-level GWAS data, but have individual-level molecular QTL genotype/phenotype data
- Major methodological challenges with E2/E3/E4 conditioning and LD artifacts
- Need to analyze ~300 genes in region - require prioritization strategies
- Need robust, validated approaches for this complex region

MEETING STRUCTURE:
This is a {rounds}-round scientific discussion. For each round:

**ROUND 1: Initial Recommendations**
Each participant provides their initial analysis and recommendations for their area of expertise.

**ROUND 2: Integration and Refinement** 
Participants build on each other's ideas, identify dependencies between approaches, and refine recommendations based on interdisciplinary feedback.

**ROUND 3: Synthesis and Implementation**
Final synthesis of all approaches into a coherent analysis pipeline with clear priorities, validation standards, and implementation steps.

Please simulate this productive {rounds}-round scientific discussion where:
- Each participant contributes their specialized expertise in each round
- Ideas are built upon and refined across rounds
- Methodological challenges are systematically addressed
- Realistic prioritization strategies are developed (given ~300 candidate genes)

Include in each round:
- Specific methodological recommendations with software/parameters
- Quality control considerations and diagnostics
- Validation strategies and negative controls
- Realistic resource and time estimates

END WITH: Concrete action items, clear task assignments for each participant, and prioritized implementation timeline.
"""

        self.logger.info(f"Running team meeting: {agenda}")
        self.logger.info(f"Participants: {', '.join(participants)}")
        
        response = await self.call_llm(team_prompt, temperature=temperature)
        
        # Save meeting results
        meeting_result = {
            'type': 'team',
            'agenda': agenda,
            'participants': participants,
            'rounds': rounds,
            'response': response,
            'timestamp': datetime.now().isoformat()
        }
        
        self.meeting_history.append(meeting_result)
        
        # Save JSON data
        json_file = self.project_dir / "json_data" / f"team_meeting_{len(self.meeting_history)}.json"
        with open(json_file, 'w') as f:
            json.dump(meeting_result, f, indent=2)
        
        # Save readable text report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        readable_file = self.project_dir / "meetings" / f"team_meeting_{timestamp}.txt"
        
        with open(readable_file, 'w') as f:
            f.write("VIRTUAL LAB TEAM MEETING\n")
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {meeting_result['timestamp']}\n")
            f.write(f"Participants: {', '.join(participants)}\n")
            f.write(f"Rounds: {rounds}\n\n")
            f.write("AGENDA:\n")
            f.write("-" * 10 + "\n")
            f.write(f"{agenda}\n\n")
            f.write("DISCUSSION:\n")
            f.write("-" * 10 + "\n")
            f.write(response)
            
        self.logger.info(f"✅ Team meeting saved: {readable_file}")
            
        return meeting_result

    async def run_parallel_meetings(self, 
                                  meeting_config: Meeting,
                                  num_parallel: int = 3,
                                  creative_temp: float = 0.8,
                                  merge_temp: float = 0.2) -> Dict[str, Any]:
        """
        Run parallel meetings for robustness (like Virtual Lab paper)
        Then merge results with lower temperature for consistency
        """
        
        self.logger.info(f"Running {num_parallel} parallel meetings")
        
        # Run parallel meetings with high temperature for creativity
        parallel_tasks = []
        for i in range(num_parallel):
            if meeting_config.meeting_type == "team":
                task = self.run_team_meeting(
                    meeting_config.agenda,
                    meeting_config.participants,
                    meeting_config.rounds,
                    creative_temp
                )
            else:  # individual
                task = self.run_individual_meeting(
                    meeting_config.participants[0],  # First participant for individual
                    meeting_config.agenda,
                    rounds=meeting_config.rounds,
                    temperature=creative_temp
                )
            parallel_tasks.append(task)
            
        parallel_results = await asyncio.gather(*parallel_tasks)
        
        # Merge results with lower temperature for consistency
        merge_prompt = f"""
PARALLEL MEETING SYNTHESIS

You are synthesizing results from {num_parallel} parallel meetings on: {meeting_config.agenda}

PARALLEL MEETING RESULTS:
"""
        
        for i, result in enumerate(parallel_results):
            if meeting_config.meeting_type == "team":
                merge_prompt += f"\n--- Meeting {i+1} ---\n{result['response']}\n"
            else:
                merge_prompt += f"\n--- Meeting {i+1} ---\n{result['final_response']}\n"
                
        merge_prompt += """

SYNTHESIS TASK:
Please synthesize these parallel discussions into a single, coherent response that:
1. Combines the best insights from all meetings
2. Resolves any contradictions or disagreements
3. Provides clear, actionable recommendations
4. Maintains scientific rigor and methodological soundness

Focus on convergent themes and most robust recommendations.
"""

        merged_response = await self.call_llm(merge_prompt, temperature=merge_temp)
        
        # Save synthesis results
        synthesis_result = {
            'type': 'parallel_synthesis',
            'original_meeting': meeting_config.__dict__,
            'num_parallel': num_parallel,
            'parallel_results': parallel_results,
            'synthesis': merged_response,
            'timestamp': datetime.now().isoformat()
        }
        
        self.meeting_history.append(synthesis_result)
        
        # Save to file
        synthesis_file = self.project_dir / f"parallel_synthesis_{len(self.meeting_history)}.json"
        with open(synthesis_file, 'w') as f:
            json.dump(synthesis_result, f, indent=2)
            
        return synthesis_result

    def get_meeting_summary(self) -> str:
        """Generate summary of all meetings conducted"""
        
        summary = f"VIRTUAL LAB MEETING SUMMARY\n"
        summary += f"Total meetings: {len(self.meeting_history)}\n\n"
        
        for i, meeting in enumerate(self.meeting_history):
            summary += f"Meeting {i+1}: {meeting['type']}\n"
            if meeting['type'] == 'team':
                summary += f"  Agenda: {meeting['agenda']}\n"
                summary += f"  Participants: {', '.join(meeting['participants'])}\n"
            elif meeting['type'] == 'individual':
                summary += f"  Agent: {meeting['agent']}\n"
                summary += f"  Task: {meeting['task']}\n"
            summary += f"  Timestamp: {meeting['timestamp']}\n\n"
            
        return summary

# APOE-specific agent setup
def setup_apoe_virtual_lab(api_provider: str = "anthropic", api_key: str = None, project_dir: str = None) -> VirtualLab:
    """Initialize Virtual Lab with APOE-specific agents"""
    
    lab = VirtualLab(api_provider=api_provider, api_key=api_key, project_dir=project_dir)
    
    # Create specialized agents for APOE analysis
    lab.create_agent(
        name="Dr. Sarah Chen",
        title="LD Reference Panel Specialist", 
        expertise="LD reference panel accuracy and population matching, conditioning artifacts from strong genetic signals, alternative conditioning strategies (imputing z-scores from given z-scores and reference panels), diagnosing spurious associations from LD mismatches, analysis with GWAS summary statistics and LD references without individual-level data, analysis of complex LD regions like MHC and chromosome 19 around APOE",
        goal="Diagnose LD problems and develop robust conditioning approaches that work despite APOE E4 dominance",
        role="Lead methodological analysis of LD-sensitive conditioning approaches and diagnostic frameworks, with focus on methodological rigor, multiple validation approaches, and identification of confounders and artifacts"
    )
    
    lab.create_agent(
        name="Dr. Raj Patel",
        title="Advanced Colocalization Methodologist",
        expertise="Multi-signal colocalization methods (COLOC-SuSiE, eCAVIAR, colocboost), distinguishing true colocalization from LD artifacts, cross-tissue molecular QTL integration, credible set interpretation and validation, using and interpreting diverse xQTL data sources for colocalization analysis",
        goal="Determine which molecular colocalizations represent true biological signals versus LD echoes from E4 dominance", 
        role="Lead molecular QTL integration and colocalization analysis with emphasis on cross-tissue validation, conditional molecular analyses, effect size coherence assessment, and both cis and trans colocalization across multiple datasets"
    )
    
    lab.create_agent(
        name="Dr. Lisa Wang", 
        title="Fine-mapping Robustness Expert",
        expertise="Fine-mapping methods for complex LD regions (SuSiE, FINEMAP, PolyFun), fine-mapping diagnostics and model validation, handling strong confounding signals, multi-method convergent evidence approaches",
        goal="Develop robust fine-mapping strategies that work despite APOE E4's overwhelming effects",
        role="Lead statistical fine-mapping and credible set validation using multiple methods for cross-validation, model diagnostics, and approaches specifically designed for strong confounder scenarios with emphasis on credible set stability and sensitivity analysis"
    )
    
    lab.create_agent(
        name="Dr. Michael Torres",
        title="APOE Biology Specialist",
        expertise="APOE isoform biology beyond E2/E3/E4, known regulatory variants in APOE region, APOE-independent AD pathways in 19q13, functional validation approaches for APOE variants, molecular regulations near APOE region, knowledge of xQTL near APOE including cis and trans effects from brain and CSF",
        goal="Evaluate biological plausibility of candidate independent variants and design functional validation strategies",
        role="Lead biological interpretation with focus on known APOE regulatory mechanisms, APOE-independent genes in the region (TOMM40, APOC1, etc.), tissue-specific and cell-type effects, and xQTL-informed analysis focused on brain and CSF regions"
    )
    
    lab.create_agent(
        name="Dr. Elena Rodriguez",
        title="Scientific Critic",
        expertise="Critical evaluation of genetic association studies, methodological weakness identification, evidence strength evaluation, reproducibility assessment, validation approach design",
        goal="Ensure methodological rigor and identify potential confounders through skeptical but constructive criticism",
        role="Provide critical evaluation of all analyses and findings, question assumptions and methodologies, suggest negative controls and validation experiments, identify alternative explanations, and establish evidence strength standards"
    )
    
    lab.create_agent(
        name="Dr. Alex Cho",
        title="Bioinformatics Implementation Engineer",
        expertise="R programming and statistical computing, bash scripting and workflow automation, Python for data analysis and integration, implementation of bioinformatics pipelines, integration of multiple software tools and databases, reproducible research workflows, data visualization and reporting",
        goal="Translate methodological recommendations into robust, reproducible computational workflows and code implementations",
        role="Lead the practical implementation of all analysis pipelines, develop R scripts for statistical analyses, create bash workflows for tool integration, implement quality control and validation procedures, and ensure reproducible and well-documented code for all recommended methodologies"
    )
    
    return lab

# Example usage script
async def run_apoe_analysis():
    """Example workflow for APOE analysis"""
    
    # Initialize Virtual Lab
    lab = setup_apoe_virtual_lab(
        api_provider="anthropic",  # or "openai"
        api_key="your-api-key-here"  # Set your API key
    )
    
    print(f"Initialized Virtual Lab with {len(lab.agents)} agents")
    
    # Phase 1: Project specification team meeting
    team_result = await lab.run_team_meeting(
        agenda="Project planning: Strategy for identifying independent APOE signals beyond E2/E3/E4 while addressing methodological challenges and implementation requirements",
        participants=["Dr. Sarah Chen", "Dr. Raj Patel", "Dr. Lisa Wang", "Dr. Michael Torres", "Dr. Elena Rodriguez", "Dr. Alex Cho"],
        rounds=3
    )
    
    print("Team meeting completed")
    
    # Phase 2: Individual consultations
    ld_result = await lab.run_individual_meeting(
        "Dr. Sarah Chen",
        "Diagnose LD reference panel accuracy and develop robust conditioning strategies for APOE E4 dominance",
        context="APOE E4 effect creates spurious independent signals when conditioning due to LD reference panel mismatches"
    )
    
    coloc_result = await lab.run_individual_meeting(
        "Dr. Raj Patel", 
        "Develop robust colocalization pipeline to distinguish true molecular signals from LD artifacts",
        context="Multiple xQTL datasets show colocalization but may be false positives from E4 LD effects"
    )
    
    # Phase 2b: Bioinformatics implementation planning
    implementation_result = await lab.run_individual_meeting(
        "Dr. Alex Cho",
        "Design comprehensive computational workflow for APOE analysis pipeline integration",
        context="Based on methodological recommendations from LD, colocalization, and fine-mapping experts, create detailed implementation plan with R/bash/Python workflows"
    )
    
    # Phase 3: Results synthesis with parallel meetings for robustness
    synthesis_meeting = Meeting(
        meeting_type="team",
        agenda="Synthesize all recommendations into integrated, implementable analysis pipeline with realistic prioritization for ~300 candidate genes",
        participants=["Dr. Sarah Chen", "Dr. Raj Patel", "Dr. Lisa Wang", "Dr. Elena Rodriguez", "Dr. Alex Cho"],
        rounds=3
    )
    
    synthesis_result = await lab.run_parallel_meetings(
        synthesis_meeting,
        num_parallel=3,  # Like Virtual Lab paper
        creative_temp=0.8,
        merge_temp=0.2,
        all_agents=True  # Include all 6 agents
    )
    
    print("Analysis complete!")
    print(lab.get_meeting_summary())
    
    return lab

if __name__ == "__main__":
    # Run the analysis
    asyncio.run(run_apoe_analysis())