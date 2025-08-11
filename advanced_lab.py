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
                 project_dir: str = "./apoe_virtual_lab"):
        
        self.api_provider = api_provider
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organized output
        (self.project_dir / "meetings").mkdir(exist_ok=True)
        (self.project_dir / "reports").mkdir(exist_ok=True)
        (self.project_dir / "json_data").mkdir(exist_ok=True)
        
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
- MAJOR CHALLENGES:
  1. APOE E4 signal too strong - overshadows other signals when conditioning
  2. LD reference panel mismatches create spurious independent signals
  3. Multiple xQTL colocalizations may be LD artifacts from E4 dominance
- Goal: Find independent signals beyond E2/E3/E4 variants

INSTRUCTIONS:
- Provide expert analysis within your area of expertise
- Be specific about methods, software, and parameters
- Suggest quality control steps and validation approaches
- Identify potential issues and limitations
- Ground recommendations in latest research

When participating in meetings:
- Contribute your specialized perspective
- Build on other agents' ideas constructively
- Ask clarifying questions when needed
- Provide concrete, actionable recommendations
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

    async def call_llm(self, 
                       prompt: str, 
                       temperature: float = 0.7,
                       max_tokens: int = 4000) -> str:
        """Make API call to LLM"""
        
        try:
            if self.api_provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
                
            elif self.api_provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
                
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
            team_prompt += f"- {agent.name}: {agent.expertise}\n"
            
        team_prompt += f"""

PROJECT CONTEXT:
- Analyzing APOE region for Alzheimer's disease independent signals
- Major methodological challenges with E4 conditioning and LD artifacts
- Need robust, validated approaches for this complex region

MEETING STRUCTURE:
This is a {rounds}-round scientific discussion. In each round, each participant should:
1. Contribute their specialized expertise to the agenda
2. Build on previous participants' ideas
3. Identify potential issues or improvements
4. Suggest concrete next steps

Please simulate a productive scientific discussion where each participant contributes their unique perspective. Include:
- Specific methodological recommendations
- Quality control considerations  
- Validation strategies
- Task assignments for follow-up work

END WITH: Concrete action items and clear task assignments for each participant.
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
def setup_apoe_virtual_lab(api_provider: str = "anthropic", api_key: str = None) -> VirtualLab:
    """Initialize Virtual Lab with APOE-specific agents"""
    
    lab = VirtualLab(api_provider=api_provider, api_key=api_key)
    
    # Create specialized agents for APOE analysis
    lab.create_agent(
        name="Dr. Sarah Chen",
        title="LD Reference Panel Specialist", 
        expertise="LD reference panel accuracy, conditioning artifacts, population stratification effects",
        goal="Diagnose and solve LD reference panel conditioning problems",
        role="Lead methodological analysis of conditioning approaches and LD diagnostics"
    )
    
    lab.create_agent(
        name="Dr. Raj Patel",
        title="Advanced Colocalization Methodologist",
        expertise="Multi-signal colocalization, COLOC-SuSiE, cross-tissue validation, LD artifact detection",
        goal="Distinguish true molecular colocalization from LD artifacts", 
        role="Lead molecular QTL integration and colocalization analysis"
    )
    
    lab.create_agent(
        name="Dr. Lisa Wang", 
        title="Fine-mapping Robustness Expert",
        expertise="SuSiE, FINEMAP, PolyFun, robust inference under model misspecification",
        goal="Develop robust fine-mapping strategies for complex LD regions",
        role="Lead statistical fine-mapping and credible set validation"
    )
    
    lab.create_agent(
        name="Dr. Michael Torres",
        title="APOE Biology Specialist",
        expertise="APOE isoform biology, regulatory variants, APOE-independent mechanisms",
        goal="Evaluate biological plausibility of candidate independent variants",
        role="Lead biological interpretation and functional validation design"
    )
    
    lab.create_agent(
        name="Dr. Elena Rodriguez",
        title="Scientific Critic",
        expertise="Critical evaluation, methodology validation, alternative explanations",
        goal="Ensure methodological rigor and identify potential confounders",
        role="Provide critical evaluation and validation requirements"
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
        agenda="Project planning: Strategy for identifying independent APOE signals beyond E2/E3/E4 while addressing methodological challenges",
        participants=["Dr. Sarah Chen", "Dr. Raj Patel", "Dr. Lisa Wang", "Dr. Michael Torres", "Dr. Elena Rodriguez"]
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
    
    # Phase 3: Results synthesis with parallel meetings for robustness
    synthesis_meeting = Meeting(
        meeting_type="team",
        agenda="Synthesize methodological recommendations into integrated analysis pipeline",
        participants=["Dr. Sarah Chen", "Dr. Raj Patel", "Dr. Lisa Wang", "Dr. Elena Rodriguez"],
        rounds=3
    )
    
    synthesis_result = await lab.run_parallel_meetings(
        synthesis_meeting,
        num_parallel=3,  # Like Virtual Lab paper
        creative_temp=0.8,
        merge_temp=0.2
    )
    
    print("Analysis complete!")
    print(lab.get_meeting_summary())
    
    return lab

if __name__ == "__main__":
    # Run the analysis
    asyncio.run(run_apoe_analysis())