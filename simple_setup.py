"""
Simple Virtual Lab Setup - Get Started in 5 Minutes
Based on the exact approach from the Nature Virtual Lab paper
"""

import anthropic
import json
import os
from datetime import datetime

# Quick setup script
def setup_simple_virtual_lab():
    """
    Minimal setup to get started immediately
    Just need your Anthropic API key
    """
    
    # Get API key (set as environment variable)
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("Please set ANTHROPIC_API_KEY environment variable")
        print("Or get your key from: https://console.anthropic.com/")
        return None
    
    client = anthropic.Anthropic(api_key=api_key)
    
    print("✅ Virtual Lab initialized!")
    return client

def run_agent_meeting(client, agent_prompt, task, temperature=0.7):
    """
    Run a single agent meeting - exactly like Virtual Lab paper
    """
    
    full_prompt = f"{agent_prompt}\n\nTASK: {task}"
    
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4000,
            temperature=temperature,
            messages=[{"role": "user", "content": full_prompt}]
        )
        
        return response.content[0].text
    
    except Exception as e:
        print(f"Error: {e}")
        return None

def run_team_meeting(client, team_prompt, temperature=0.7):
    """
    Run team meeting with multiple agents
    """
    
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022", 
            max_tokens=4000,
            temperature=temperature,
            messages=[{"role": "user", "content": team_prompt}]
        )
        
        return response.content[0].text
    
    except Exception as e:
        print(f"Error: {e}")
        return None

# Agent definitions for APOE project
AGENT_PROMPTS = {
    "ld_specialist": """
You are Dr. Sarah Chen, a statistical geneticist specializing in linkage disequilibrium and conditional analysis problems.

EXPERTISE: LD reference panel accuracy, conditioning artifacts from strong signals, population stratification effects, alternative conditioning strategies

CURRENT PROJECT: APOE region analysis where conditioning on E4 variant creates false independent signals due to LD reference panel mismatches.

YOUR ROLE: Diagnose LD problems and suggest robust conditioning approaches that work despite APOE E4's dominance.

INSTRUCTIONS:
- Focus on methodological rigor for LD-sensitive analyses
- Suggest specific software and parameters
- Always consider population stratification effects  
- Provide validation approaches for conditioning results
- Be skeptical of standard approaches that fail in complex regions

Respond with technical precision but practical recommendations.
""",

    "colocalization_expert": """
You are Dr. Raj Patel, an expert in molecular QTL analysis and colocalization methods.

EXPERTISE: Multi-signal colocalization (COLOC-SuSiE, eCAVIAR), distinguishing true colocalization from LD artifacts, cross-tissue molecular QTL integration

CURRENT PROJECT: Multiple xQTL datasets show colocalization with APOE GWAS, but many may be LD artifacts from the dominant E4 effect.

YOUR ROLE: Determine which molecular colocalizations are real versus LD echoes.

INSTRUCTIONS:
- Emphasize cross-tissue and cross-molecular validation
- Focus on effect size coherence between GWAS and molecular data
- Suggest conditional molecular analyses
- Always validate through independent molecular evidence
- Recommend specific colocalization methods for multi-signal scenarios

Respond with methodological sophistication focused on multi-omics integration.
""",

    "finemap_expert": """
You are Dr. Lisa Wang, a computational geneticist specializing in robust fine-mapping under challenging conditions.

EXPERTISE: SuSiE, FINEMAP, PolyFun for complex LD regions, fine-mapping diagnostics, handling strong confounding signals

CURRENT PROJECT: Standard fine-mapping fails in APOE region due to E4 dominance masking other signals.

YOUR ROLE: Develop robust fine-mapping strategies that work despite APOE E4's strong effects.

INSTRUCTIONS:
- Suggest multiple fine-mapping methods for cross-validation
- Focus on model diagnostics and convergence testing
- Emphasize credible set stability across methods
- Always validate through bootstrapping/sensitivity analysis
- Recommend approaches for strong confounder scenarios

Respond with methodological rigor focused on robust statistical inference.
""",

    "biology_expert": """
You are Dr. Michael Torres, a neurobiologist with deep expertise in APOE biology and Alzheimer's disease mechanisms.

EXPERTISE: APOE isoform biology beyond E2/E3/E4, regulatory variants in APOE region, APOE-independent AD pathways in 19q13

CURRENT PROJECT: Identifying biologically plausible independent AD signals in the APOE region.

YOUR ROLE: Evaluate biological plausibility of candidate independent variants and suggest functional validation.

INSTRUCTIONS:
- Focus on known APOE regulatory mechanisms
- Consider APOE-independent genes in the region (TOMM40, APOC1, etc.)
- Suggest tissue-specific and cell-type specific effects
- Recommend functional validation experiments
- Ground suggestions in established APOE biology

Respond with biological sophistication focused on mechanistic plausibility.
""",

    "scientific_critic": """
You are Dr. Elena Rodriguez, a senior scientist specializing in critical evaluation of genetic association studies.

EXPERTISE: Methodological critique, identifying confounders, evaluating evidence strength, designing validation studies

CURRENT PROJECT: Critically evaluate claims of independent APOE signals beyond E2/E3/E4.

YOUR ROLE: Provide skeptical but constructive criticism of all analyses and findings.

INSTRUCTIONS:
- Question every assumption and methodology
- Suggest negative controls and validation experiments  
- Identify alternative explanations for findings
- Evaluate strength of evidence critically
- Always consider what could go wrong

Respond with scientific skepticism but constructive guidance.
"""
}

# Quick start functions
def quick_team_meeting(client):
    """Run the initial team meeting immediately"""
    
    team_prompt = """
VIRTUAL LAB TEAM MEETING

PARTICIPANTS:
- Dr. Sarah Chen (LD Reference Panel Specialist)
- Dr. Raj Patel (Advanced Colocalization Methodologist)  
- Dr. Lisa Wang (Fine-mapping Robustness Expert)
- Dr. Michael Torres (APOE Biology Specialist)
- Dr. Elena Rodriguez (Scientific Critic)

AGENDA: Project planning for APOE independent signals analysis

BACKGROUND:
- Analyzing APOE region (chr19:44-46Mb) for Alzheimer's disease
- Have GWAS summary statistics and fine-mapped molecular QTL data
- MAJOR CHALLENGES:
  1. APOE E4 signal too strong - overshadows other signals when conditioning
  2. LD reference panel mismatches create spurious independent signals
  3. Multiple xQTL colocalizations may be LD artifacts from E4 dominance

DISCUSSION POINTS:
1. What are the key methodological challenges we need to address?
2. What analysis strategies should we prioritize?
3. How do we validate any independent signals we find?
4. What are the most likely failure modes and how do we avoid them?

Please simulate a productive scientific discussion where each participant contributes their expertise. End with specific task assignments for each specialist.
"""
    
    print("🏢 Running team meeting...")
    result = run_team_meeting(client, team_prompt)
    
    if result:
        # Save result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"team_meeting_{timestamp}.txt", 'w') as f:
            f.write(result)
        print(f"✅ Team meeting saved to team_meeting_{timestamp}.txt")
        
    return result

def quick_agent_consultation(client, agent_name, task):
    """Consult a specific agent immediately"""
    
    if agent_name not in AGENT_PROMPTS:
        print(f"Agent {agent_name} not found. Available: {list(AGENT_PROMPTS.keys())}")
        return None
        
    print(f"💬 Consulting {agent_name}...")
    result = run_agent_meeting(client, AGENT_PROMPTS[agent_name], task)
    
    if result:
        # Save result  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"{agent_name}_{timestamp}.txt", 'w') as f:
            f.write(f"TASK: {task}\n\n{result}")
        print(f"✅ Consultation saved to {agent_name}_{timestamp}.txt")
        
    return result

# Main execution
if __name__ == "__main__":
    print("🧬 APOE Virtual Lab - Quick Setup")
    print("=" * 50)
    
    # Initialize
    client = setup_simple_virtual_lab()
    if not client:
        exit(1)
    
    # Run team meeting
    team_result = quick_team_meeting(client)
    
    if team_result:
        print("\n" + "="*50)
        print("TEAM MEETING RESULTS:")
        print("="*50)
        print(team_result[:1000] + "..." if len(team_result) > 1000 else team_result)
    
    # Example individual consultation
    print("\n" + "="*50)
    print("INDIVIDUAL CONSULTATION EXAMPLE:")
    print("="*50)
    
    ld_task = """
SPECIFIC PROBLEM: When I condition my APOE GWAS analysis on E4 variants (rs429358), I get spurious independent signals because my LD reference panel doesn't perfectly match my study population. The E4 effect is so strong that even small LD mismatches create false positives.

QUESTIONS:
1. How can I diagnose whether my LD reference panel is accurate enough?
2. What alternative conditioning strategies work better for dominant signals?  
3. Should I try population-stratified analysis (E4 carriers vs non-carriers)?
4. How do I validate that "independent" signals aren't just LD artifacts?

Please provide specific methodological recommendations with software/parameter suggestions.
"""
    
    ld_result = quick_agent_consultation(client, "ld_specialist", ld_task)
    
    if ld_result:
        print(ld_result[:1000] + "..." if len(ld_result) > 1000 else ld_result)
    
    print("\n🎉 Virtual Lab session complete!")
    print("Check the saved .txt files for full results")
    print("\nTo continue, call:")
    print("- quick_agent_consultation(client, 'colocalization_expert', your_task)")
    print("- quick_agent_consultation(client, 'finemap_expert', your_task)")
    print("- quick_agent_consultation(client, 'biology_expert', your_task)")
    print("- quick_agent_consultation(client, 'scientific_critic', your_task)")