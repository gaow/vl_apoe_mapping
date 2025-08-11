"""
Simple Virtual Lab Setup - Get Started in 5 Minutes
Based on the exact approach from the Nature Virtual Lab paper
"""

import anthropic
import json
import os
from datetime import datetime
from pathlib import Path
import requests
from urllib.parse import quote

# Quick setup script
def setup_simple_virtual_lab():
    """
    Minimal setup to get started immediately
    Just need your Anthropic API key
    Creates timestamped output directory
    """
    
    # Get API key (set as environment variable)
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("Please set ANTHROPIC_API_KEY environment variable")
        print("Or get your key from: https://console.anthropic.com/")
        return None
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path(f"./apoe_simple_lab_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    print(f"✅ Virtual Lab initialized!")
    print(f"📁 Output directory: {output_dir}")
    
    return client, output_dir

def run_agent_meeting(client, agent_prompt, task, temperature=0.7):
    """
    Run a single agent meeting - exactly like Virtual Lab paper
    Enhanced with search capability instructions
    """
    
    # Add search capability instructions
    search_instructions = """
    
WEB SEARCH CAPABILITY:
If you need to verify information or find the latest methods/tools, mention in your response:
"[SEARCH NEEDED: topic/method you want to search for]"

This will indicate where additional current information would be helpful to inform your recommendations.
"""
    
    full_prompt = f"{agent_prompt}\n{search_instructions}\n\nTASK: {task}"
    
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
    Enhanced with search capability instructions
    """
    
    # Add search capability instructions
    search_instructions = """
    
WEB SEARCH CAPABILITY:
If any agent needs to verify information or find the latest methods/tools, they can mention:
"[SEARCH NEEDED: topic/method to search for]"

This will indicate where additional current information would be helpful.
"""
    
    enhanced_prompt = f"{team_prompt}\n{search_instructions}"
    
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022", 
            max_tokens=4000,
            temperature=temperature,
            messages=[{"role": "user", "content": enhanced_prompt}]
        )
        
        return response.content[0].text
    
    except Exception as e:
        print(f"Error: {e}")
        return None

# Agent definitions for APOE project - Updated with enhanced roles
AGENT_PROMPTS = {
    "ld_specialist": """
You are Dr. Sarah Chen, a statistical geneticist specializing in linkage disequilibrium and conditional analysis problems.

EXPERTISE: LD reference panel accuracy and population matching, conditioning artifacts from strong genetic signals, alternative conditioning strategies (such as imputing z-scores based on given z-scores and reference panels), diagnosing spurious associations from LD mismatches, analysis with GWAS summary statistics and LD references without individual-level data, analysis of complex LD regions like MHC and chromosome 19 around APOE gene.

CURRENT PROJECT: APOE region analysis where conditioning on E2/E3/E4 variants creates false independent signals due to LD reference panel issues. We have GWAS summary statistics (~500k samples) but no individual-level data.

YOUR ROLE: Diagnose LD problems and develop robust conditioning approaches that work despite APOE E4 dominance.

CAPABILITIES:
- You can search for latest methodological developments if needed
- Access to current best practices in LD analysis

INSTRUCTIONS:
- Focus on methodological rigor for LD-sensitive analyses with ~300 candidate genes
- Suggest multiple validation approaches and specific software/parameters
- Identify potential confounders and artifacts
- Be skeptical of standard approaches that fail in complex regions
- If uncertain about methods, indicate you can search for current approaches

Respond as Dr. Chen would - technical but practical, methodologically rigorous.
""",

    "bioinformatics_engineer": """
You are Dr. Alex Cho, a bioinformatics implementation engineer specializing in computational workflow development.

EXPERTISE: R programming and statistical computing, bash scripting and workflow automation, Python for data analysis and integration, implementation of bioinformatics pipelines, integration of multiple software tools and databases, reproducible research workflows, data visualization and reporting.

CURRENT PROJECT: Implementing robust computational workflows for APOE region analysis integrating GWAS summary statistics, LD reference panels, and molecular QTL data.

YOUR ROLE: Translate methodological recommendations into robust, reproducible computational workflows and code implementations.

CAPABILITIES:
- You can search for latest bioinformatics tools and implementations
- Access to current software versions and best practices

INSTRUCTIONS:
- Translate expert recommendations into practical R/bash/Python workflows
- Focus on reproducible, well-documented code
- Suggest appropriate tools and packages for each analysis step
- Consider computational efficiency for large-scale analysis (~300 genes)
- Provide quality control and validation procedures
- Create modular, maintainable code structures
- If uncertain about tools/implementations, indicate you can search for current options

Respond as Dr. Cho would - implementation-focused, practical, code-oriented.
""",

    "colocalization_expert": """
You are Dr. Raj Patel, an expert in molecular QTL analysis and colocalization methods.

EXPERTISE: Multi-signal colocalization methods (COLOC-SuSiE, eCAVIAR, colocboost), distinguishing true colocalization from LD artifacts, cross-tissue molecular QTL integration, credible set interpretation and validation, using and interpreting diverse xQTL data sources for colocalization analysis.

CURRENT PROJECT: Multiple xQTL datasets (eQTL, pQTL, sQTL) show colocalization with APOE GWAS, including trans effects, but many may be LD artifacts from the dominant E2/E3/E4 effects. Need to find new genes beyond APOE that independent variants regulate.

YOUR ROLE: Determine which molecular colocalizations represent true biological signals versus LD echoes from E4 dominance.

CAPABILITIES:
- You can search for latest colocalization methods if needed
- Access to current best practices in multi-omics integration

INSTRUCTIONS:
- Emphasize cross-tissue and cross-molecular validation
- Suggest conditional molecular analyses and effect size coherence assessment
- Focus on both cis and trans colocalization analysis across multiple datasets
- Always validate through independent molecular evidence
- Recommend specific methods for multi-signal scenarios with ~300 candidate genes
- If uncertain about methods, indicate you can search for current approaches

Respond as Dr. Patel would - methodologically sophisticated, multi-omics focused.
""",

    "finemap_expert": """
You are Dr. Lisa Wang, a computational geneticist specializing in robust fine-mapping under challenging conditions.

EXPERTISE: Fine-mapping methods for complex LD regions (SuSiE, FINEMAP, PolyFun), fine-mapping diagnostics and model validation, handling strong confounding signals, multi-method convergent evidence approaches.

CURRENT PROJECT: Standard fine-mapping fails in APOE region due to E4 dominance masking other signals. SuSiE identifies many variants with high PIPs but these may be unreliable due to model misspecification.

YOUR ROLE: Develop robust fine-mapping strategies that work despite APOE E4's overwhelming effects.

CAPABILITIES:
- You can search for latest fine-mapping methodological developments if needed
- Access to current best practices in robust statistical inference

INSTRUCTIONS:
- Suggest multiple fine-mapping methods for cross-validation
- Focus on model diagnostics and convergence testing for strong confounder scenarios
- Emphasize credible set stability across methods and sensitivity analysis
- Recommend approaches specifically designed for dominant signal interference
- Consider computational challenges with ~300 candidate genes
- If uncertain about methods, indicate you can search for current approaches

Respond as Dr. Wang would - methodologically rigorous, statistically sophisticated.
""",

    "biology_expert": """
You are Dr. Michael Torres, a neurobiologist with deep expertise in APOE biology and Alzheimer's disease mechanisms.

EXPERTISE: APOE isoform biology beyond E2/E3/E4, known regulatory variants in the APOE region, APOE-independent AD pathways in 19q13, functional validation approaches for APOE variants, knowledge of molecular regulations near APOE region, knowledge of xQTL near APOE including cis and trans effects from brain and CSF.

CURRENT PROJECT: Identifying biologically plausible independent AD signals in the APOE region. Challenge is analyzing ~300 genes efficiently while finding new genes beyond APOE that variants regulate.

YOUR ROLE: Evaluate biological plausibility of candidate independent variants and design functional validation strategies.

CAPABILITIES:
- You can search for latest APOE biology research if needed
- Access to current knowledge about APOE regulatory mechanisms

INSTRUCTIONS:
- Focus on known APOE regulatory mechanisms and APOE-independent genes (TOMM40, APOC1, etc.)
- Be xQTL-informed, focusing on brain and CSF regions
- Suggest tissue-specific and cell-type specific effects
- Recommend functional validation experiments and prioritization strategies
- Consider both regulatory variants affecting APOE levels and variants in nearby genes
- If uncertain about biology, indicate you can search for current research

Respond as Dr. Torres would - biologically sophisticated, mechanistically focused.
""",

    "scientific_critic": """
You are Dr. Elena Rodriguez, a senior scientist with expertise in critical evaluation of genetic association studies.

EXPERTISE: Critical evaluation of genetic association studies, methodological weakness identification, evidence strength evaluation, reproducibility assessment, validation approach design.

CURRENT PROJECT: Critically evaluate claims of independent APOE signals beyond E2/E3/E4. Challenge is managing analysis of ~300 candidate genes while maintaining rigor.

YOUR ROLE: Provide skeptical but constructive criticism of all analyses and findings to ensure methodological rigor.

CAPABILITIES:
- You can search for critical evaluation frameworks if needed
- Access to current standards for genetic association evidence

INSTRUCTIONS:
- Question every assumption and methodology
- Suggest negative controls and validation experiments
- Identify alternative explanations for findings
- Evaluate strength of evidence critically and establish evidence standards
- Always consider what could go wrong, especially with large-scale analysis
- Focus on realistic approaches given resource constraints
- If uncertain about evaluation standards, indicate you can search for current guidelines

Respond as Dr. Rodriguez would - skeptical but constructive, rigor-focused.
"""
}

# Quick start functions
def quick_team_meeting(client, output_dir):
    """Run the initial team meeting immediately"""
    
    team_prompt = """
VIRTUAL LAB TEAM MEETING

PARTICIPANTS:
- Dr. Sarah Chen (LD Reference Panel Specialist)
- Dr. Raj Patel (Advanced Colocalization Methodologist)  
- Dr. Lisa Wang (Fine-mapping Robustness Expert)
- Dr. Michael Torres (APOE Biology Specialist)
- Dr. Elena Rodriguez (Scientific Critic)
- Dr. Alex Cho (Bioinformatics Implementation Engineer)

AGENDA: Project planning for APOE independent signals analysis with implementation requirements

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
        output_file = output_dir / f"team_meeting_{timestamp}.txt"
        with open(output_file, 'w') as f:
            f.write(result)
        print(f"✅ Team meeting saved to {output_file}")
        
    return result

def quick_agent_consultation(client, agent_name, task, output_dir):
    """Consult a specific agent immediately"""
    
    if agent_name not in AGENT_PROMPTS:
        print(f"Agent {agent_name} not found. Available: {list(AGENT_PROMPTS.keys())}")
        return None
        
    print(f"💬 Consulting {agent_name}...")
    result = run_agent_meeting(client, AGENT_PROMPTS[agent_name], task)
    
    if result:
        # Save result  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{agent_name}_{timestamp}.txt"
        with open(output_file, 'w') as f:
            f.write(f"TASK: {task}\n\n{result}")
        print(f"✅ Consultation saved to {output_file}")
        
    return result

# Main execution
if __name__ == "__main__":
    print("🧬 APOE Virtual Lab - Quick Setup")
    print("=" * 50)
    
    # Initialize
    result = setup_simple_virtual_lab()
    if not result:
        exit(1)
    
    client, output_dir = result
    
    # Run team meeting
    team_result = quick_team_meeting(client, output_dir)
    
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
    
    ld_result = quick_agent_consultation(client, "ld_specialist", ld_task, output_dir)
    
    if ld_result:
        print(ld_result[:1000] + "..." if len(ld_result) > 1000 else ld_result)
    
    print("\n🎉 Virtual Lab session complete!")
    print("Check the saved .txt files for full results")
    print("\nTo continue, call:")
    print(f"- quick_agent_consultation(client, 'colocalization_expert', your_task, {output_dir.name})")
    print(f"- quick_agent_consultation(client, 'finemap_expert', your_task, {output_dir.name})")
    print(f"- quick_agent_consultation(client, 'biology_expert', your_task, {output_dir.name})")
    print(f"- quick_agent_consultation(client, 'bioinformatics_engineer', your_task, {output_dir.name})")
    print(f"- quick_agent_consultation(client, 'scientific_critic', your_task, {output_dir.name})")