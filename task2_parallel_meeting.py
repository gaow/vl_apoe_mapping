# Use advanced script for systematic research with enhanced parallel meetings
import asyncio
from advanced_lab import setup_apoe_virtual_lab, Meeting

async def systematic_apoe_analysis():
    """Run comprehensive APOE analysis with all 6 agents"""
    
    lab = setup_apoe_virtual_lab(api_provider="anthropic")
    print(f"🧬 Initialized Virtual Lab with {len(lab.agents)} agents")
    print(f"📁 Output directory: {lab.project_dir}")
    
    # Phase 1: Individual expert consultations (refined)
    print("\n=== PHASE 1: INDIVIDUAL EXPERT CONSULTATIONS ===")
    
    ld_result = await lab.run_individual_meeting(
        "Dr. Sarah Chen",
        "Develop comprehensive LD diagnostic framework and robust conditioning strategies for APOE region with ~300 candidate genes",
        context="GWAS summary stats (~500k samples), multiple LD reference panels, need to handle E4 dominance and prioritize efficiently",
        rounds=2
    )
    
    coloc_result = await lab.run_individual_meeting(
        "Dr. Raj Patel", 
        "Design robust multi-signal colocalization pipeline for diverse xQTL data integration",
        context="eQTL, pQTL, sQTL, trans effects; need to distinguish true colocalizations from LD artifacts; find new genes beyond APOE",
        rounds=2
    )
    
    finemap_result = await lab.run_individual_meeting(
        "Dr. Lisa Wang",
        "Develop multi-method fine-mapping approach robust to E4 dominance and model misspecification", 
        context="SuSiE identifies many high-PIP variants but unreliable; need validation strategies for strong confounder scenarios",
        rounds=2
    )
    
    biology_result = await lab.run_individual_meeting(
        "Dr. Michael Torres",
        "Create biological prioritization framework for ~300 candidate genes and validation strategies",
        context="Need to prioritize variants/genes efficiently while not missing APOE-independent mechanisms; focus on brain/CSF xQTL",
        rounds=2
    )
    
    implementation_result = await lab.run_individual_meeting(
        "Dr. Alex Cho",
        "Design comprehensive computational workflow integrating all methodological recommendations",
        context="Need R/bash/Python pipelines for LD analysis, colocalization, fine-mapping, and biological prioritization at scale",
        rounds=2
    )
    
    # Phase 2: Enhanced parallel team synthesis with ALL 6 agents and 3 rounds each
    print("\n=== PHASE 2: PARALLEL TEAM SYNTHESIS (3 sessions × 3 rounds each) ===")
    
    synthesis_meeting = Meeting(
        meeting_type="team",
        agenda="Synthesize all expert recommendations into integrated, implementable APOE analysis pipeline with realistic prioritization for ~300 candidate genes",
        participants=["Dr. Sarah Chen", "Dr. Raj Patel", "Dr. Lisa Wang", "Dr. Michael Torres", "Dr. Elena Rodriguez", "Dr. Alex Cho"],
        rounds=3  # Each parallel meeting will have 3 rounds
    )
    
    # Run 3 parallel meetings, each with 3 rounds and all 6 agents
    final_result = await lab.run_parallel_meetings(
        synthesis_meeting,
        num_parallel=3,  # 3 parallel sessions
        creative_temp=0.8,
        merge_temp=0.2,
        all_agents=True  # Use all 6 agents
    )
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("✅ Systematic analysis complete!")
    print(f"📊 {lab.get_meeting_summary()}")
    print(f"📁 All results saved in: {lab.project_dir}")
    print(f"🔄 Parallel synthesis: 3 sessions × 3 rounds × 6 agents = robust methodology")
    
    return final_result

# Run full analysis
if __name__ == "__main__":
    print("🧬 APOE Virtual Lab - Advanced Parallel Analysis")
    print("=" * 60)
    result = asyncio.run(systematic_apoe_analysis())