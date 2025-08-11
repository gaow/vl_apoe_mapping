# Use advanced script for systematic research
import asyncio
from advanced_lab import setup_apoe_virtual_lab, Meeting

async def systematic_apoe_analysis():
    lab = setup_apoe_virtual_lab(api_provider="anthropic")
    
    # Phase 1: Individual expert consultations
    ld_result = await lab.run_individual_meeting(
        "Dr. Sarah Chen",
        "Develop comprehensive LD diagnostic framework for APOE conditioning",
        rounds=2  # Refine through iteration
    )
    
    coloc_result = await lab.run_individual_meeting(
        "Dr. Raj Patel", 
        "Design robust colocalization pipeline for multi-signal scenarios",
        rounds=2
    )
    
    # Phase 2: Team synthesis with parallel meetings
    synthesis_meeting = Meeting(
        meeting_type="team",
        agenda="Integrate methodological recommendations into unified APOE analysis pipeline",
        participants=["Dr. Sarah Chen", "Dr. Raj Patel", "Dr. Lisa Wang", "Dr. Elena Rodriguez"]
    )
    
    # Run parallel meetings for robustness (like Nature paper)
    final_result = await lab.run_parallel_meetings(
        synthesis_meeting,
        num_parallel=3
    )
    
    print("✅ Systematic analysis complete!")
    print(lab.get_meeting_summary())
    
    return final_result

# Run full analysis
result = asyncio.run(systematic_apoe_analysis())