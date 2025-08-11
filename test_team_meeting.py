# Full team discussions with all 6 agents and 3 rounds
import asyncio
from advanced_lab import setup_apoe_virtual_lab

async def test_team_meeting():
    # Initialize lab
    lab = setup_apoe_virtual_lab(api_provider="anthropic")
    
    print(f"🧬 Testing team meeting with {len(lab.agents)} agents")
    print(f"📁 Output directory: {lab.project_dir}")
    
    # Test full team meeting with all agents
    result = await lab.run_team_meeting(
        agenda="Integration of methodological approaches for APOE independent signal discovery with implementation planning",
        participants=["Dr. Sarah Chen", "Dr. Raj Patel", "Dr. Lisa Wang", "Dr. Michael Torres", "Dr. Elena Rodriguez", "Dr. Alex Cho"],
        rounds=3,  # 3-round discussion
        temperature=0.7
    )
    
    print("✅ Team meeting completed!")
    print(f"📁 Results saved in: {lab.project_dir}")
    return result

# Run the test
if __name__ == "__main__":
    print("🧪 Testing Team Meeting (All 6 Agents × 3 Rounds)")
    print("=" * 50)
    result = asyncio.run(test_team_meeting())