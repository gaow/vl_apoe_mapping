import asyncio
from advanced_lab import setup_apoe_virtual_lab

async def test_advanced():
    # Initialize with your API key
    lab = setup_apoe_virtual_lab(
        api_provider="anthropic",
        api_key="your-key-here"  # or use environment variable
    )
    
    print(f"Created {len(lab.agents)} agents")
    
    # Run one individual meeting
    result = await lab.run_individual_meeting(
        "Dr. Sarah Chen",
        "Diagnose LD reference panel accuracy for APOE conditioning",
        context="E4 effect creates spurious signals due to LD mismatches"
    )
    
    print("Meeting completed!")
    return result

# Run the test
result = asyncio.run(test_advanced())