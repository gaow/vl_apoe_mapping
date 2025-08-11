import asyncio
from advanced_lab import setup_apoe_virtual_lab

async def test_advanced():
    # Initialize with your API key (uses environment variable)
    lab = setup_apoe_virtual_lab(
        api_provider="anthropic"  # API key from environment variable
    )
    
    print(f"Created {len(lab.agents)} agents (including Dr. Alex Cho)")
    print(f"Output directory: {lab.project_dir}")
    
    # Test individual meeting with enhanced context
    result = await lab.run_individual_meeting(
        "Dr. Sarah Chen",
        "Diagnose LD reference panel accuracy for APOE conditioning with ~300 candidate genes",
        context="E2/E3/E4 effects create spurious signals due to LD mismatches; need efficient prioritization strategies for large-scale analysis",
        rounds=1
    )
    
    # Test bioinformatics engineer consultation
    impl_result = await lab.run_individual_meeting(
        "Dr. Alex Cho",
        "Create implementation plan for LD diagnostic workflow",
        context="Based on Dr. Chen's recommendations, design R/bash scripts for LD analysis pipeline",
        rounds=1
    )
    
    print("✅ Test meetings completed!")
    print(f"📁 Results saved in: {lab.project_dir}")
    return result

# Run the test
if __name__ == "__main__":
    print("🧪 Testing Advanced Virtual Lab")
    print("=" * 40)
    result = asyncio.run(test_advanced())