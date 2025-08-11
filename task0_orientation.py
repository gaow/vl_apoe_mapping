# Consult all agents on your core problem
from simple_setup import *

client = setup_simple_virtual_lab()

# Define your core problem
core_problem = """
I'm analyzing APOE region for independent AD signals beyond E2/E3/E4.
Main challenges:
1. E4 effect too strong - creates conditioning artifacts
2. LD reference panel mismatches  
3. Multiple xQTL colocalizations may be false positives

How do I address this in your area of expertise?
"""

# Consult each agent
agents = ["ld_specialist", "colocalization_expert", "finemap_expert", "biology_expert"]

for agent in agents:
    print(f"\n🔬 Consulting {agent}...")
    result = quick_agent_consultation(client, agent, core_problem)
    print(f"✅ {agent} consultation complete")