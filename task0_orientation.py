# Consult all agents on your core problem
from simple_setup import *

result = setup_simple_virtual_lab()
if not result:
    exit(1)

client, output_dir = result

# Define your core problem - updated with more context
core_problem = """
I'm analyzing APOE region (chr19:44-46Mb) for independent AD signals beyond E2/E3/E4.

DATASETS:
- GWAS summary statistics (~500k samples)
- Fine-mapped molecular QTL data (eQTL, pQTL, sQTL, trans effects)
- Multiple LD reference panels (1000G, UKB, TOPMed)
- ~300 candidate genes in the region

MAIN CHALLENGES:
1. E2/E3/E4 effects too strong, especially E4 - creates conditioning artifacts
2. LD reference panel mismatches with study population
3. Multiple xQTL colocalizations may be LD artifacts from E4 dominance
4. Need to find new genes beyond APOE that variants regulate
5. Scale: analyzing ~300 genes requires prioritization strategies

QUESTION: How do I address these challenges in your area of expertise?
Provide specific methodological recommendations and implementation approaches.
"""

# Consult all agents including the new bioinformatics engineer
agents = ["ld_specialist", "colocalization_expert", "finemap_expert", "biology_expert", "bioinformatics_engineer", "scientific_critic"]

for agent in agents:
    print(f"\n🔬 Consulting {agent}...")
    result = quick_agent_consultation(client, agent, core_problem, output_dir)
    print(f"✅ {agent} consultation complete")

print(f"\n🎉 All consultations complete! Check results in {output_dir}")