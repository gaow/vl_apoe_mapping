# Continue in Python interpreter or new script
import os
from simple_setup import *

# Initialize
client = setup_simple_virtual_lab()

# Consult colocalization expert
coloc_task = """
I have eQTL, pQTL, and sQTL data showing multiple colocalizations with my APOE GWAS signals. 
However, I suspect many are false positives due to LD artifacts from the dominant E4 effect.

My results show:
- 15+ molecular QTL signals with colocalization PP4 > 0.8
- Most are in high LD with APOE E4 variants  
- Effect sizes seem inflated compared to other regions

How do I distinguish true colocalization from LD echoes?
"""

result = quick_agent_consultation(client, "colocalization_expert", coloc_task)