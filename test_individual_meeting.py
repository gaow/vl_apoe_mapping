# More sophisticated individual consultations
result = await lab.run_individual_meeting(
    agent_name="Dr. Sarah Chen",
    task="Your specific task",
    context="Additional context",
    rounds=2,  # Multiple rounds of refinement
    temperature=0.7
)