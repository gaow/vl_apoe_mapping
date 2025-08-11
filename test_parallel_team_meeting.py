from advanced_lab import Meeting

# Define meeting configuration
meeting_config = Meeting(
    meeting_type="team", 
    agenda="Synthesize APOE analysis approaches",
    participants=["Dr. Sarah Chen", "Dr. Raj Patel", "Dr. Elena Rodriguez"],
    rounds=3
)

# Run 3 parallel meetings, then synthesize (like the paper)
result = await lab.run_parallel_meetings(
    meeting_config,
    num_parallel=3,
    creative_temp=0.8,  # High creativity for parallel meetings
    merge_temp=0.2      # Low temperature for consistent synthesis
)