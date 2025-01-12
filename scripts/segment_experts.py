import os
import json
import random
from collections import defaultdict
from smart_open import smart_open

# URL to the dataset we're using.
dataset_url = "https://huggingface.co/datasets/jondurbin/airoboros-2.1/resolve/main/instructions.jsonl"

# Select the subset of data for each of our experts.
experts = {
  "qa": [
    "quiz",
    "multiple_choice",
    "contextual",
    "counterfactual_contextual"
  ],
  "creative": [
    "card",
    "writing",
    "experience",
    "song",
    "roleplay",
    "gtkm",
    "rp",
    "detailed_writing",
    "joke"
  ],
  "code": [
    "coding"
  ],
  "reasoning": [
    "cot",
    "theory_of_mind",
    "riddle",
    "orca"
  ],
  "function": [
    "agent",
    "plan"
  ],
  "general": [
    "wordgame",
    "trivia",
    "general"
  ]
}

# Map all of our training data into the categories per expert.
categories = defaultdict(list)
with smart_open(dataset_url, "r") as infile:
    for line in infile.readlines():
        item = json.loads(line)
        if not item.get("category"):
            continue
        categories[item["category"]].append(item)

# Include a random sampling of each expert's data in each other expert's dataset.
samples = {}
for expert, expert_cats in experts.items():
    samples[expert] = []
    for category in expert_cats:
        samples[expert] += random.sample(categories[category], int(len(categories[category]) * 0.15) or 1)

# Save the split datasets.
if not os.path.exists("training_data"):
    os.mkdir("training_data")
if not os.path.exists("routing_data"):
    os.mkdir("routing_data")
for expert, expert_cats in experts.items():
    with open(f"training_data/expert_{expert}.jsonl", "w") as outfile:
        # Also, be sure to include stylized responses so it adapts to system prompt well.
        for category in expert_cats + ["stylized_response"]:
            for item in categories[category]:
                outfile.write(json.dumps(item) + "\n")
        for other in samples:
            if other == expert:
                continue
            for item in samples[other]:
                outfile.write(json.dumps(item) + "\n")
    with open(f"routing_data/expert_{expert}.jsonl", "w") as outfile:
        for category in expert_cats:
            for item in categories[category]:
                outfile.write(json.dumps({"instruction": item.get("system", "A chat.") + " " + item["instruction"]}) + "\n")
