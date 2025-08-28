from transformers import pipeline
import json

with open("test-data.json", "r") as f:
    dataset = json.load(f)

classifier = pipeline("zero-shot-classification")
candidate_labels = ["pain", "need"]

print("Zero-Shot Classification Results\n")

for entry in dataset:
    text = entry["words"][0]["text"]

    result = classifier(text, candidate_labels)

    print(f"Text: {text}")
    print("Predicted Labels:", result["labels"])
    print("Scores:", result["scores"])
    print("Ground Truth:", entry.get("labels", {}))
    print("-" * 50)
