from transformers import pipeline
import json
from sklearn.metrics import classification_report

with open("test-data.json", "r") as f:
    dataset = json.load(f)

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
candidate_labels = ["pain", "need", "depth_low", "depth_medium", "depth_high"]

y_true = []
y_pred = []

print("Zero-Shot Classification Results\n")

for entry in dataset:
    text = entry["words"][0]["text"]

    result = classifier(text, candidate_labels)

    best_label = result["labels"][0]
    best_score = result["scores"][0]

    # map labels for ground truth flatten: pain vs need vs depth
    gt = []
    if "labels" in entry:
        if entry["labels"].get("pain"):
            gt.append("pain")
        if entry["labels"].get("need"):
            gt.append("need")
        if entry["labels"].get("depth"):
            gt.append(f"depth_{entry['labels']['depth']}")

    print(f"Text: {text}")
    print(f"Predicted: {best_label} (score={best_score:.2f})")
    print(f"Ground Truth: {gt}")
    print("-" * 60)

    if gt:
        y_true.append(gt[0])
        y_pred.append(best_label)

print(" Classification Report ")
print(classification_report(y_true, y_pred, zero_division=0))