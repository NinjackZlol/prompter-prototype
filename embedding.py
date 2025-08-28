from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json

with open("test-data.json", "r") as f:
    data = json.load(f)

texts = [entry["words"][0]["text"] for entry in data]

# prepare multi-label targets: pain, need, depth_low, depth_medium, depth_high
all_labels = []
for entry in data:
    labels = entry.get("labels", {})
    label_vector = [
        1 if labels.get("pain") else 0,
        1 if labels.get("need") else 0,
        1 if labels.get("depth")=="low" else 0,
        1 if labels.get("depth")=="medium" else 0,
        1 if labels.get("depth")=="high" else 0
    ]
    all_labels.append(label_vector)

# encode text to embedding
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)

assert len(embeddings) == len(all_labels), "Mismatch between embeddings and labels"

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, all_labels, test_size=0.25, random_state=42
)

# train classifier
clf = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

for i, idx in enumerate([i for i in range(len(X_test))]):
    print(f"Text: {texts[idx]}")
    print(f"Predicted: {y_pred[i]}")
    print(f"Ground Truth: {y_test[i]}")
    print("-"*50)

print(" Embedding Classifcation Report ")
print(classification_report(y_test, y_pred, zero_division=0))