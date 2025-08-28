from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import json

with open("test-data.json", "r") as f:
    data = json.load(f)

texts = [entry["words"][0]["text"] for entry in data]
labels = [entry["labels"]["pain"] for entry in data]

# encode text to embedding
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(" Embedding Classifcation Report ")
print(classification_report(y_test, y_pred, zero_division=0))