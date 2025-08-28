from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import json

with open("test-data.json", "r") as f:
    data = json.load(f)

# Prepare multi-label targets: pain, need, depth_low, depth_medium, depth_high
texts = [entry['words'][0]['text'] for entry in data]
labels = []

for entry in data:
    lbl = [0]*5
    if entry['labels'].get('pain'):
        lbl[0] = 1
    if entry['labels'].get('need'):
        lbl[1] = 1
    depth = entry['labels'].get('depth')
    if depth == 'low':
        lbl[2] = 1
    elif depth == 'medium':
        lbl[3] = 1
    elif depth == 'high':
        lbl[4] = 1
    labels.append(lbl)

# split dataset before tokenization
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.25, random_state=42
)

# tokenize
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

# dataset object
class TranscriptDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

# dataset instantiation
train_dataset = TranscriptDataset(train_encodings, train_labels)
test_dataset = TranscriptDataset(test_encodings, test_labels)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=5)

# trainig args
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_steps=5
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

predictions = trainer.predict(test_dataset)
pred_labels = (predictions.predictions > 0.5).astype(int)  # Threshold for multi-label
print("Predicted labels:\n", pred_labels)
print("Ground truth labels:\n", test_labels)
