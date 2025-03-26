import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import json
import yaml
import sys
import re
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from datetime import datetime

# Add root directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    BERT_MODEL,
    INTENT_MODEL_PATH,
    INTENT_LABELS_PATH,
    NUM_INTENTS,
    INTENT_LABELS,
    MODEL_CONFIG,
    NLU_DATA_PATH
)

# Create directories if they don't exist
os.makedirs(os.path.dirname(INTENT_MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(NLU_DATA_PATH), exist_ok=True)

# Define intent mapping
intent_mapping = {
    label: idx for idx, label in enumerate(INTENT_LABELS)
}

# Save intent mapping to JSON
with open(INTENT_LABELS_PATH, 'w', encoding='utf-8') as f:
    json.dump(intent_mapping, f)

# Load training data from YAML
def load_training_data():
    with open(NLU_DATA_PATH, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    texts = []
    labels = []
    
    for intent, intent_data in data['intents'].items():
        for example in intent_data['examples']:
            texts.append(example)
            labels.append(intent_mapping[intent])
    
    return texts, labels

# Add more variations for robust training
def generate_variations(texts, labels, num_variations=2):
    """
    Generate variations of training examples for data augmentation.
    
    Args:
        texts: List of text examples
        labels: List of corresponding labels
        num_variations: Number of variations to generate per example
        
    Returns:
        Tuple of (new_texts, new_labels) with augmented data
    """
    variations_texts = []
    variations_labels = []
    
    for text, label in zip(texts, labels):
        # Ensure text is a string
        text = str(text)
        
        # Add the original example
        variations_texts.append(text)
        variations_labels.append(label)
        
        intent = INTENT_LABELS[label]
        
        if intent == "greeting":
            variations_texts.extend([
                f"{text}! How are you doing?",
                f"{text}, nice to meet you"
            ])
            variations_labels.extend([label, label])
        elif intent == "provide_name":
            name_parts = text.split()
            name = name_parts[-1] if len(name_parts) > 1 else text
            variations_texts.extend([
                f"I would like to be called {name}",
                f"Please call me {name}"
            ])
            variations_labels.extend([label, label])
        elif intent == "provide_gender":
            gender_parts = text.split()
            gender = gender_parts[-1] if len(gender_parts) > 1 else text
            variations_texts.extend([
                f"I identify as {gender}",
                f"My gender identification is {gender}"
            ])
            variations_labels.extend([label, label])
        elif intent == "provide_age":
            age_numbers = re.findall(r'\d+', text)
            if age_numbers:
                age = age_numbers[0]
                variations_texts.extend([
                    f"I'm {age} years of age",
                    f"{age} is my age"
                ])
                variations_labels.extend([label, label])
        elif intent == "provide_location":
            location_parts = text.split()
            location = location_parts[-1] if len(location_parts) > 1 else text
            variations_texts.extend([
                f"I currently live in {location}",
                f"My residence is in {location}"
            ])
            variations_labels.extend([label, label])
        elif intent == "goodbye":
            variations_texts.extend([
                f"{text}, thank you for your help",
                f"{text}, have a nice day"
            ])
            variations_labels.extend([label, label])
        elif intent == "out_of_scope":
            variations_texts.extend([
                f"Can you tell me {text}?",
                f"I want to know about {text}"
            ])
            variations_labels.extend([label, label])
    
    return variations_texts, variations_labels

# Load and prepare training data
print("Loading training data...")
texts, labels = load_training_data()
print(f"Loaded {len(texts)} examples")

# Check class balance
class_counts = {}
for label in labels:
    intent = INTENT_LABELS[label]
    class_counts[intent] = class_counts.get(intent, 0) + 1

print("Class distribution before augmentation:")
for intent, count in class_counts.items():
    print(f"  {intent}: {count} examples")

print("\nGenerating variations for training data...")
augmented_texts, augmented_labels = generate_variations(texts, labels)
print(f"Generated {len(augmented_texts)} examples after augmentation")

# Check class balance after augmentation
class_counts = {}
for label in augmented_labels:
    intent = INTENT_LABELS[label]
    class_counts[intent] = class_counts.get(intent, 0) + 1

print("Class distribution after augmentation:")
for intent, count in class_counts.items():
    print(f"  {intent}: {count} examples")

# Split data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    augmented_texts, augmented_labels, test_size=0.2, random_state=42, stratify=augmented_labels
)

print(f"\nSplit data into {len(train_texts)} training and {len(val_texts)} validation examples")

# Create dataset class
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.texts)

# Initialize tokenizer
print("\nInitializing tokenizer...")
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

# Create datasets
train_dataset = IntentDataset(train_texts, train_labels, tokenizer, max_length=MODEL_CONFIG['max_length'])
val_dataset = IntentDataset(val_texts, val_labels, tokenizer, max_length=MODEL_CONFIG['max_length'])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=MODEL_CONFIG['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=MODEL_CONFIG['batch_size'])

# Initialize model
print("Initializing model...")
model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=len(INTENT_LABELS))

# Move model to available device
device = torch.device(MODEL_CONFIG["device"])
print(f"Using device: {device}")
model.to(device)

# Initialize optimizer
optimizer = AdamW(model.parameters(), lr=MODEL_CONFIG['learning_rate'], weight_decay=MODEL_CONFIG.get('weight_decay', 0.01))

# Learning rate scheduler with warmup
total_steps = len(train_loader) * MODEL_CONFIG['num_epochs']
warmup_steps = int(total_steps * 0.1)  # 10% of total steps for warmup
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# For tracking training metrics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training loop
print(f"\nStarting training for {MODEL_CONFIG['num_epochs']} epochs...")
model.train()
best_val_loss = float('inf')
best_val_accuracy = 0
patience = 3
patience_counter = 0
start_time = datetime.now()

for epoch in range(MODEL_CONFIG['num_epochs']):
    # Training phase
    model.train()
    total_train_loss = 0
    correct_train_preds = 0
    total_train_samples = 0
    
    train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{MODEL_CONFIG['num_epochs']} [Train]")
    
    for batch in train_progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_train_loss += loss.item() * input_ids.size(0)
        
        # Calculate training accuracy
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        correct_train_preds += (predictions == labels).sum().item()
        total_train_samples += input_ids.size(0)
        
        # Update progress bar
        train_progress_bar.set_postfix({
            'loss': loss.item(),
            'acc': (predictions == labels).sum().item() / input_ids.size(0)
        })
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        scheduler.step()
    
    avg_train_loss = total_train_loss / total_train_samples
    train_accuracy = correct_train_preds / total_train_samples
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)
    
    # Validation phase
    model.eval()
    total_val_loss = 0
    correct_val_preds = 0
    total_val_samples = 0
    
    with torch.no_grad():
        val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{MODEL_CONFIG['num_epochs']} [Validation]")
        
        for batch in val_progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_val_loss += loss.item() * input_ids.size(0)
            
            # Calculate validation accuracy
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            correct_val_preds += (predictions == labels).sum().item()
            total_val_samples += input_ids.size(0)
            
            # Update progress bar
            val_progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': (predictions == labels).sum().item() / input_ids.size(0)
            })
    
    avg_val_loss = total_val_loss / total_val_samples
    val_accuracy = correct_val_preds / total_val_samples
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)
    
    # Print epoch results
    print(f"Epoch {epoch+1}/{MODEL_CONFIG['num_epochs']}")
    print(f"  Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"  Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    # Check if this is the best model so far
    if val_accuracy > best_val_accuracy:
        print(f"  Validation accuracy improved from {best_val_accuracy:.4f} to {val_accuracy:.4f}")
        best_val_accuracy = val_accuracy
        best_val_loss = avg_val_loss
        
        # Save the model
        torch.save(model.state_dict(), INTENT_MODEL_PATH)
        print(f"  Saved new best model to {INTENT_MODEL_PATH}")
        
        # Reset patience counter
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"  Validation accuracy did not improve. Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"  Early stopping after {epoch+1} epochs")
            break

# Calculate training time
end_time = datetime.now()
training_time = end_time - start_time
print(f"\nTraining completed in {training_time}")
print(f"Best validation accuracy: {best_val_accuracy:.4f}")
print(f"Best validation loss: {best_val_loss:.4f}")

# Generate and save learning curves
plt.figure(figsize=(12, 5))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Accuracy curve
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()

# Create plots directory if it doesn't exist
plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots")
os.makedirs(plots_dir, exist_ok=True)

# Create a more human-readable timestamp format
readable_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
filename = f"learning_curves_{readable_timestamp}.png"

# Save the file with the new timestamp format
plt.savefig(os.path.join(plots_dir, filename))

print(f"Learning curves saved to plots directory as '{filename}'")

# Evaluate the model on the validation set
print("\nDetailed validation metrics:")
model.load_state_dict(torch.load(INTENT_MODEL_PATH))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        _, predictions = torch.max(outputs.logits, dim=1)
        
        all_preds.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

# Calculate per-class metrics
class_correct = {i: 0 for i in range(len(INTENT_LABELS))}
class_total = {i: 0 for i in range(len(INTENT_LABELS))}

for pred, label in zip(all_preds, all_labels):
    if pred == label:
        class_correct[label] += 1
    class_total[label] += 1

print("\nPer-class accuracy:")
for i in range(len(INTENT_LABELS)):
    accuracy = class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    print(f"  {INTENT_LABELS[i]}: {accuracy:.4f} ({class_correct[i]}/{class_total[i]})")

print("\nTraining completed successfully!")
