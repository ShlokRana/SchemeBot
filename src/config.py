import torch
import os
import yaml
import json
import pickle
import numpy as np
from pathlib import Path

# Get the root directory path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Create necessary directories
os.makedirs(os.path.join(ROOT_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, "data"), exist_ok=True)

# Model paths
MODEL_DIR = os.path.join(ROOT_DIR, "models")
INTENT_MODEL_PATH = os.path.join(MODEL_DIR, "intent_model.pth")
INTENT_LABELS_PATH = os.path.join(MODEL_DIR, "intent_labels.json")
ENTITY_RECOGNITION_MODEL_PATH = os.path.join(MODEL_DIR, "entity_recognition")

# Training data paths
TRAINING_DATA_DIR = os.path.join(ROOT_DIR, "data")
NLU_DATA_PATH = os.path.join(TRAINING_DATA_DIR, "nlu_data.yml")

# BERT model configuration
BERT_MODEL = "bert-base-uncased"
NUM_INTENTS = 7  # Number of intents in the intent classification model

# User input validation
MAX_ATTEMPTS = 3  # Maximum attempts for user to provide valid input
MIN_AGE = 0  # Minimum age allowed
MAX_AGE = 120  # Maximum age allowed

# Chatbot configuration
GREETING_RESPONSES = [
    "Hello! I'm SchemeBot. How can I help you today?",
    "Hi there! I'm here to help you find government schemes you may be eligible for.",
    "Welcome! I'm SchemeBot, your assistant for finding relevant government schemes."
]

# Model training configuration
TRAIN_BATCH_SIZE = 16  # Reduced batch size for better learning
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 1e-4  # Reduced learning rate
NUM_EPOCHS = 5  # Increased epochs
MAX_LENGTH = 128

# Confidence thresholds
INTENT_CONFIDENCE_THRESHOLD = 0.7
ENTITY_CONFIDENCE_THRESHOLD = 0.6

# Intent labels
INTENT_LABELS = [
    "greeting",
    "provide_name",
    "provide_gender",
    "provide_age",
    "provide_location",
    "goodbye",
    "out_of_scope"
]

# Entity types
ENTITY_TYPES = ["NAME", "AGE", "GENDER", "LOCATION"]

# Model configuration
MODEL_CONFIG = {
    "model_name": "bert-base-uncased",
    "max_length": 128,
    "num_epochs": 10,  # Increased epochs
    "batch_size": 8,  # Reduced batch size
    "learning_rate": 2e-5,  # Reduced learning rate
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "device": "mps" if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu"
} 