import os
import re
import pickle
import yaml
import spacy
import random
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.cli.train import train
import sys
from spacy.util import minibatch, compounding
import unicodedata

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ENTITY_RECOGNITION_MODEL_PATH
from src.entity_patterns import (
    NAME_PATTERNS, AGE_PATTERNS, GENDER_PATTERNS, LOCATION_PATTERNS,
    INDIAN_STATES, STATE_ALIASES, CITY_ALIASES, INDIAN_CITIES,
    INDIAN_NAMES, TRAINING_EXAMPLES
)
import json

# Define entity types
ENTITY_TYPES = ["NAME", "AGE", "GENDER", "LOCATION"]

# Define paths for training data
ENTITY_TRAINING_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "entity_training_data.txt")
MODEL_OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "entity_recognition")

# Make sure directories exist
os.makedirs(os.path.dirname(ENTITY_TRAINING_DATA_PATH), exist_ok=True)
os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)

def normalize_text(text):
    """Normalize text by removing accents and converting to lowercase."""
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    return text.lower().strip()

def extract_entities(text):
    """
    Extract entities from user input.
    Returns a dictionary with entity types as keys and extracted values as values.
    """
    entities = {}
    
    # Handle single-word inputs better
    text = text.strip()
    text_lower = text.lower()
    
    # Initialize spaCy model
    try:
        nlp = spacy.load(ENTITY_RECOGNITION_MODEL_PATH)
    except Exception as e:
        print(f"Warning: Could not load spaCy model: {e}")
        # Fallback to blank model
        nlp = spacy.blank("en")
    
    # Define common greetings that should not be treated as names
    common_greetings = ['hi', 'hello', 'hey', 'yo', 'sup', 'hola', 'greetings', 'howdy', 'namaste', 'bye', 'goodbye']
    
    # Age extraction - handle direct numeric input
    if text.isdigit():
        age = int(text)
        if 0 <= age <= 120:  # Reasonable age range
            entities['age'] = age
            return entities
    
    # Gender extraction - handle direct gender inputs and patterns first
    gender_map = {
        'male': 'male', 'm': 'male', 'man': 'male', 'boy': 'male', 'guy': 'male', 'gentleman': 'male',
        'female': 'female', 'f': 'female', 'woman': 'female', 'girl': 'female', 'lady': 'female'
    }
    
    # Check for single word gender
    if text_lower in gender_map:
        entities['gender'] = gender_map[text_lower]
        return entities
    
    # Check for gender patterns before name patterns
    gender_patterns = [
        r"(?:I am|I'm|I identify as)\s*(?:a|an)?\s*(male|female|man|woman|boy|girl|guy)",
        r"gender\s*(?:is|:)?\s*(male|female|man|woman|boy|girl|guy)"
    ]
    
    for pattern in gender_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            gender_term = match.group(1).lower()
            if gender_term in gender_map:
                entities['gender'] = gender_map[gender_term]
                # If the text is just indicating gender, return immediately
                if re.match(f"(?:I am|I'm|I identify as)\\s*(?:a|an)?\\s*{gender_term}\\s*$", text, re.IGNORECASE):
                    return entities
                break
    
    # Location extraction - handle direct location inputs for Indian states
    indian_states = [
        'andhra pradesh', 'arunachal pradesh', 'assam', 'bihar', 'chhattisgarh', 'goa', 
        'gujarat', 'haryana', 'himachal pradesh', 'jharkhand', 'karnataka', 'kerala', 
        'madhya pradesh', 'maharashtra', 'manipur', 'meghalaya', 'mizoram', 'nagaland', 
        'odisha', 'punjab', 'rajasthan', 'sikkim', 'tamil nadu', 'telangana', 'tripura', 
        'uttar pradesh', 'uttarakhand', 'west bengal', 'delhi', 'chandigarh'
    ]
    
    if text_lower in indian_states:
        entities['location'] = text.title()
        return entities
    
    # Check if the single word is a common Indian name but not a greeting
    if len(text.split()) == 1 and text_lower not in common_greetings:
        # Check in male names (but not if it's a gender term)
        if text_lower in INDIAN_NAMES['male'] and text_lower not in gender_map:
            entities['name'] = text.title()
            return entities
        # Check in female names (but not if it's a gender term)
        if text_lower in INDIAN_NAMES['female'] and text_lower not in gender_map:
            entities['name'] = text.title()
            return entities
    
    # Use spaCy for more complex entity extraction
    doc = nlp(text)
    
    # Extract name
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            # Ensure it's not a gender term or greeting
            if ent.text.lower() not in gender_map and ent.text.lower() not in common_greetings:
                entities['name'] = ent.text
                break
    
    # Direct age patterns
    age_patterns = [
        r"(\d+)\s*(?:years|yrs|yr|year)(?:\s*old)?",
        r"(?:I am|I'm)\s*(\d+)(?:\s*years|yrs|yr|year)?(?:\s*old)?",
        r"age\s*(?:is|:)?\s*(\d+)",
        r"(\d+)\s*(?:y\.?o\.?|yo)"
    ]
    
    if 'age' not in entities:
        for pattern in age_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    entities['age'] = int(match.group(1))
                    break
                except:
                    pass
    
    # Additional name patterns
    name_patterns = [
        r"(?:my name is|I am|I'm|call me)\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)",
        r"name\s*(?:is|:)?\s*([A-Za-z]+(?:\s+[A-Za-z]+)*)"
    ]
    
    if 'name' not in entities:
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Don't extract name if it's a gender term or greeting
                name_candidate = match.group(1)
                if name_candidate.lower() not in gender_map and name_candidate.lower() not in common_greetings:
                    entities['name'] = name_candidate.title()
                    break
    
    # Extract location if not already found
    if 'location' not in entities:
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                entities['location'] = ent.text
                break
    
    # Additional location patterns
    location_patterns = [
        r"(?:I live in|I am from|I'm from|I reside in)\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)",
        r"location\s*(?:is|:)?\s*([A-Za-z]+(?:\s+[A-Za-z]+)*)"
    ]
    
    if 'location' not in entities:
        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                entities['location'] = match.group(1).title()
                break
    
    return entities

def create_training_data():
    # Load existing training data
    training_data = []
    with open(ENTITY_TRAINING_DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                text, entities = line.strip().split('|')
                training_data.append((text, entities))
    
    # Add more diverse examples
    additional_examples = [
        ("My name is Priya Sharma", "name:priya sharma"),
        ("I am 30 years old", "age:30"),
        ("I am a girl", "gender:female"),
        ("I live in Delhi, India", "location:delhi"),
        ("My father's name is Rajesh Kumar", "father_name:rajesh kumar"),
        ("My mother's name is Sunita Sharma", "mother_name:sunita sharma"),
        ("I am a student", "occupation:student"),
        ("I work as a software engineer", "occupation:software engineer"),
        ("My email is john.doe@example.com", "email:john.doe@example.com"),
        ("My phone number is 9876543210", "phone:9876543210"),
        ("I am from Bangalore, Karnataka", "location:bangalore"),
        ("I am 28 years old", "age:28"),
        ("My name is Amit Patel", "name:amit patel"),
        ("I am a doctor", "occupation:doctor"),
        ("I live in Chennai, Tamil Nadu", "location:chennai")
    ]
    
    training_data.extend(additional_examples)
    return training_data

def create_spacy_training_data(training_data):
    """
    Create spaCy DocBin for training from the custom training data format.
    
    Args:
        training_data: List of (text, entities) tuples
        
    Returns:
        DocBin: SpaCy DocBin object with training examples
    """
    nlp = spacy.blank("en")
    db = DocBin()
    
    for text, entities_str in training_data:
        doc = nlp.make_doc(text)
        ents = []
        
        # Parse entities from string format
        for entity in entities_str.split(','):
            if ':' not in entity:
                continue
                
            label, value = entity.split(':')
            label = label.upper()
            
            # Find the span in the text
            start = text.lower().find(value.lower())
            if start >= 0:
                end = start + len(value)
                # Make sure span bounds are valid
                if start < end and end <= len(doc):
                    span = spacy.tokens.Span(doc, start, end, label=label)
                    ents.append(span)
        
        # Only add examples with valid entities
        if ents:
            doc.ents = ents
            db.add(doc)
    
    return db

def train_entity_recognition_model():
    """Train a spaCy NER model for entity recognition."""
    # Create training data
    if not os.path.exists(ENTITY_TRAINING_DATA_PATH):
        with open(ENTITY_TRAINING_DATA_PATH, 'w', encoding='utf-8') as f:
            for example in TRAINING_EXAMPLES:
                f.write(f"{example[0]}|{example[1]}\n")
    
    training_data = create_training_data()
    
    # Convert to spaCy format
    nlp = spacy.blank("en")
    
    # Add entity labels
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
        
    for entity_type in ENTITY_TYPES:
        ner.add_label(entity_type)
    
    # Create training examples
    examples = []
    for text, annotations in training_data:
        doc = nlp.make_doc(text)
        entities = []
        for annotation in annotations.split(","):
            if ":" in annotation:
                entity_type, value = annotation.split(":")
                entity_type = entity_type.upper()
                start = text.lower().find(value.lower())
                if start >= 0:
                    end = start + len(value)
                    entities.append((start, end, entity_type))
        
        if entities:
            examples.append(Example.from_dict(doc, {"entities": entities}))
    
    # Create a new model
    nlp = spacy.blank("en")
    
    # Add NER component
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    
    # Add entity labels
    for entity_type in ENTITY_TYPES:
        ner.add_label(entity_type)
    
    # Train the model
    optimizer = nlp.begin_training()
    
    # Create batches
    batch_size = 8
    for i in range(100):  # 100 iterations
        random.shuffle(examples)
        batches = [examples[i:i+batch_size] for i in range(0, len(examples), batch_size)]
        
        losses = {}
        for batch in batches:
            nlp.update(batch, sgd=optimizer, drop=0.2, losses=losses)
        
        if i % 10 == 0:
            print(f"Iteration {i}, Losses: {losses}")
    
    # Save the model
    if not os.path.exists(MODEL_OUTPUT_PATH):
        os.makedirs(MODEL_OUTPUT_PATH)
    
    nlp.to_disk(MODEL_OUTPUT_PATH)
    print(f"Model saved to {MODEL_OUTPUT_PATH}")
    
    # Training is complete, so we can skip the spaCy CLI training
    print("Entity recognition model training completed.")
    
    # Let's test the model
    test_entity_extraction()

def test_entity_extraction():
    """Test entity extraction on sample queries."""
    test_queries = [
        "My name is John Doe",
        "I am 25 years old",
        "I am from Delhi",
        "I'm a male",
        # Single word examples
        "shlok",
        "45",
        "male",
        "delhi",
        "guy"
    ]
    
    # Try to load the model
    try:
        # This will trigger loading the model
        for text in test_queries:
            print(f"\nQuery: '{text}'")
            entities = extract_entities(text)
            print(f"Entities: {entities}")
    except Exception as e:
        print(f"Error testing entity extraction: {str(e)}")
        # Don't raise the exception here to allow the program to continue

if __name__ == "__main__":
    train_entity_recognition_model()
