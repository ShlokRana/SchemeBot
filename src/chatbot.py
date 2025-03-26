import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
import sys
import spacy
import random
import json
from typing import Dict, Tuple, Any, List

# Add root directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.entity_extraction import extract_entities
from src.schemes_data import match_schemes
from src.config import (
    INTENT_MODEL_PATH, 
    ENTITY_RECOGNITION_MODEL_PATH, 
    BERT_MODEL, 
    NUM_INTENTS,
    MAX_ATTEMPTS,
    MIN_AGE,
    MAX_AGE,
    GREETING_RESPONSES,
    INTENT_LABELS
)

# Determine device
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"
device = torch.device(device)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

# Load intent model
try:
    intent_model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=NUM_INTENTS)
    intent_model.load_state_dict(torch.load(INTENT_MODEL_PATH, map_location=device))
    intent_model.to(device)
    intent_model.eval()
    print("Intent model loaded successfully")
except Exception as e:
    print(f"Error loading intent model: {str(e)}")
    print("Please train the model first using python main.py --train")
    intent_model = None

# Load spaCy model
try:
    nlp = spacy.load(ENTITY_RECOGNITION_MODEL_PATH)
    print("Entity recognition model loaded successfully")
except OSError as e:
    print(f"Error loading entity recognition model: {str(e)}")
    print("Please train the model first using python main.py --train")
    nlp = None

# Define conversation states
CONVERSATION_STATES = {
    "INITIAL": "initial",
    "WAITING_FOR_NAME": "waiting_for_name",
    "WAITING_FOR_GENDER": "waiting_for_gender",
    "WAITING_FOR_AGE": "waiting_for_age",
    "WAITING_FOR_LOCATION": "waiting_for_location",
    "COLLECT_OCCUPATION": "collect_occupation",
    "COLLECT_INCOME": "collect_income",
    "SHOW_SCHEMES": "show_schemes",
    "SCHEME_DETAILS": "scheme_details",
    "END_CONVERSATION": "end_conversation"
}

# Questions for each state
STATE_QUESTIONS = {
    CONVERSATION_STATES["INITIAL"]: "Hello! I'm SchemeBot. I can help you find government schemes you may be eligible for. Can you tell me your name?",
    CONVERSATION_STATES["WAITING_FOR_NAME"]: "Can you tell me your name?",
    CONVERSATION_STATES["WAITING_FOR_GENDER"]: "Can you tell me your gender?",
    CONVERSATION_STATES["WAITING_FOR_AGE"]: "What is your age?",
    CONVERSATION_STATES["WAITING_FOR_LOCATION"]: "Which state in India do you live in?",
    CONVERSATION_STATES["COLLECT_OCCUPATION"]: "What is your occupation? (If you're a student, farmer, etc.)",
    CONVERSATION_STATES["COLLECT_INCOME"]: "What is your approximate annual income?",
    CONVERSATION_STATES["SHOW_SCHEMES"]: "Based on your information, here are some schemes you might be eligible for.",
    CONVERSATION_STATES["SCHEME_DETAILS"]: "Would you like to know more details about any specific scheme?",
    CONVERSATION_STATES["END_CONVERSATION"]: "Thank you for using SchemeBot. Is there anything else you'd like to know?"
}

# Follow-up prompts
FOLLOW_UP_PROMPTS = {
    "MISSING_NAME": [
        "I didn't catch your name. Could you please tell me your name?",
        "I need your name to proceed. What should I call you?",
        "Sorry, I missed your name. Can you tell me again?"
    ],
    "MISSING_GENDER": [
        "I didn't understand your gender. Could you specify if you're male, female, or other?",
        "Please tell me your gender (male, female, or other)",
        "I need to know your gender to find relevant schemes. Are you male, female, or other?"
    ],
    "MISSING_AGE": [
        "I didn't catch your age. How old are you?",
        "Could you tell me your age in years?",
        "Your age will help me find age-specific schemes. How old are you?"
    ],
    "MISSING_LOCATION": [
        "I didn't understand your state. Could you specify which state you live in?",
        "Please tell me which state you reside in.",
        "I need to know your state to find schemes available for you."
    ],
    "INVALID_AGE": [
        f"Please provide a valid age between {MIN_AGE} and {MAX_AGE}",
        "That doesn't seem like a valid age. Can you provide your age in years?",
        f"Your age should be between {MIN_AGE} and {MAX_AGE} years. What is your actual age?"
    ]
}

def get_intent(text: str, model=None, tokenizer=None) -> int:
    """Get intent from user input."""
    try:
        # Handle single-word special cases
        text_lower = text.lower().strip()
        
        # Basic direct intent mapping for common single-word responses
        if text_lower in ['hi', 'hello', 'hey', 'yo', 'sup', 'hola']:
            return INTENT_LABELS.index("greeting")
            
        if text_lower in ['bye', 'goodbye', 'exit', 'quit', 'cya', 'thanks']:
            return INTENT_LABELS.index("goodbye")
        
        # Check if input is just a number (likely an age)
        if text.strip().isdigit():
            age = int(text.strip())
            if 0 <= age <= 120:
                return INTENT_LABELS.index("provide_age")
                
        # Check for gender terms
        if text_lower in ['male', 'female', 'man', 'woman', 'boy', 'girl', 'guy', 'm', 'f']:
            return INTENT_LABELS.index("provide_gender")
            
        # Standard model-based intent classification
        if model is None:
            model = intent_model
        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
            
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get confidence scores
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, predicted_class = torch.max(probabilities, dim=1)
        
        # If confidence is too low, default to out_of_scope
        if confidence.item() < 0.6:  # Threshold can be adjusted
            return INTENT_LABELS.index("out_of_scope")
            
        return predicted_class.item()
    except Exception as e:
        print(f"Error in intent classification: {str(e)}")
        return INTENT_LABELS.index("greeting")  # Default to greeting intent on error

def determine_next_state(current_state: str, intent: int, entities: Dict[str, Any], user_info: Dict[str, Any]) -> str:
    """Determine the next conversation state based on current state, intent, and entities."""
    
    # Map intent index to label for better readability
    intent_label = INTENT_LABELS[intent]
    
    # Handle goodbye intent from any state
    if intent_label == "goodbye":
        return CONVERSATION_STATES["END_CONVERSATION"]
        
    # State transitions
    if current_state == CONVERSATION_STATES["INITIAL"]:
        if intent_label == "greeting":
            return CONVERSATION_STATES["WAITING_FOR_NAME"]
        elif "name" in entities or intent_label == "provide_name":
            return CONVERSATION_STATES["WAITING_FOR_GENDER"]
        else:
            return CONVERSATION_STATES["WAITING_FOR_NAME"]
            
    elif current_state == CONVERSATION_STATES["WAITING_FOR_NAME"]:
        if "name" in entities or intent_label == "provide_name":
            return CONVERSATION_STATES["WAITING_FOR_GENDER"]
        else:
            return CONVERSATION_STATES["WAITING_FOR_NAME"]
            
    elif current_state == CONVERSATION_STATES["WAITING_FOR_GENDER"]:
        if "gender" in entities or intent_label == "provide_gender":
            return CONVERSATION_STATES["WAITING_FOR_AGE"]
        else:
            return CONVERSATION_STATES["WAITING_FOR_GENDER"]
            
    elif current_state == CONVERSATION_STATES["WAITING_FOR_AGE"]:
        if "age" in entities or intent_label == "provide_age":
            return CONVERSATION_STATES["WAITING_FOR_LOCATION"]
        else:
            return CONVERSATION_STATES["WAITING_FOR_AGE"]
            
    elif current_state == CONVERSATION_STATES["WAITING_FOR_LOCATION"]:
        if "location" in entities or intent_label == "provide_location":
            return CONVERSATION_STATES["SHOW_SCHEMES"]
        else:
            return CONVERSATION_STATES["WAITING_FOR_LOCATION"]
            
    elif current_state == CONVERSATION_STATES["SHOW_SCHEMES"]:
        return CONVERSATION_STATES["SCHEME_DETAILS"]
            
    elif current_state == CONVERSATION_STATES["SCHEME_DETAILS"]:
        return CONVERSATION_STATES["END_CONVERSATION"]
            
    # Default to stay in the same state
    return current_state

def get_response(
    user_input: str,
    intent: int,
    entities: Dict[str, Any],
    user_info: Dict[str, Any],
    conversation_state: str
) -> Tuple[str, Dict[str, Any]]:
    """Generate response based on intent, entities, and current state."""
    debug_info = {
        "intent": INTENT_LABELS[intent],
        "entities": entities,
        "user_info": user_info.copy(),
        "conversation_state": conversation_state
    }
    
    # Update user info based on entities
    for entity_type, value in entities.items():
        if entity_type.lower() in ['name', 'gender', 'age', 'location']:
            user_info[entity_type.lower()] = value
    
    # Determine next state
    next_state = determine_next_state(conversation_state, intent, entities, user_info)
    conversation_state = next_state
    debug_info["next_state"] = next_state
    
    # Generate response based on state
    if conversation_state == CONVERSATION_STATES["INITIAL"] or intent == INTENT_LABELS.index("greeting"):
        if conversation_state == CONVERSATION_STATES["INITIAL"]:
            response = random.choice(GREETING_RESPONSES)
        else:
            response = f"Hello! {STATE_QUESTIONS[CONVERSATION_STATES['WAITING_FOR_NAME']]}"
            
    elif conversation_state == CONVERSATION_STATES["WAITING_FOR_NAME"]:
        if "name" in entities:
            response = random.choice(FOLLOW_UP_PROMPTS["MISSING_NAME"])
        else:
            response = STATE_QUESTIONS[CONVERSATION_STATES["WAITING_FOR_NAME"]]
            
    elif conversation_state == CONVERSATION_STATES["WAITING_FOR_GENDER"]:
        if "gender" in entities:
            response = random.choice(FOLLOW_UP_PROMPTS["MISSING_GENDER"])
        else:
            response = STATE_QUESTIONS[CONVERSATION_STATES["WAITING_FOR_GENDER"]]
            
    elif conversation_state == CONVERSATION_STATES["WAITING_FOR_AGE"]:
        if "age" in entities:
            age = entities["age"]
            if MIN_AGE <= int(age) <= MAX_AGE:
                response = random.choice(FOLLOW_UP_PROMPTS["MISSING_AGE"])
            else:
                response = random.choice(FOLLOW_UP_PROMPTS["INVALID_AGE"])
        else:
            response = STATE_QUESTIONS[CONVERSATION_STATES["WAITING_FOR_AGE"]]
            
    elif conversation_state == CONVERSATION_STATES["WAITING_FOR_LOCATION"]:
        if "location" in entities:
            response = random.choice(FOLLOW_UP_PROMPTS["MISSING_LOCATION"])
        else:
            response = STATE_QUESTIONS[CONVERSATION_STATES["WAITING_FOR_LOCATION"]]
            
    elif conversation_state == CONVERSATION_STATES["SHOW_SCHEMES"]:
        matching_schemes = match_schemes(user_info)
        
        # Format schemes as a list
        if matching_schemes:
            schemes_list = "\n\n".join([f"**{i+1}. {scheme['name']}**\n{scheme['description']}" 
                                     for i, scheme in enumerate(matching_schemes)])
            response = f"Based on your information, here are schemes you might be eligible for:\n\n{schemes_list}\n\nWould you like to know more details about any specific scheme?"
        else:
            response = "I couldn't find any schemes matching your profile. Please check back later as we update our database regularly."
            
    elif conversation_state == CONVERSATION_STATES["SCHEME_DETAILS"]:
        # Check if the user is asking about a specific scheme
        matching_schemes = match_schemes(user_info)
        scheme_found = False
        
        for scheme in matching_schemes:
            if scheme["name"].lower() in user_input.lower():
                scheme_found = True
                # Provide detailed information about the scheme
                response = f"**{scheme['name']}**\n\n"
                response += f"Description: {scheme['description']}\n\n"
                response += f"Benefits: {scheme['benefits']}\n\n"
                response += f"Documents Required: {', '.join(scheme['documents_required'])}\n\n"
                response += f"How to Apply: {scheme['application_process']}"
                break
                
        if not scheme_found:
            # Try to match by scheme number
            numbers = [int(s) for s in user_input.split() if s.isdigit()]
            if numbers and 1 <= numbers[0] <= len(matching_schemes):
                scheme = matching_schemes[numbers[0]-1]
                response = f"**{scheme['name']}**\n\n"
                response += f"Description: {scheme['description']}\n\n"
                response += f"Benefits: {scheme['benefits']}\n\n"
                response += f"Documents Required: {', '.join(scheme['documents_required'])}\n\n"
                response += f"How to Apply: {scheme['application_process']}"
            else:
                response = "Please specify which scheme you'd like to know more about by name or number."
                
    elif conversation_state == CONVERSATION_STATES["END_CONVERSATION"]:
        response = "Thank you for using SchemeBot! Is there anything else you'd like to know about government schemes?"
        
    else:
        # Default response for unexpected states
        response = "I'm not sure how to proceed. Can you please provide more information?"
    
    debug_info["response"] = response
    debug_info["user_info"] = user_info
    debug_info["conversation_state"] = conversation_state
    
    return response, debug_info

def chatbot():
    print("Hello! My name is SchemeBot. I help you find government schemes.")
    
    # Initialize conversation state
    conversation_state = CONVERSATION_STATES["INITIAL"]
    user_info = {}
    
    while True:
        # Get user input
        user_input = input("> ")
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Thank you for using SchemeBot. Goodbye!")
            break
            
        # Process user input
        intent = get_intent(user_input, intent_model, tokenizer)
        entities = extract_entities(user_input)
        
        # Generate response
        response, debug_info = get_response(user_input, intent, entities, user_info, conversation_state)
        
        # Update conversation state
        conversation_state = debug_info["conversation_state"]
        
        # Print response
        print(response)
        
        # Debug information
        if "--debug" in sys.argv:
            print("\nDEBUG INFO:")
            print(f"Intent: {debug_info['intent']}")
            print(f"Entities: {debug_info['entities']}")
            print(f"User Info: {debug_info['user_info']}")
            print(f"State: {debug_info['conversation_state']}")
            print("----------------------")

if __name__ == "__main__":
    chatbot()
