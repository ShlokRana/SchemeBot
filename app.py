import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json
import os
from src.entity_extraction import extract_entities
from src.chatbot import get_intent, get_response, CONVERSATION_STATES
from src.schemes_data import match_schemes
from src.config import (
    INTENT_MODEL_PATH, INTENT_LABELS_PATH, ENTITY_RECOGNITION_MODEL_PATH,
    INTENT_LABELS, ENTITY_TYPES, MODEL_CONFIG
)

# Set page config
st.set_page_config(
    page_title="SchemeBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_info" not in st.session_state:
    st.session_state.user_info = {
        "name": None,
        "age": None,
        "gender": None,
        "location": None
    }
if "conversation_state" not in st.session_state:
    st.session_state.conversation_state = CONVERSATION_STATES["INITIAL"]
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "matched_schemes" not in st.session_state:
    st.session_state.matched_schemes = []

# Load models
@st.cache_resource
def load_models():
    try:
        # Load intent classification model
        tokenizer = BertTokenizer.from_pretrained(MODEL_CONFIG["model_name"])
        model = BertForSequenceClassification.from_pretrained(
            MODEL_CONFIG["model_name"],
            num_labels=len(INTENT_LABELS)
        )
        model.load_state_dict(torch.load(INTENT_MODEL_PATH, map_location=MODEL_CONFIG["device"]))
        model.to(MODEL_CONFIG["device"])
        model.eval()
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

# Initialize models
intent_model, tokenizer = load_models()

# Title and description
st.title("SchemeBot 🤖")
st.markdown("""
    Welcome to SchemeBot! I'm here to help you find government schemes that match your profile.
    Please provide your information, and I'll guide you through the process.
""")

# Debug mode toggle in sidebar
with st.sidebar:
    st.subheader("Settings")
    st.session_state.debug_mode = st.checkbox("Debug Mode", value=st.session_state.debug_mode)
    if st.session_state.debug_mode:
        st.markdown("### Debug Information")
        st.json(st.session_state.user_info)
        st.markdown("### Current State")
        st.write(st.session_state.conversation_state)
        
        # Display matched schemes in debug mode
        if st.session_state.matched_schemes:
            st.markdown("### Matched Schemes")
            for i, scheme in enumerate(st.session_state.matched_schemes):
                st.markdown(f"**{i+1}. {scheme['name']}**")

# Main chat interface
chat_container = st.container()

# Display messages
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.text_area("You:", value=message["content"], height=50, disabled=True)
        else:
            st.text_area("Bot:", value=message["content"], height=100, disabled=True)
            if st.session_state.debug_mode and "debug_info" in message:
                with st.expander("Debug Info"):
                    st.json(message["debug_info"])

# Add initial bot message if there are no messages yet
if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Welcome! I'm SchemeBot, your assistant for finding relevant government schemes."
    })

# User input
with st.form(key="message_form"):
    user_input = st.text_input("Type your message here:", key="user_message")
    submit_button = st.form_submit_button("Send")

if submit_button and user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Process user input
    try:
        # Get intent
        intent = get_intent(user_input, intent_model, tokenizer)
        
        # Extract entities
        entities = extract_entities(user_input)
        
        # Get response
        response, debug_info = get_response(
            user_input, intent, entities,
            st.session_state.user_info,
            st.session_state.conversation_state
        )
        
        # Update user info and conversation state
        if debug_info:
            st.session_state.user_info.update(debug_info.get("user_info", {}))
            st.session_state.conversation_state = debug_info.get("conversation_state", st.session_state.conversation_state)
            
            # Check if we're in a state to show schemes
            if st.session_state.conversation_state in [CONVERSATION_STATES["SHOW_SCHEMES"], CONVERSATION_STATES["SCHEME_DETAILS"]]:
                st.session_state.matched_schemes = match_schemes(st.session_state.user_info)
        
        # Add assistant response to chat history
        message_content = {"role": "assistant", "content": response}
        if st.session_state.debug_mode and debug_info:
            message_content["debug_info"] = debug_info
        st.session_state.messages.append(message_content)
        
        # Use experimental_rerun instead of rerun
        st.experimental_rerun()
        
    except Exception as e:
        st.error(f"Error processing message: {str(e)}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "I apologize, but I encountered an error processing your message. Please try again."
        })

# Display scheme cards if matched schemes are available
if st.session_state.matched_schemes and st.session_state.conversation_state in [CONVERSATION_STATES["SHOW_SCHEMES"], CONVERSATION_STATES["SCHEME_DETAILS"]]:
    st.markdown("---")
    st.subheader("Matching Schemes")
    
    # Create a column layout for scheme cards
    cols = st.columns(2)
    for i, scheme in enumerate(st.session_state.matched_schemes):
        col_idx = i % 2
        with cols[col_idx]:
            st.markdown(f"### {scheme['name']}")
            st.markdown(f"**Description:** {scheme['description']}")
            st.markdown(f"**Benefits:** {scheme['benefits']}")
            with st.expander("Details"):
                st.markdown(f"**Documents Required:** {', '.join(scheme['documents_required'])}")
                st.markdown(f"**How to Apply:** {scheme['application_process']}")

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.user_info = {
        "name": None,
        "age": None,
        "gender": None,
        "location": None
    }
    st.session_state.conversation_state = CONVERSATION_STATES["INITIAL"]
    st.session_state.matched_schemes = []
    st.experimental_rerun() 