# SchemeBot 🤖

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B.svg)](https://streamlit.io)
[![spaCy](https://img.shields.io/badge/spaCy-3.7.2-09A3D5.svg)](https://spacy.io)
[![Transformers](https://img.shields.io/badge/Transformers-4.36.0-FFB000.svg)](https://huggingface.co/transformers/)

SchemeBot is an intelligent chatbot designed to help Indian citizens find government schemes they may be eligible for. Using advanced NLP and machine learning techniques, it engages in natural conversations to gather user information and recommend relevant government schemes.

## ⚠️ Important Note

Before running the application, you **MUST** train the models first. This step is crucial as the models are not included in the repository due to size constraints.

```bash
# First, train the models
python main.py --train

# Then, run the application
python main.py --run
```

## 🌟 Features

- 🗣️ Natural language understanding for seamless conversations
- 🎯 Intelligent intent classification and entity extraction
- 🔄 Context-aware dialogue management
- 📊 Real-time scheme matching based on user profile
- 🎨 Beautiful Streamlit web interface
- 🧠 BERT-based deep learning model for intent classification
- 🏷️ Custom spaCy model for named entity recognition
- 🔍 Fuzzy matching for location recognition
- 📱 Mobile-friendly responsive design

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **Backend**: Python 3.8+
- **NLP**: spaCy, Transformers (BERT)
- **ML Framework**: PyTorch
- **Data Format**: YAML, JSON
- **Development Tools**: Python Virtual Environment

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git
- 4GB+ RAM recommended
- CUDA-compatible GPU (optional, for faster processing)

## ⚡ Quick Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/schemeBot.git
   cd schemeBot
   ```

2. **Create and activate virtual environment**
   ```bash
   # On Windows
   python -m venv myenv
   myenv\\Scripts\\activate

   # On macOS/Linux
   python3 -m venv myenv
   source myenv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the models (Required)**
   ```bash
   # This step is mandatory before running the application
   python main.py --train
   ```

5. **Start the application**
   ```bash
   # After training is complete
   python main.py --run
   # OR use Streamlit interface
   streamlit run app.py
   ```

## 🗂️ Project Structure

```
schemeBot/
├── app.py                 # Streamlit web application
├── main.py               # Main script for training and CLI
├── requirements.txt      # Project dependencies
├── config.cfg           # Configuration settings
│
├── src/
│   ├── chatbot.py        # Core chatbot logic
│   ├── entity_extraction.py  # Entity recognition
│   ├── train_nlu.py      # Training pipeline
│   ├── schemes_data.py   # Scheme matching logic
│   ├── config.py         # Global configurations
│   └── entity_patterns.py # Pattern definitions
│
├── data/
│   ├── nlu_data.yml      # Training data for NLU
│   └── entity_training_data.txt  # Entity training data
│
├── models/               # Trained model files
│   ├── intent_model.pth
│   └── entity_recognition/
│
└── plots/               # Training visualization plots
```

## 🚀 Usage

1. **Start the Web Interface**
   ```bash
   python main.py --run
   ```
   Access the application at `http://localhost:8501`

2. **Command Line Interface**
   ```bash
   python main.py --chat
   ```

3. **Training Models**
   ```bash
   python main.py --train
   ```

## 💬 Example Conversation

```
Bot: Hello! I'm SchemeBot. How can I help you today?
User: Hi
Bot: Can you tell me your name?
User: John Doe
Bot: Nice to meet you, John! Can you tell me your gender?
User: Male
Bot: What is your age?
User: 25
Bot: Which state in India do you live in?
User: Karnataka
Bot: Based on your information, here are schemes you might be eligible for...
```

## 🎯 Intent Categories

- Greeting
- Provide Name
- Provide Gender
- Provide Age
- Provide Location
- Goodbye
- Out of Scope

## 🏷️ Entity Types

- Name
- Age
- Gender
- Location

## 🔧 Configuration

You can modify the following settings in `config.py`:
- Model parameters
- Training configurations
- Validation thresholds
- State management rules

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Your Name - *Initial work* - [YourGithub](https://github.com/yourusername)

## 🙏 Acknowledgments

- BERT model from Hugging Face
- spaCy for NER capabilities
- Streamlit for the web interface
- Indian Government Schemes Database

## 📞 Support

For support, email your.email@example.com or open an issue in the repository.

---
Made with ❤️ in India 