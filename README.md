# Medical Chatbot

## Description
This project implements a medical chatbot using Streamlit and a fine-tuned language model based on DistilGPT-2. The chatbot is designed to provide general medical information and answer health-related questions.

## Features
- Web-based interface using Streamlit
- Powered by a fine-tuned DistilGPT-2 model using PEFT (Parameter-Efficient Fine-Tuning)
- Provides concise and informative responses to medical queries
- Chat history functionality
- Responsive design with custom CSS styling
- Informative sidebar with model details and limitations

## Requirements
- Python 3.7+
- Streamlit
- PyTorch
- Transformers
- PEFT (Parameter-Efficient Fine-Tuning)

## Installation
1. Clone this repository:
   ```
   git clone https://github.com/yourusername/medical-chatbot.git
   cd medical-chatbot
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and go to the URL displayed in the terminal (usually `http://localhost:8501`).

3. Type your medical question in the text input field and press Enter to get a response from the chatbot.

## Model Details
- Base Model: DistilGPT-2
- Fine-tuning: PEFT (Parameter-Efficient Fine-Tuning)
- Dataset: Medical Q&A corpus

## Limitations and Disclaimers
- This chatbot is for informational purposes only and should not replace professional medical advice.
- The model may generate incorrect or nonsensical information, especially for complex or uncommon medical topics.
- The chatbot's knowledge is based on its training data and may not include the most recent medical developments.
- It cannot perform physical examinations or tests, which are often necessary for accurate diagnoses.
- Users should not rely solely on this chatbot for medical decisions.

## Contributing
Contributions to improve the chatbot are welcome. Please follow these steps:
1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Acknowledgements
- [Streamlit](https://streamlit.io/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PEFT](https://github.com/huggingface/peft)
