import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

@st.cache_resource
def load_model():
    config = PeftConfig.from_pretrained("Melbi/bert-large_finetune-melbi")
    base_model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    model = PeftModel.from_pretrained(base_model, "Melbi/bert-large_finetune-melbi")
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    return model, tokenizer

def generate_response(prompt, model, tokenizer, max_length=200):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs, 
            max_length=max_length, 
            num_return_sequences=1, 
            temperature=0.5, 
            no_repeat_ngram_size=2,  
            repetition_penalty=1.0  
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def process_response(response):
    response_lower = response.lower()
    formatted_response = response_lower.strip()
    if formatted_response:
        formatted_response = formatted_response[0].upper() + formatted_response[1:]
    return formatted_response

st.set_page_config(page_title="Medical Chatbot", page_icon="🩺", layout="wide")

st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-weight: bold;
}
.chat-container {
    border: 1px solid #ddd;
    border-radius: 5px;
    padding: 10px;
    margin-bottom: 10px;
    max-height: 400px;
    overflow-y: auto;
}
.message {
    display: flex;
    margin-bottom: 10px;
}
.user-message {
    justify-content: flex-end;
}
.bot-message {
    justify-content: flex-start;
}
.message-content {
    padding: 5px 10px;
    border-radius: 15px;
    max-width: 70%;
}
.user-content {
    background-color: #36454f;
}
.bot-content {
    background-color: #000000;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("About This Chatbot")
st.sidebar.warning("Limitations and Disadvantages")
st.sidebar.write("""
- Potential for Hallucinations: This AI model may sometimes generate incorrect or nonsensical information, especially for complex or uncommon medical topics.
- Limited Knowledge: The model's knowledge is based on its training data and may not include the most recent medical developments or rare conditions.
- Lack of Context: It may not fully understand the user's medical history or specific circumstances, which are crucial for accurate medical advice.
- No Real-time Updates: Unlike human medical professionals, this model doesn't receive real-time updates on new medical research or emerging health issues.
- Inability to Perform Physical Examinations: The chatbot cannot conduct physical examinations or tests, which are often necessary for accurate diagnoses.
- Risk of Misinterpretation: Users may misinterpret the information provided, leading to incorrect self-diagnosis or treatment.
""")

st.sidebar.markdown("---")
st.sidebar.info("For emergencies, always contact your local emergency services or consult with a healthcare professional.")

st.sidebar.markdown("---")
st.sidebar.subheader("Model Details")
st.sidebar.write("""
- Base Model: DistilGPT-2
- Fine-tuning: PEFT (Parameter-Efficient Fine-Tuning)
- Dataset: Medical Q&A corpus
""")

st.sidebar.markdown("---")
st.sidebar.write("""
This medical chatbot is powered by a fine-tuned language model based on DistilGPT-2. 
It's designed to provide general medical information and answer health-related questions.

Key Features:
- Trained on medical data
- Provides concise and informative responses
- Uses advanced natural language processing techniques

Remember: This chatbot is for informational purposes only and should not replace professional medical advice.
""")



# Main content
st.markdown('<p class="big-font">Medical Chatbot</p>', unsafe_allow_html=True)

model, tokenizer = load_model()

if 'history' not in st.session_state:
    st.session_state.history = []

chat_container = st.container()

def on_input_change():
    user_input = st.session_state.user_input
    if user_input:
        st.session_state.history.append({"type": "user", "content": user_input})
        
        full_prompt = f"Answer the following medical question: {user_input}\nAnswer:"
        with st.spinner("Thinking..."):
            response = generate_response(full_prompt, model, tokenizer)
        
        formatted_response = process_response(response)
        
        st.session_state.history.append({"type": "bot", "content": formatted_response})
        st.session_state.user_input = ""  # Clear the input

with chat_container:
    for message in st.session_state.history:
        if message['type'] == 'user':
            st.markdown(f'<div class="message user-message"><div class="message-content user-content">{message["content"]}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="message bot-message"><div class="message-content bot-content">{message["content"]}</div></div>', unsafe_allow_html=True)

st.text_input("Ask a medical question:", key="user_input", on_change=on_input_change)

st.markdown("---")
st.write("Disclaimer: This AI assistant provides general information and should not be used as a substitute for professional medical advice, diagnosis, or treatment.")


# +================+==================+====================+================+================+
# import streamlit as st
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel, PeftConfig
# import torch

# @st.cache_resource
# def load_model():
#     # Load Peft configuration and base model
#     config = PeftConfig.from_pretrained("Melbi/bert-large_finetune-melbi")
#     base_model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
    
#     # Load Peft model
#     model = PeftModel.from_pretrained(base_model, "Melbi/bert-large_finetune-melbi")
    
#     # Load tokenizer
#     tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
    
#     return model, tokenizer

# def generate_response(prompt, model, tokenizer, max_length=100):
#     inputs = tokenizer.encode(prompt, return_tensors="pt")
#     with torch.no_grad():
#         outputs = model.generate(
#             inputs, 
#             max_length=max_length, 
#             num_return_sequences=1, 
#             temperature=0.7,
#             no_repeat_ngram_size=2,  # Prevent repeating n-grams
#             repetition_penalty=1.2   # Penalize repetition
#         )
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Add your app logo
# st.set_page_config(page_title="Medical Q&A Assistant", page_icon="⚕️", layout="wide")


# st.title("Medical Q&A Assistant")

# model, tokenizer = load_model()

# st.write("Note: This is a general-purpose language model and may not have specialized medical knowledge.")

# # Initialize session state for history
# if 'history' not in st.session_state:
#     st.session_state.history = []

# user_input = st.text_input("Ask a medical question:")

# if user_input:
#     full_prompt = f"Answer the following medical question: {user_input}\nAnswer:"
#     with st.spinner("Generating response..."):
#         response = generate_response(full_prompt, model, tokenizer)

#     # Convert response to lowercase
#     response_lower = response.lower()

#     # Split the answer into points
#     points = response_lower.split('answer:')
#     formatted_response = []
#     for point in points:
#         point = point.strip()
#         if point:
#             # Capitalize the first letter
#             formatted_point = point[0].upper() + point[1:]
#             formatted_response.append(formatted_point)

#     # Join the formatted points into a single response string
#     final_response = "\n".join(formatted_response)

#     # Add to history
#     st.session_state.history.append((user_input, final_response))
    
#     # Display the answer
#     st.write("Answer:", final_response)

# # Display the history of questions and answers
# st.markdown("### History")
# for question, answer in st.session_state.history:
#     st.write(f"**Question:** {question}")
#     st.write(f"**Answer:** {answer}")
#     st.markdown("---")

# st.markdown("---")
# st.write("Disclaimer: This AI assistant provides general information and should not be used as a substitute for professional medical advice, diagnosis, or treatment.")
