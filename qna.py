import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import json
import datetime
import pdfplumber
import io
import gc

class AIAssistant:
    def __init__(self):
        self.history = []
        self.models = {}

    def load_model(self, task):
        if task not in self.models:
            if task == "question-answering":
                self.models[task] = pipeline("question-answering", model="deepset/roberta-base-squad2")
            elif task == "summarization":
                self.models[task] = pipeline("summarization", model="facebook/bart-large-cnn")
            elif task == "fill-mask":
                self.models[task] = pipeline("fill-mask", model="bert-base-uncased")
            elif task == "sentence-similarity":
                self.models[task] = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            elif task == "table-question-answering":
                self.models[task] = pipeline("table-question-answering", model="google/tapas-base-finetuned-wtq")
            elif task == "sentiment-analysis":
                self.models[task] = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        return self.models[task]

    def answer_question(self, question, context):
        model = self.load_model("question-answering")
        result = model(question=question, context=context)
        return result['answer']

    def summarize_text(self, text):
        model = self.load_model("summarization")
        summary = model(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        return summary

    def fill_mask(self, text):
        model = self.load_model("fill-mask")
        return model(text)

    def sentence_similarity(self, sentence1, sentence2):
        model = self.load_model("sentence-similarity")
        embeddings = model.encode([sentence1, sentence2])
        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        return similarity.item()

    def table_qa(self, table, question):
        model = self.load_model("table-question-answering")
        return model(table=table, query=question)['answer']

    def sentiment_analysis(self, text):
        model = self.load_model("sentiment-analysis")
        return model(text)[0]

    def add_to_history(self, item):
        item['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.history.append(item)

    def get_history(self):
        return self.history

    def clear_history(self):
        self.history = []

def load_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(io.BytesIO(uploaded_file.getvalue())) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
            gc.collect()  # Force garbage collection after each page
    return text

def add_bg_from_url(image_url):
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("{image_url}");
             background-attachment: fixed;
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# Initialize session state
if 'assistant' not in st.session_state:
    st.session_state.assistant = AIAssistant()
if 'context' not in st.session_state:
    st.session_state.context = ""

st.set_page_config(page_title="InSightful AI", page_icon="🤖", layout="wide")

# Sidebar for page selection
page = st.sidebar.selectbox("Select a page:", ["Question Answering", "Text Summarization", "Fill Mask", "Sentence Similarity", "Table Question Answering", "Sentiment Analysis"])

# Common sidebar elements
with st.sidebar:
    st.title("InSightful AI")
    st.caption("From Data to Story, in a Heartbeat")
    if st.button("Clear History"):
        st.session_state.assistant.clear_history()
        st.success("History cleared!")

# Page-specific content
if page == "Question Answering":
    add_bg_from_url("https://images.unsplash.com/photo-1596496050755-c923e73e42e1?auto=format&fit=crop&q=80&w=2940&ixlib=rb-4.0.3")
    st.title("🤖 AI Q&A Assistant")
    st.caption("Using model: [deepset/roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2)")

    with st.sidebar:
        st.header("Context Settings")
        context_input = st.text_area("Enter or modify context here:", value=st.session_state.context, height=300)
        uploaded_file = st.file_uploader("Or upload a PDF file for context", type="pdf")
        if uploaded_file:
            st.session_state.context = load_pdf(uploaded_file)
            st.success("Context loaded from PDF!")

    st.header("Chat")
    for item in st.session_state.assistant.get_history():
        if item.get('type') == 'qa':
            st.info(f"Q: {item['question']}")
            st.success(f"A: {item['answer']}")

    question = st.text_input("Ask a question based on the context:")
    if st.button("Ask"):
        if not st.session_state.context:
            st.warning("Please provide some context first!")
        elif not question:
            st.warning("Please enter a question!")
        else:
            answer = st.session_state.assistant.answer_question(question, st.session_state.context)
            st.session_state.assistant.add_to_history({'type': 'qa', 'question': question, 'answer': answer})
            st.info(f"Q: {question}")
            st.success(f"A: {answer}")

# ... [Rest of the code remains the same, just ensure you're using st.session_state.assistant.load_model() before using any model] ...

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and Hugging Face Transformers")