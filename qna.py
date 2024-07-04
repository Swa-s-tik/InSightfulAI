import streamlit as st
from transformers import pipeline
import json
import datetime
import fitz  # Instead of pdfplumber
import io
class AIAssistant:
    def __init__(self):
        self.qa_pipe = pipeline("question-answering", model="deepset/roberta-base-squad2")
        self.summary_pipe = pipeline("summarization", model="facebook/bart-large-cnn")
        self.mask_pipe = pipeline("fill-mask", model="bert-base-uncased")
        self.table_qa_pipe = pipeline("table-question-answering", model="google/tapas-base-finetuned-wtq")
        self.sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.history = []

    def answer_question(self, question, context):
        result = self.qa_pipe(question=question, context=context)
        return result['answer']

    def summarize_text(self, text):
        summary = self.summary_pipe(text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        return summary

    def fill_mask(self, text):
        return self.mask_pipe(text)

    def table_qa(self, table, question):
        return self.table_qa_pipe(table=table, query=question)['answer']

    def sentiment_analysis(self, text):
        return self.sentiment_pipe(text)[0]

    def add_to_history(self, item):
        item['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.history.append(item)

    def get_history(self):
        return self.history

    def clear_history(self):
        self.history = []

# Initialize session state
if 'assistant' not in st.session_state:
    st.session_state.assistant = AIAssistant()
if 'context' not in st.session_state:
    st.session_state.context = ""

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")

# Sidebar for page selection
page = st.sidebar.selectbox("Select a page:", ["Question Answering", "Text Summarization", "Fill Mask", "Table Question Answering", "Sentiment Analysis"])

def load_pdf(uploaded_file):
    with fitz.open(stream=uploaded_file.getvalue(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()
    return text

# Common sidebar elements
with st.sidebar:
    if st.button("Clear History"):
        st.session_state.assistant.clear_history()
        st.success("History cleared!")

# Page-specific content
if page == "Question Answering":
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

elif page == "Text Summarization":
    st.title("🤖 AI Text Summarizer")
    st.caption("Using model: [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn)")

    with st.sidebar:
        st.header("Text Settings")
        text_input = st.text_area("Enter or modify text here:", value=st.session_state.context, height=300)
        uploaded_file = st.file_uploader("Or upload a PDF file for summarization", type="pdf")
        if uploaded_file:
            st.session_state.context = load_pdf(uploaded_file)
            st.success("Text loaded from PDF!")

    st.header("Summarization")
    for item in st.session_state.assistant.get_history():
        if item.get('type') == 'summary':
            with st.expander(f"Summary from {item['timestamp']}"):
                st.info(item['original'][:100] + "...")
                st.success(item['summary'])

    if st.button("Summarize"):
        if not st.session_state.context:
            st.warning("Please provide some text first!")
        else:
            summary = st.session_state.assistant.summarize_text(st.session_state.context)
            st.session_state.assistant.add_to_history({'type': 'summary', 'original': st.session_state.context, 'summary': summary})
            st.info(st.session_state.context[:100] + "...")
            st.success(f"Summary: {summary}")

elif page == "Fill Mask":
    st.title("🤖 AI Mask Filler")
    st.caption("Using model: [bert-base-uncased](https://huggingface.co/bert-base-uncased)")

    masked_text = st.text_input("Enter text with [MASK] token:", "The capital of France is [MASK].")
    if st.button("Fill Mask"):
        results = st.session_state.assistant.fill_mask(masked_text)
        st.session_state.assistant.add_to_history({'type': 'mask', 'input': masked_text, 'results': results})
        for result in results:
            st.success(f"{result['score']:.2f}: {result['sequence']}")

elif page == "Table Question Answering":
    st.title("🤖 AI Table Question Answering")
    st.caption("Using model: [google/tapas-base-finetuned-wtq](https://huggingface.co/google/tapas-base-finetuned-wtq)")

    st.write("Enter your table data (CSV format):")
    table_data = st.text_area("", "Name,Age,City\nAlice,25,New York\nBob,30,San Francisco\nCharlie,35,Chicago")
    question = st.text_input("Ask a question about the table:")
    if st.button("Answer"):
        answer = st.session_state.assistant.table_qa(table_data, question)
        st.session_state.assistant.add_to_history({'type': 'table_qa', 'table': table_data, 'question': question, 'answer': answer})
        st.success(f"Answer: {answer}")

elif page == "Sentiment Analysis":
    st.title("🤖 AI Sentiment Analysis")
    st.caption("Using model: [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)")

    text = st.text_area("Enter text for sentiment analysis:")
    if st.button("Analyze Sentiment"):
        result = st.session_state.assistant.sentiment_analysis(text)
        st.session_state.assistant.add_to_history({'type': 'sentiment', 'text': text, 'sentiment': result})
        st.success(f"Sentiment: {result['label']} (Score: {result['score']:.4f})")

# Download history
if st.sidebar.button("Download History"):
    history = st.session_state.assistant.get_history()
    if history:
        json_str = json.dumps(history, indent=2)
        st.sidebar.download_button(
            label="Download JSON",
            file_name="ai_assistant_history.json",
            mime="application/json",
            data=json_str
        )
    else:
        st.sidebar.warning("No history to download.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and Hugging Face Transformers")