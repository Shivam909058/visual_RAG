import streamlit as st
import os
from dotenv import load_dotenv
from utils.pdf_processor import PDFProcessor
from utils.image_processor import ImageProcessor
from utils.rag_engine import RAGEngine

# Load environment variables
load_dotenv()

# Set up page config
st.set_page_config(
    page_title="Visual RAG System",
    layout="wide",
    page_icon="📚"
)

# Custom CSS for formula rendering
st.markdown("""
    <style>
        .katex { 
            font-size: 1.5em !important; 
            text-align: center;
            padding: 1em 0;
        }
        .math-block { 
            margin: 2em 0;
            display: block;
        }
        .math-inline {
            padding: 0 0.2em;
        }
        .stMarkdown {
            max-width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_text' not in st.session_state:
    st.session_state.processed_text = None
if 'extracted_images' not in st.session_state:
    st.session_state.extracted_images = []
if 'image_contexts' not in st.session_state:
    st.session_state.image_contexts = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = RAGEngine()

# Sidebar for model selection
with st.sidebar:
    st.title("Model Settings")
    model_choice = st.radio(
        "Choose Vision Model",
        ["GPT-4 Vision model", "Llama Vision model"],
        index=1  # Default to Llama 2 Vision
    )
    
    api_key = None
    if model_choice == "GPT-4 Vision model":
        api_key = st.text_input("Enter OpenAI API Key", type="password")
        if not api_key:
            st.warning("Please enter your OpenAI API key to use GPT-4 Vision")

# Initialize processors
pdf_processor = PDFProcessor()
image_processor = ImageProcessor(
    model_choice="gpt-4o-mini" if model_choice == "GPT-4 Vision model" else "Llama Vision model",
    api_key=api_key
)

# Main content
st.title("PDF Analysis with Text and Image Understanding")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Process button
    if st.button("Process PDF"):
        if model_choice == "GPT-4 Vision" and not api_key:
            st.error("Please enter your OpenAI API key to use GPT-4 Vision")
        else:
            with st.spinner("Processing PDF..."):
                # Extract text and images
                text, images, image_locations = pdf_processor.extract_text_and_images(uploaded_file)
                st.session_state.processed_text = text
                st.session_state.extracted_images = images
                
                # Process images
                image_contexts = []
                for idx, img in enumerate(images):
                    with st.spinner(f"Processing image {idx+1}..."):
                        context = image_processor.process_image(img)
                        image_contexts.append(f"Image {idx+1} Context:\n{context}")
                
                st.session_state.image_contexts = image_contexts
                st.session_state.rag_engine.process_text(text, image_contexts)
                st.success("PDF processed successfully!")
    
    # Display processed content
    if st.session_state.processed_text is not None:
        st.subheader("Extracted Content")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Text Content")
            st.text_area("Extracted Text", st.session_state.processed_text[:1000] + "...", height=300)
        
        with col2:
            st.subheader("Extracted Images")
            for idx, img in enumerate(st.session_state.extracted_images):
                st.image(img, caption=f"Image {idx+1}", use_container_width=True)
                if idx < len(st.session_state.image_contexts):
                    with st.expander(f"Image {idx+1} Analysis"):
                        st.markdown(st.session_state.image_contexts[idx])
        
        # Question input
        user_question = st.text_input("Ask a question about the document:")
        
        if user_question:
            with st.spinner("Generating answer..."):
                answer = st.session_state.rag_engine.ask_question(
                    user_question,
                    st.session_state.chat_history
                )
                
                # Update chat history
                st.session_state.chat_history.append((user_question, answer))
                
                # Display answer with LaTeX support
                st.markdown(answer)
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("Chat History")
            for q, a in st.session_state.chat_history:
                st.write("Q:", q)
                st.markdown("A: " + a)
                st.markdown("---")

# Usage instructions
with st.sidebar:
    st.subheader("How to use")
    st.write("""
    1. Choose your preferred vision model
    2. If using GPT-4 Vision, enter your API key
    3. Upload a PDF file
    4. Click 'Process PDF' to analyze
    5. Ask questions about the document
    
    The system will understand both text and images!
    """)
