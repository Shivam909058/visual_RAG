from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import os

class RAGEngine:
    def __init__(self):
        # Use HuggingFace embeddings if no OpenAI key
        if os.getenv("OPENAI_API_KEY"):
            self.embeddings = OpenAIEmbeddings()
        else:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
            
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Use local model if no OpenAI key
        if os.getenv("OPENAI_API_KEY"):
            self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        else:
            from langchain_community.llms import Ollama
            self.llm = Ollama(model="llama3")
            
        self.vector_store = None
        self.qa_chain = None
    
    def process_text(self, text, image_contexts):
        """Process text and image contexts into a vector store"""
        # Combine text with image descriptions
        combined_text = text + "\n\nImage Contexts:\n" + "\n".join(image_contexts)
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(combined_text)
        
        # Create vector store
        self.vector_store = Chroma.from_texts(
            chunks,
            self.embeddings,
            collection_name="pdf_store"
        )
        
        # Create QA chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=self.vector_store.as_retriever(),
            return_source_documents=True
        )
    
    def ask_question(self, question, chat_history=[]):
        """Ask a question about the processed document"""
        if not self.qa_chain:
            return "Please process a document first."
        
        try:
            response = self.qa_chain.invoke({
                "question": question, 
                "chat_history": chat_history
            })
            
            # Format the response
            answer = response["answer"]
            
            # Format mathematical formulas if present
            if "$$" in answer or "\[" in answer:
                answer = answer.replace("$$", "$")
                answer = answer.replace("\[", "$$").replace("\]", "$$")
                answer = answer.replace("\text", "\operatorname")
            
            return answer
            
        except Exception as e:
            return f"Error generating response: {str(e)}" 