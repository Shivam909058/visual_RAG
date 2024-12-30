from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
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
            chunk_size=1024,
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
        # Create documents for main text
        text_chunks = self.text_splitter.split_text(text)
        text_docs = [
            Document(
                page_content=chunk,
                metadata={"type": "text"}
            ) for chunk in text_chunks
        ]
        
        # Create documents for image contexts
        image_docs = [
            Document(
                page_content=context,
                metadata={
                    "type": "image_context",
                    "image_number": idx + 1
                }
            ) for idx, context in enumerate(image_contexts)
        ]
        
        # Combine all documents
        all_docs = text_docs + image_docs
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=all_docs,
            embedding=self.embeddings,
            collection_name="pdf_store"
        )
        
        # Create QA chain with search kwargs
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=self.vector_store.as_retriever(
                search_kwargs={
                    "k": 5,  # Increased to get more context
                    "filter": None
                }
            ),
            return_source_documents=True,
            verbose=True
        )
    
    def ask_question(self, question, chat_history=[]):
        """Ask a question about the processed document"""
        if not self.qa_chain:
            return "Please process a document first."
        
        try:
            # Check if question is about specific image
            import re
            image_number = None
            image_match = re.search(r'image\s+(\d+)', question.lower())
            
            if image_match:
                image_number = int(image_match.group(1))
                # Fix: Update retriever filter with proper Chroma syntax
                self.qa_chain.retriever.search_kwargs["filter"] = {
                    "metadata": {
                        "$and": [
                            {"type": {"$eq": "image_context"}},
                            {"image_number": {"$eq": image_number}}
                        ]
                    }
                }
            else:
                # Reset filter for general questions
                self.qa_chain.retriever.search_kwargs["filter"] = None
            
            # Get response
            response = self.qa_chain.invoke({
                "question": question, 
                "chat_history": chat_history
            })
            
            answer = response["answer"]
            
            # Add context about which image was referenced
            if image_number:
                answer = f"Regarding Image {image_number}:\n\n{answer}"
            
            # Format mathematical formulas if present
            if "$$" in answer or "\[" in answer:
                answer = answer.replace("$$", "$")
                answer = answer.replace("\[", "$$").replace("\]", "$$")
                answer = answer.replace("\text", "\operatorname")
            
            return answer
            
        except Exception as e:
            return f"Error generating response: {str(e)}" 