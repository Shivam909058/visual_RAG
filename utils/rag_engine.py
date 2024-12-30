from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

class RAGEngine:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=200
        )
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
        self.vector_store = None
        self.qa_chain = None
        
    def process_text(self, text, image_contexts):
        """Process text and image contexts into a vector store"""
        combined_text = text + "\n\nImage Contexts:\n" + "\n".join(image_contexts)
        chunks = self.text_splitter.split_text(combined_text)
        
        self.vector_store = Chroma.from_texts(
            chunks,
            self.embeddings,
            collection_name="pdf_store"
        )
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=self.vector_store.as_retriever(),
            return_source_documents=True
        )
    
    def ask_question(self, question, chat_history=[]):
        """Ask a question about the processed document"""
        if not self.qa_chain:
            return "Please process a document first."
        
        response = self.qa_chain.invoke({
            "question": question, 
            "chat_history": chat_history
        })
        
        answer = response["answer"]
        
        # Format inline math expressions
        answer = answer.replace("(", "\\(").replace(")", "\\)")
        
        # Format block math expressions
        answer = answer.replace("$$", "\\[").replace("$$", "\\]")
        
        # Special handling for common math expressions
        answer = answer.replace("\\text{softmax}", "\\operatorname{softmax}")
        answer = answer.replace("\\text{Attention}", "\\operatorname{Attention}")
        
        # Add proper spacing and formatting
        answer = answer.replace("\\[", "\n\\[\n").replace("\\]", "\n\\]\n")
        
        return answer 