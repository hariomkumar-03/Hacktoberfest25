# Install required packages:
# pip install transformers torch sentence-transformers chromadb langchain pypdf docx2txt gradio accelerate bitsandbytes

import os
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Tuple
import PyPDF2
import docx2txt

class AdvancedRAGChatbot:
    def __init__(self, model_name="google/gemma-2-2b-it"):
        """Initialize the RAG chatbot with Gemma2 model"""
        print("Initializing Advanced RAG Chatbot...")
        
        # Quantization config for efficient memory usage
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load Gemma2 model
        print("Loading Gemma2 model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Load embedding model for RAG
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB for vector storage
        print("Setting up vector database...")
        self.chroma_client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            is_persistent=False
        ))
        self.collection = self.chroma_client.create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Conversation history
        self.conversation_history = []
        
        print("Initialization complete!")
    
    def extract_text_from_file(self, file_path: str) -> str:
        """Extract text from various file formats"""
        if file_path.endswith('.pdf'):
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
        elif file_path.endswith('.docx'):
            return docx2txt.process(file_path)
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def add_documents(self, file_paths: List[str]) -> str:
        """Process and add documents to the vector database"""
        total_chunks = 0
        
        for file_path in file_paths:
            try:
                # Extract text
                text = self.extract_text_from_file(file_path)
                
                # Split into chunks
                chunks = self.text_splitter.split_text(text)
                
                # Generate embeddings
                embeddings = self.embedding_model.encode(chunks).tolist()
                
                # Add to ChromaDB
                ids = [f"{os.path.basename(file_path)}_{i}" for i in range(len(chunks))]
                metadatas = [{"source": os.path.basename(file_path), "chunk_id": i} 
                           for i in range(len(chunks))]
                
                self.collection.add(
                    embeddings=embeddings,
                    documents=chunks,
                    ids=ids,
                    metadatas=metadatas
                )
                
                total_chunks += len(chunks)
                
            except Exception as e:
                return f"Error processing {file_path}: {str(e)}"
        
        return f"Successfully processed {len(file_paths)} document(s) into {total_chunks} chunks!"
    
    def retrieve_relevant_chunks(self, query: str, n_results: int = 3) -> Tuple[List[str], List[dict]]:
        """Retrieve relevant document chunks for the query"""
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        documents = results['documents'][0] if results['documents'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        
        return documents, metadatas
    
    def _format_history(self) -> str:
        """Format conversation history for prompt"""
        if not self.conversation_history:
            return "No previous conversation."
        
        formatted = []
        for user_msg, bot_msg in self.conversation_history[-3:]:  # Last 3 exchanges
            formatted.append(f"User: {user_msg}\nAssistant: {bot_msg}")
        return "\n\n".join(formatted)
    
    def generate_response(self, query: str, use_rag: bool = True, 
                         max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        """Generate response using Gemma2 with optional RAG"""
        
        # Retrieve relevant context if RAG is enabled
        context = ""
        sources = []
        if use_rag and self.collection.count() > 0:
            documents, metadatas = self.retrieve_relevant_chunks(query, n_results=3)
            if documents:
                context = "\n\n".join([f"[Source: {m['source']}]\n{doc}" 
                                      for doc, m in zip(documents, metadatas)])
                sources = list(set([m['source'] for m in metadatas]))
        
        # Build prompt with context and conversation history
        system_prompt = "You are a helpful AI assistant. Answer questions accurately based on the provided context."
        
        if context:
            prompt = f"""{system_prompt}

Context from documents:
{context}

Previous conversation:
{self._format_history()}

User: {query}
Assistant:"""
        else:
            prompt = f"""{system_prompt}

Previous conversation:
{self._format_history()}

User: {query}
Assistant:"""
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        response = response.split("Assistant:")[-1].strip()
        
        # Add sources if available
        if sources:
            response += f"\n\n📚 Sources: {', '.join(sources)}"
        
        # Update conversation history
        self.conversation_history.append((query, response))
        
        return response
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        return "Conversation history cleared!"
    
    def get_stats(self) -> str:
        """Get chatbot statistics"""
        return f"""
        📊 Chatbot Statistics:
        - Documents in database: {self.collection.count()} chunks
        - Conversation exchanges: {len(self.conversation_history)}
        - Model: Gemma-2-2B-IT (4-bit quantized)
        - Embedding model: all-MiniLM-L6-v2
        """


# ============================================
# Gradio Interface
# ============================================

def create_gradio_interface():
    """Create Gradio web interface"""
    
    # Initialize chatbot
    chatbot = AdvancedRAGChatbot()
    
    def upload_files(files):
        if not files:
            return "No files uploaded."
        file_paths = [f.name for f in files]
        return chatbot.add_documents(file_paths)
    
    def chat(message, history, use_rag, temperature):
        response = chatbot.generate_response(
            message, 
            use_rag=use_rag, 
            temperature=temperature
        )
        return response
    
    def clear():
        chatbot.clear_history()
        return []
    
    def show_stats():
        return chatbot.get_stats()
    
    # Create interface
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 Advanced RAG Chatbot with Gemma2")
        gr.Markdown("Upload documents and chat with an AI that can reference your documents!")
        
        with gr.Tab("💬 Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbox = gr.Chatbot(height=500, label="Conversation")
                    msg = gr.Textbox(
                        placeholder="Ask a question...",
                        label="Your Message",
                        lines=2
                    )
                    with gr.Row():
                        submit = gr.Button("Send", variant="primary")
                        clear_btn = gr.Button("Clear History")
                
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Settings")
                    use_rag = gr.Checkbox(label="Use RAG (Document Context)", value=True)
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature"
                    )
                    stats_btn = gr.Button("Show Statistics")
                    stats_output = gr.Textbox(label="Statistics", lines=6)
        
        with gr.Tab("📁 Document Upload"):
            file_upload = gr.File(
                label="Upload Documents (PDF, DOCX, TXT)",
                file_count="multiple",
                file_types=[".pdf", ".docx", ".txt"]
            )
            upload_btn = gr.Button("Process Documents", variant="primary")
            upload_output = gr.Textbox(label="Upload Status", lines=3)
        
        with gr.Tab("ℹ️ Instructions"):
            gr.Markdown("""
            ## How to Use:
            
            1. **Upload Documents** (Optional):
               - Go to the "Document Upload" tab
               - Upload PDF, DOCX, or TXT files
               - Click "Process Documents"
            
            2. **Start Chatting**:
               - Go to the "Chat" tab
               - Type your question
               - Enable "Use RAG" to include document context
               - Adjust temperature for creativity (lower = more focused)
            
            3. **Features**:
               - Document-aware responses with source citations
               - Conversation history maintained
               - Adjustable generation parameters
               - Statistics and insights
            
            ## Tips:
            - Upload relevant documents for better answers
            - Use lower temperature (0.3-0.5) for factual queries
            - Use higher temperature (0.7-0.9) for creative responses
            """)
        
        # Event handlers
        submit.click(
            chat,
            inputs=[msg, chatbox, use_rag, temperature],
            outputs=[chatbox]
        ).then(lambda: "", None, msg)
        
        msg.submit(
            chat,
            inputs=[msg, chatbox, use_rag, temperature],
            outputs=[chatbox]
        ).then(lambda: "", None, msg)
        
        clear_btn.click(clear, outputs=[chatbox])
        upload_btn.click(upload_files, inputs=[file_upload], outputs=[upload_output])
        stats_btn.click(show_stats, outputs=[stats_output])
    
    return demo


# ============================================
# Main Execution
# ============================================

if __name__ == "__main__":
    print("Starting Advanced RAG Chatbot...")
    demo = create_gradio_interface()
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
