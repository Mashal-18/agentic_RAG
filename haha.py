from langchain.tools import tool
from tavily import TavilyClient  # Updated import statement
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from transformers import pipeline
import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up API keys
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Instantiate TavilyClient
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# Define the tools
@tool
def search_tavily(query: str) -> str:
    """Use Tavily to search the web for the query."""
    context = tavily_client.get_search_context(query=query)
    return context

@tool
def search_pdf(query: str, pdf_path: str) -> str:
    """Search a PDF document for the query."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # Use Hugging Face's model for Q&A
    hf_qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
    
    answers = []
    for doc in texts:
        answer = hf_qa_pipeline(question=query, context=doc.page_content)
        answers.append(answer["answer"])
    
    return " ".join(answers)

@tool
def ask_huggingface(query: str) -> str:
    """Ask Hugging Face's model directly for the answer."""
    hf_pipeline = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", token="<your_huggingface_token>")
    
    # Add a strict instruction to return only one word
    formatted_query = f"{query}\n\nRespond with only one word: Tavily, PDF, or HuggingFace."
    
    response = hf_pipeline(formatted_query, max_length=10, do_sample=False)
    
    # Extract and clean up response
    generated_text = response[0]["generated_text"].strip().split("\n")[0]  # Only take first word
    
    if "Tavily" in generated_text:
        return "Tavily"
    elif "PDF" in generated_text:
        return "PDF"
    elif "HuggingFace" in generated_text:
        return "HuggingFace"
    
    return "HuggingFace"  # Default fallback

# Define the manager agent
class ManagerAgent:
    def __init__(self):
        self.tools = {
            "Tavily": search_tavily,
            "PDF": search_pdf,
            "HuggingFace": ask_huggingface
        }

    def decide_tool(self, query: str) -> str:
        """Decide which tool to use based on the query using Hugging Face's LLM."""
        decision_prompt = f"""
        You are a tool selection AI. Choose the best tool for the query.

        Query: {query}

        Options:
        - "Tavily" (for web search)
        - "PDF" (for document-based queries)
        - "HuggingFace" (for general knowledge)
        
        Reply with ONLY the tool name.
        """
        
        response = ask_huggingface(decision_prompt).strip()

        # Ensure valid selection
        valid_tools = ["Tavily", "PDF", "HuggingFace"]
        for tool in valid_tools:
            if tool.lower() in response.lower():
                return tool  # Return the exact tool name

        return "HuggingFace"  # Default fallback

    def execute_query(self, query: str) -> str:
        """Execute the query using the appropriate tool."""
        pdf_path = r"C:\Users\hp\Desktop\tavily\paper02.pdf"  # Ensure this path exists
        decision = self.decide_tool(query)
        
        print(f"Selected tool: {decision}")  # Debugging log
        
        if decision == "Tavily":
            return self.tools["Tavily"](query)
        elif decision == "PDF":
            return self.tools["PDF"](query, pdf_path)
        elif decision == "HuggingFace":
            return self.tools["HuggingFace"](query)
        else:
            return "Error: Unable to determine the appropriate tool. Please rephrase your query."

# Example usage
if __name__ == "__main__":
    manager = ManagerAgent()
    print("Welcome to the Multi-Agent RAG System!")
    while True:
        query = input("Enter your query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            print("Exiting the system. Goodbye!")
            break
        answer = manager.execute_query(query)
        print(f"Answer: {answer}\n")
