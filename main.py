from langchain.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI
import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in the .env file.")

# Define the tools
@tool
def search_pdf(query: str, pdf_path: str) -> str:
    """Search a PDF document for the query."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    openai_llm = OpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm=openai_llm, chain_type="stuff")
    return chain.run(input_documents=texts, question=query)

@tool
def ask_openai(query: str) -> str:
    """Ask the OpenAI LLM directly for the answer using the OpenAI API."""
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": query}],
        "max_tokens": 200
    }
    
    # Make the API request to the OpenAI API
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=data
    )
    
    # Comment out or remove these print statements to avoid extra information in the output
    # print(f"OpenAI response status code: {response.status_code}")  # Debugging response status
    # print(f"OpenAI response text: {response.text}")  # Debugging response text
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        return f"Error: {response.status_code}, {response.text}"

# Define the manager agent
class ManagerAgent:
    def __init__(self):
        self.tools = {
            "PDF": search_pdf,
            "OpenAI": ask_openai
        }

    def decide_tool(self, query: str) -> str:
        """Decide which tool to use based on the query."""
        decision_prompt = f"""You are a manager agent. Your task is to decide which tool to use for the following query:
        Query: {query}
        Options:
        1. PDF - Use this for queries related to a specific PDF document.
        2. OpenAI - Use this for general knowledge questions.
        Decision:"""
        
        decision_response = self.tools["OpenAI"].invoke(decision_prompt)  # Use OpenAI to decide the tool
        # Comment out or remove this print statement to avoid extra information in the output
        # print(f"Decision response: {decision_response}")  # Debugging decision response
        if "PDF" in decision_response:
            return "PDF"
        elif "OpenAI" in decision_response:
            return "OpenAI"
        else:
            return "Unable to determine the appropriate tool. Please rephrase your query."

    def execute_query(self, query: str) -> str:
        """Execute the query using the appropriate tool."""
        pdf_path = r"C:\Users\hp\Desktop\tavily\paper02.pdf"  # Replace with the actual PDF path
        decision = self.decide_tool(query)
        
        if decision == "PDF":
            return self.tools["PDF"].invoke(query, pdf_path)
        elif decision == "OpenAI":
            return self.tools["OpenAI"].invoke(query)
        else:
            return decision

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