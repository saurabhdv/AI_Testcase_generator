import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredHTMLLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

# Note: Ensure you have the necessary permissions and configurations to run this script.    
# This script is designed to be run in an environment with access to Azure OpenAI services.
# Adjust the paths and configurations as necessary for your setup. 
# This script loads documents from a specified folder, splits them into chunks,
# creates embeddings using Azure OpenAI, and sets up a Retrieval-QA chain to generate test cases.
# Ensure you have the required packages installed:
# pip install langchain langchain-community langchain-openai faiss-cpu  
# Set the environment variable for Azure OpenAI API key


# Set Azure OpenAI credentials
AZURE_ENDPOINT = "https://oooo.openai.azure.com/"
AZURE_API_KEY = "oooooooooooo"
AZURE_API_VERSION = "2025-01-01-preview"
EMBEDDING_DEPLOYMENT = "text-embedding-ada-002"

# Load and split documents
### Comment this section if not needed multiple files ###

# Define the folder path Change this to your folder path
doc_folder = Path("\\docs")
# Ensure the folder exists
if not doc_folder.exists():
    raise FileNotFoundError(f"The folder {doc_folder} does not exist.")

# Initialize a list to hold all documents
all_docs = []   
# Check if the folder is empty
if not any(doc_folder.iterdir()):
    raise ValueError(f"The folder {doc_folder} is empty. Please add some documents to process.")

# Loop through all files in the folder  
for file_path in doc_folder.glob("*"):
    if file_path.suffix == ".pdf":
          loader = PyPDFLoader(str(file_path))
    elif file_path.suffix == ".txt":
          loader = TextLoader(str(file_path), encoding="utf-8")
    elif file_path.suffix == ".html":
          loader = UnstructuredHTMLLoader(str(file_path))
    else:
          continue    # Skip unsupported files
    docs = loader.load()
    all_docs.extend(docs)

# If you want to load a single file, uncomment the following lines and comment out the loop above
###  Single File ### 
#loader = TextLoader(r"C:\WORK-CM\hackday_cm\docs\RELEASE.TXT")

# Split the documents into chunks 
# # Now `split_docs` is ready for Azure OpenAI ingestion
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(all_docs)

# Create Embeddings and vector store
# Initialize Azure OpenAI Embeddings
# Ensure you have the necessary environment variables set for Azure OpenAI API key  
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    deployment=EMBEDDING_DEPLOYMENT
)

# Create a vector store from the document chunks
# This will create a FAISS vector store from the document chunks using the embeddings
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

# Set the retriever parameters
retriever.search_kwargs = {
    "k": 5,  # Number of documents to retrieve
    "filter": None,  # No specific filter applied
    "score_threshold": 0.5  # Minimum score threshold for relevance
}

# Initialize the Azure OpenAI LLM
AzureChatOpenAIllm = AzureChatOpenAI(  
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
    deployment_name="gpt-35-turbo",
    api_version=AZURE_API_VERSION,
   # api_version="2024-12-01-preview",

    model_name="gpt-35-turbo",  # Specify the model name
    max_tokens=1000,  # Maximum tokens for the response 
    # Adjust the temperature for response variability
    # Lower temperature for more deterministic responses, higher for more creative responses
    temperature= 0.7,
    # Temperature controls the randomness of the response generation
    # A value of 0.0 makes the model deterministic, while higher values increase randomness 
    # Top-p and top-k sampling parameters for response generation
    # Top-p sampling controls the cumulative probability of token selection
    # Top-k sampling limits the number of tokens to consider for each step
    # top_p=0.9,  # Top-p sampling for response generation 
    # top_k=50,  # Top-k sampling for response generation
    # stop=None,  # No specific stop sequence defined
    # n=1,   # Number of responses to generate
    # model_kwargs={},  # Additional model parameters if needed

    
    # best_of=1,   # Number of best responses to consider
    # logit_bias=None,  # Logit bias for response generation
    # user=None,  # User identifier for the request
    # system=None,  # System message for the request
    # logprobs=None,  # Log probabilities for response generation
    # echo=False,  # Whether to echo the input in the response

    # stop_sequences=None,  # Stop sequences for response generation
    # temperature_decay=None,  # Temperature decay for response generation
    
    # stream=False,  # Whether to stream the response

    # presence_penalty=0.0,  # Presence penalty for response generation
    # frequency_penalty=0.0,  # Frequency penalty for response generation
    # user_id=None,  # User ID for the request
    # response_model=None,  # Model for the response
    # response_format=None,  # Format of the response

    # seed=None,  # Random seed for response generation
    # max_tokens_per_request=4096,  # Maximum tokens per request
    # max_tokens_per_minute=60,  # Maximum tokens per minute
    # max_tokens_per_second=1,  # Maximum tokens per second
    # max_tokens_per_minute_per_user=60,  # Maximum tokens per minute per user
    # max_tokens_per_second_per_user=1,  # Maximum tokens per second per user
    # max_tokens_per_minute_per_model=60,  # Maximum tokens per minute per model
    # max_tokens_per_second_per_model=1,  # Maximum tokens per second per model

    # max_retries=3,  # Maximum retries for the request
    # timeout=30,  # Timeout for the request in seconds
    # request_timeout=60,  # Request timeout in seconds
    # retry_on_timeout=True,  # Whether to retry on timeout

    # log_level="info",  # Logging level for the request
    # frequency_penalty=0.0,  # Frequency penalty for response generation 
    # presence_penalty=0.0,  # Presence penalty for response generation 

    # response_format=None ,  # Format of the response
    # response_model=None,  # Model for the response 
    )   

# Ensure the necessary environment variables are set
if not all([AZURE_ENDPOINT, AZURE_API_KEY, AZURE_API_VERSION, EMBEDDING_DEPLOYMENT]):
    raise ValueError("Azure OpenAI credentials are not set. Please check your environment variables.")
# Ensure the LLM is properly initialized
if AzureChatOpenAIllm is None:
    raise ValueError("Failed to initialize the Azure OpenAI LLM. Please check your credentials and configuration.")     

# Set up the Retrieval-QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=AzureChatOpenAIllm,
    retriever=retriever,
    return_source_documents=False
)   
# Ensure the QA chain is properly initialized
if qa_chain is None:
    raise ValueError("Failed to initialize the Retrieval-QA chain. Please check your LLM and retriever configuration.") 

# Ask for test cases
print("Generating AI-Generated Manual Test Cases...\n")
query = "suggest 1 test scenarios for PKI CM testing scenarios" 
#query = "suggest steps to install cm server in linux "
#response = qa_chain.invoke({"query": query})
response_1 = qa_chain.invoke(query)
   
# Output the results
print("\nAI-Generated Manual Test Cases:\n")
print(response_1)  

# Ask for Playwright test cases based on the generated scenarios
print("\n Momemt... Generating Playwright Test Cases...\n")
query = "generate playwright test cases for the above scenarios "+ response_1.toString()
#query = "generate playwright test cases for the above scenarios "+ response_1
response_2 = qa_chain.invoke(query)
# Output the results
print("\nAI-Generated Playwright Test Cases:\n")
print(response_2)

""" # Save the results to a text file
output_file = doc_folder / "generated_test_cases.txt" 

with open(output_file, "w", encoding="utf-8") as f:
    f.write("AI-Generated Manual Test Cases:\n")
    f.write(response + "\n\n")
    f.write("AI-Generated Playwright Test Cases:\n")
    f.write(response + "\n")
print(f"\nTest cases saved to {output_file}") """

# End of the script

 
