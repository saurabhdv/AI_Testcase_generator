# AI_Testcase_generator

# Generate test cases that are relevant to my product 

## Tools used 
# Core Language & Frameworks
•	  Python: The main programming language used to build and orchestrate the entire RAG pipeline.
•	  LangChain: The Orchestrator A powerful framework LangChain simplifies the complex choreography between retrieval, generation, and prompt engineering. It allowed me to focus on logic and flow rather than reinventing the wheel for chaining LLM calls.
# Model & API Providers
•	OpenAI: Cloud-based access to OpenAI models via Microsoft Azure.
•	Text-embedding-ada-002: Used for converting input text into high-dimensional vectors for semantic search.
•	Gpt-35-turbo: generates test cases that are not just syntactically correct but contextually relevant—thanks to the retrieved knowledge.
# Retrieval System
•	FAISS: Facebook AI Similarity Search – a high-performance vector database used to store and retrieve embeddings efficiently. Enables fast semantic search over large document sets.


for documents in docs folder creating the vector can also be aded to azure open ai vector module
link the open AI to access AI Models in RAG_CLIENT 
# Set Azure OpenAI credentials
AZURE_ENDPOINT = "https://<..>.openai.azure.com/"
AZURE_API_KEY = "......."
AZURE_API_VERSION = "..."

Run the code to experience AI generated results 
# python rag_client.py

Enjoy exploring power of AI;)
