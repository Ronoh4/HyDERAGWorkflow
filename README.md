RAG Workflow with LlamaIndex
Overview
This repository showcases a robust implementation of a Retrieval-Augmented Generation (RAG) pipeline using LlamaIndex workflows. The workflow efficiently manages complex AI tasks by chaining multiple steps, each triggered by specific events.

Workflow Description
The RAGWorkflow class demonstrates a comprehensive pipeline to handle document processing and querying tasks:

Document Parsing: The workflow begins by parsing a PDF document to extract text data using the LlamaParse library.
Embedding Generation: Parsed text is split into chunks, and embeddings are generated using NVIDIA's embedding model.
Index Creation: The embeddings and corresponding documents are indexed using VectorStoreIndex.
Index Loading: The workflow loads the previously created index from storage.
HyDE Query Transformation: Queries are transformed using the HyDEQueryTransform to enhance retrieval accuracy.
Re-ranking: Retrieved documents are re-ranked to identify the most relevant content using NVIDIA's reranking model.
LLM Synthesis: The top-ranked document is used as context for a language model, which generates a final response based on a user-defined question.
Completion: The workflow concludes with the StopEvent, signaling the end of the processing.
Features
Event-Driven: Each step of the workflow is triggered by specific events, making it adaptable and easy to manage.
Integration: Utilizes NVIDIA and LlamaIndex models for embedding, reranking, and querying tasks.
Visualization: Includes a visual representation of the workflow, providing clarity on the execution flow.
Setup
Ensure you have the required API keys for NVIDIA and LlamaParse, and set them in the environment variables. Install necessary dependencies and run the main function to execute the workflow.

Visualization
An HTML file (ragworkflow_visual.html) is generated to visualize the workflow, illustrating the step-by-step execution.
