# Import Modules
from llama_index.core.workflow import StartEvent, StopEvent, Event, Workflow, step
from langchain_nvidia_ai_endpoints import NVIDIARerank
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.query_engine import TransformQueryEngine
from langchain_core.documents import Document as LangDocument
from llama_index.llms.nvidia import NVIDIA
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.core import Document as LlamaDocument
from llama_parse import LlamaParse
from llama_index.core import Settings
import nest_asyncio
import os

# Apply Nest Asyc (for LlamaParse)
nest_asyncio.apply()

# Set Env Variables
os.environ["NVIDIA_API_KEY"] = "YOUR_NVIDIA_API_KEY"
nvidia_api_key = os.environ["NVIDIA_API_KEY"]

os.environ["LLAMAPARSE_API_KEY"] = "YOUR_LLAMAPARSE_KEY"
llamaparse_api_key = os.environ["LLAMAPARSE_API_KEY"]

# Initialize parser and models
parser = LlamaParse(api_key=llamaparse_api_key, result_type="markdown", verbose=True)
client = NVIDIA(model="meta/llama-3.1-8b-instruct", api_key=nvidia_api_key, temperature=0.2, top_p=0.7, max_tokens=1024)
embed_model = NVIDIAEmbedding(model="nvidia/nv-embedqa-e5-v5", api_key=nvidia_api_key, truncate="NONE")
reranker = NVIDIARerank(model="nvidia/nv-rerankqa-mistral-4b-v3", api_key=nvidia_api_key,)

# Set the NVIDIA models globally
Settings.embed_model = embed_model
Settings.llm = client

# Initialize all classes
class ParseEvent(Event):
    documents: list

class EmbeddingsEvent(Event):
    all_documents: list
    all_embeddings: list

class IndexCreateEvent(Event):
    index: any

class IndexLoadEvent(Event):
    index: any

class HyDEQueryEvent(Event):
    nodes: list

class ReRankEvent(Event):
    top_ranked_document: str

class LLMSynthEvent(Event):
    response_text: str

# Define your workflow
class RAGWorkflow(Workflow):
    # File Parsing step
    @step
    async def parse_document(self, ev: StartEvent) -> ParseEvent:
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        document_path = os.path.join(script_dir, "Scripts", "PhilDataset.pdf")
        print(f"Parsing document: {document_path}")
    
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"The file {document_path} does not exist.")

        # Load and parse the document
        documents = parser.load_data(document_path)
        if not documents:
            raise ValueError("Document parsing failed or returned empty.")

        print("Document parsed successfully.")
        return ParseEvent(documents=documents)

    # Embeddings generation step
    @step
    async def generate_embeddings(self, ev: ParseEvent) -> EmbeddingsEvent:
        def split_text(text, max_tokens=512):
            words = text.split()
            chunks = []
            current_chunk = []
            current_length = 0

            for word in words:
                word_length = len(word)
                if current_length + word_length + 1 > max_tokens:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                    current_length = word_length + 1
                else:
                    current_chunk.append(word)
                    current_length += word_length + 1

            if current_chunk:
                chunks.append(" ".join(current_chunk))

            return chunks

        all_embeddings = []
        all_documents = []
        total_chunks = 0  # Initialize a counter for the total number of chunks

        for doc in ev.documents:
            print(f"Document structure: {doc}")
        
            if hasattr(doc, 'text'):
                text_chunks = split_text(doc.text)
                total_chunks += len(text_chunks)
                for chunk in text_chunks:
                    embedding = embed_model.get_text_embedding(chunk)
                    all_embeddings.append(embedding)
                    all_documents.append(LlamaDocument(text=chunk))
            else:
                print(f"Document is missing 'text' attribute: {doc}")

        print(f"Total chunks created: {total_chunks}")
        print("Embeddings generated.")
        return EmbeddingsEvent(all_documents=all_documents, all_embeddings=all_embeddings)

    # Index creation step
    @step
    async def create_index(self, ev: EmbeddingsEvent) -> IndexCreateEvent:
        index = VectorStoreIndex.from_documents(ev.all_documents, embeddings=ev.all_embeddings, embed_model=embed_model)
        index.set_index_id("vector_index")
        index.storage_context.persist("./storage")
    
        print("Index created.")
        return IndexCreateEvent(index=index)
    
    # Loading index with vectors and docs
    @step
    async def load_index(self, ev: IndexCreateEvent) -> IndexLoadEvent:
        storage_context = StorageContext.from_defaults(persist_dir="storage")
        index = load_index_from_storage(storage_context, index_id="vector_index")
    
        print("Index loaded.")
        return IndexLoadEvent(index=index)
    
    # Step for HyDE Query Transformation and Top-k node retrieval
    @step
    async def hydequery_topk(self, ev: IndexLoadEvent) -> HyDEQueryEvent:
        hyde = HyDEQueryTransform(include_original=True)
        query_engine = ev.index.as_query_engine()
        hyde_query_engine = TransformQueryEngine(query_engine, hyde)
    
        # Define the question you want to query
        question = "Which course did Philemon study in campus?"
    
        # Generate a hypothetical document using HyDE
        hyde_response = hyde_query_engine.query(question)
        print(f"HyDE Response: {hyde_response}")

        hyde_query = hyde_response.response if not isinstance(hyde_response, str) else hyde_response

        retriever = ev.index.as_retriever(similarity_top_k=3)
        nodes = retriever.retrieve(hyde_query)

        for node in nodes:
            print(node)
    
        return HyDEQueryEvent(nodes=nodes)
    
    # Step for reranking the retrieved nodes 
    @step
    async def nodes_rerank(self, ev: HyDEQueryEvent) -> ReRankEvent:
        # Define the question
        question = "Which course did Philemon study in campus?"

        # Rerank the retrieved documents
        ranked_documents = reranker.compress_documents(
            query=question,
            documents=[LangDocument(page_content=node.text) for node in ev.nodes]
        )

        # Print the most relevant and least relevant node
        print(f"Most relevant node: {ranked_documents[0].page_content}")

        # Return the top-ranked document
        top_ranked_document = ranked_documents[0].page_content

        return ReRankEvent(top_ranked_document=top_ranked_document)
    
    # Step for passing the top ranked node as context for the LLM
    @step
    async def llm_synthesis(self, ev: ReRankEvent) -> LLMSynthEvent:
        # Define the question
        question = "Which course did Philemon study in campus?"

        # Use the top-ranked document as context
        context = ev.top_ranked_document

        # Construct the messages using the ChatMessage class
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=context),
            ChatMessage(role=MessageRole.USER, content=str(question))
        ]

        # Call the chat method to get the response
        completion = client.chat(messages)

        # Process response - assuming completion is a single string or a tuple containing a string
        response_text = ""

        if isinstance(completion, (list, tuple)):
            # Join elements of tuple/list if it's in such format
            response_text = ' '.join(completion)
        elif isinstance(completion, str):
            # Directly assign if it's a string
            response_text = completion
        else:
            # Fallback for unexpected types, convert to string
            response_text = str(completion)
    
        # Clean up response text
        response_text = response_text.replace("assistant:", "Final Response:").strip()

        print(response_text)

        return LLMSynthEvent(response_text=response_text)


    # Implement the StopEvent to end the workflow
    @step
    async def end_workflow(self, ev: LLMSynthEvent) -> StopEvent:
        print("Workflow completed.")
        return StopEvent()



# Set up the main function to pass the question
async def main():
    # Define the question you want to query
    question = "Which course did Philemon study in campus?"

    # Create an instance of your workflow
    rag_workflow = RAGWorkflow(timeout=100, verbose=True)

    # Run the workflow with the question passed as an argument
    result = await rag_workflow.run(question=question)
    print(result)

# Run the main function
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

