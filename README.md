# Gemma RAG

A lightweight Retrieval-Augmented Generation (RAG) implementation using semantic search and a language model to answer questions from a predefined text collection.

The project uses Sentence Transformers to generate text embeddings, FAISS to retrieve relevant passages, and Hugging Face Transformers for answer generation. The notebook runs in Google Colab and demonstrates the core components of a RAG pipeline.

## Architecture

1. **Text embeddings:** Encode the source text using `all-MiniLM-L6-v2`.
2. **Vector indexing:** Store the embeddings in a FAISS index for similarity search.
3. **Retrieval:** Encode each question and retrieve the closest matching text snippet.
4. **Answer generation:** Pass the retrieved context and question to a language model.
5. **Output:** Display the question, retrieved passage and generated answer.

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- FAISS
- NumPy
- Google Colab

## Example

The notebook uses a short fictional story about Ethan and Fiona to demonstrate question answering.

Example queries:
- Why did Fiona thank Ethan?
- What challenges did Ethan and Fiona face in the forest?
- What tools did Ethan use to navigate?
- Describe the forest.

## Run the Notebook

Open `Gemma_RAG.ipynb` in Google Colab and execute the cells in order. The notebook installs the required Python packages, creates the vector index and runs the example queries.

The original configuration uses Gemma. The current working version uses TinyLlama, which avoids the gated-model access requirement for Gemma.

## Limitations

The knowledge base consists of predefined text snippets, and retrieval returns the single closest match by default. Answer quality therefore depends on the relevance of the retrieved passage and the language model's response.

## Possible Extensions

- Support document and PDF ingestion.
- Retrieve multiple passages per question.
- Evaluate retrieval quality and answer faithfulness.
- Compare embedding models and retrieval strategies.
- Add an interactive interface for querying documents.
