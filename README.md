# Contextual RAG with ChromaDB

## Overview

This project implements a Contextual Retrieval Augmented Generation (RAG) system using ChromaDB as the vector store. The approach is inspired by Anthropic's research on enhancing RAG systems with better context handling to improve the quality and relevance of AI-generated responses.

## Features

- **Contextual RAG**: Unlike basic RAG implementations, this system considers document context when retrieving relevant information
- **ChromaDB Integration**: Utilizing ChromaDB as an efficient and scalable vector database
- **Multi-model Support**: Designed to work with various embedding and language models
- **Semantic Search**: Enhanced retrieval based on semantic meaning rather than just keyword matching

## How It Works

1. **Document Processing**: Documents are chunked and processed to preserve context
2. **Contextual Embeddings**: Text chunks are embedded while maintaining their contextual relationships
3. **Efficient Retrieval**: ChromaDB enables fast similarity search to find relevant information
4. **Enhanced Generation**: Retrieved context is used to augment the capabilities of large language models

## Installation

```bash
# Clone this repository
git clone https://github.com/Gokul712003/CRAG_Study_assistant.git
cd vector_store_chroma

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- ChromaDB
- Transformers or other embedding libraries
- Access to a language model (local or API-based)

## Acknowledgements

This implementation is inspired by Anthropic's research on contextual RAG systems and best practices for retrieval augmented generation.

## License

MIT
