Description:
FinInsight is a scalable and modular financial intelligence platform that leverages Big Data frameworks and state-of-the-art NLP models to transform unstructured financial information into actionable insights. Developed as part of the CIS-5570 Introduction to Big Data course at the University of Michigan-Dearborn, the project integrates distributed data processing, entity recognition, sentiment analysis, dense vector search, and large language models to answer complex financial queries.

Key Features:
Multi-source data ingestion using PySpark from SEC filings (CSV), CNBC articles (JSON), and Reddit posts (via PRAW).

NLP enrichment with Spark NLP for Named Entity Recognition (NER) and sentiment tagging.

Semantic embeddings generated using SentenceTransformers (all-MiniLM-L6-v2) for high-quality text representation.

Efficient document retrieval with FAISS (vector similarity search).

Retrieval-Augmented Generation (RAG) using LangChain and google/flan-t5-base to provide grounded, context-aware responses to user queries.

Fully modular architecture designed for scalability and real-time financial analysis.

Tech Stack:
PySpark for distributed data processing

Spark NLP for NER & sentiment analysis

SentenceTransformers for text embeddings

FAISS for vector indexing & retrieval

LangChain + FLAN-T5 for RAG-based QA system

Data Sources: SEC EDGAR, CNBC, Reddit (r/wallstreetbets)

Use Cases:
Real-time analysis of market sentiment

Financial trend discovery from news and social media

Natural language interface for financial data exploration

Educational platform for finance students and researchers

Evaluation Summary:
NER accuracy ~91%

Sentiment classification ~85%

Query response latency ~2–6 seconds

Document relevance accuracy ~87%
