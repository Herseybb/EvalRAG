# EvalRAG

**Embedding Model Evaluation for Retrieval-Augmented Generation (RAG)**

## 📌 Overview
This project aims to benchmark different text embedding models within a **Retrieval-Augmented Generation (RAG)** framework.  
By comparing retrieval and answer generation performance across multiple datasets, the project provides insights into the strengths and weaknesses of various embeddings in real-world QA scenarios.
This is a **self-learning project** designed to explore the research process in NLP.

## 🎯 Objectives
- Build a reproducible RAG evaluation framework  
- Compare multiple embedding models (e.g., `all-MiniLM-L6-v2`, `multi-qa-mpnet-base-dot-v1`)  
- Evaluate performance on standard datasets 
- Metrics: Recall@k, MRR, Answer F1, Human evaluation (TBD)

## 📂 Project Structure (TBD)
EvalRAG/
│── data/ # Datasets
│── notebooks/ # Experiment notebooks
│── README.md # Project documentation

## Data Source
data used come from repo: https://github.com/yixuantt/MultiHop-RAG/tree/main

## 📝 Self-Learning Log
Date/Phase | What I Did | What I Learned | Next Steps | Things to Explore Further
2025-09-22 | Found a prepared dataset (mentioned above); Created a basic RAG infrastructure which can split chunks, create index, and generate answers based on llm; | basic steps for RAG | create metrics and infrastructure for evaluation of results | theory knowledge for transformers and FAISS



