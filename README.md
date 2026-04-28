# ML Course Document RAG

A trust-first Retrieval-Augmented Generation system that answers questions strictly from uploaded ML course documents using similarity gating, citation-based responses, and refusal handling.

This project is designed for learning environments where answers must stay grounded in the provided course material instead of relying on general LLM knowledge.

---

## Overview

Large Language Models are powerful, but they can sometimes produce answers that are outside the syllabus, unsupported by uploaded material, or difficult to verify.

This project solves that problem by building a strict document-grounded RAG system for machine learning course content. Users can upload course PDFs, ask questions, and receive answers only when the system finds enough relevant evidence in the uploaded documents.

If the retrieved context is not strong enough, the system refuses to answer instead of guessing.

> Not enough information in the uploaded documents.

The goal is to make AI-assisted learning more reliable, auditable, and aligned with instructor-provided material.

---

## Problem Statement

Generic LLMs often create problems in academic and enterprise knowledge settings:

- They may answer using general internet knowledge instead of provided documents
- They can introduce concepts not covered in the course material
- They may hallucinate facts while sounding confident
- They often do not provide clear source traceability
- They can make it difficult for students or instructors to verify correctness

This project addresses those issues by enforcing a source-grounded answer pipeline.

---

## Key Features

- Upload and process ML course PDFs
- Extract and chunk document text
- Generate embeddings for semantic similarity search
- Retrieve the most relevant document chunks for a user question
- Apply similarity threshold gating before answering
- Generate answers only from retrieved document context
- Provide citation-based responses for traceability
- Refuse to answer when uploaded documents do not contain enough information
- Designed for future multi-LLM council fallback and judge-based evaluation

---

## Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python |
| App Framework | Streamlit |
| LLM / API | OpenAI |
| RAG Pipeline | LangChain-style retrieval workflow |
| Vector Search | FAISS / Vector similarity search |
| Document Processing | PDF text extraction |
| Embeddings | OpenAI Embeddings |
| Environment | pip, virtual environment |
| Deployment Ready | Render-compatible runtime setup |

---

## System Workflow


    User Uploads PDF Documents
        ↓
    PDF Text Extraction
        ↓
    Text Chunking
        ↓
    Embedding Generation
        ↓
    Vector Store Creation
        ↓
    User Asks a Question
        ↓
    Similarity Search
        ↓
    Threshold Gate
        ↓
    Answer with Citations OR Refusal

## Architecture

                ┌──────────────────────┐
                │   Uploaded PDFs       │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │  Text Extraction      │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │  Chunking Pipeline    │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Embedding Generation  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Vector Store / FAISS  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Similarity Retrieval  │
                └──────────┬───────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
     Enough Relevant Context       Low Similarity Score
              │                         │
              ▼                         ▼
     Answer with Citations       Refuse to Answer

## Why Strict RAG?

This project intentionally avoids answering when the uploaded documents do not support the question.

Most chatbot systems try to answer everything. That can be risky in education because students may receive convincing but unsupported explanations.

This project follows a stricter principle:

No retrieved evidence = no answer.

This makes the system useful for:

Course-specific learning assistants
Academic document Q&A
Internal training material search
Enterprise knowledge bases
Research document assistants
Compliance-sensitive AI applications
Example Use Case
Uploaded Documents
Machine Learning lecture PDFs
Course notes
Assignment reference material
Textbook chapters
User Question
What is the difference between supervised and unsupervised learning?
System Behavior

If the concept exists in the uploaded course documents, the system answers using the retrieved context and includes citations.

If the concept is not found with enough confidence, the system responds:

Not enough information in the uploaded documents.
Example Output
Supervised learning uses labeled data where the model learns from input-output pairs.
Unsupervised learning uses unlabeled data and identifies patterns or structures in the data.

**Sources:**
[Document 1, Page 4]
[Document 2, Page 7]
Project Structure
ml-course-document-rag/
│
├── app/
│   └── Streamlit application files
│
├── data/
│   └── Sample or uploaded document data
│
├── src/
│   └── Core RAG pipeline logic
│
├── requirements.txt
├── runtime.txt
└── README.md

## Getting Started

**1. Clone the Repository**
git clone https://github.com/Shreevikas-BJ/ml-course-document-rag.git
cd ml-course-document-rag

**2. Create a Virtual Environment**
python -m venv venv

Activate the environment:

**Windows**

venv\Scripts\activate

**macOS / Linux**

source venv/bin/activate

**3. Install Dependencies**
pip install -r requirements.txt

## Environment Variables

Create a .env file in the project root and add your API key:

OPENAI_API_KEY=your_openai_api_key_here

Do not commit your .env file to GitHub.

## Run the Application

streamlit run app/app.py

If your main Streamlit file has a different name, update the command accordingly.

## Core Design Principles
**1. Document-Grounded Answers**

The system only answers from uploaded PDFs and does not rely on general model knowledge.

**2. Similarity Threshold Gating**

Before generating an answer, the system checks whether the retrieved chunks are relevant enough.

**3. Citation-First Output**

Answers include source references so users can verify where the information came from.

**4. Refusal Over Hallucination**

If the system cannot find enough evidence, it refuses to answer instead of making assumptions.

**5. Future Council-Based Reasoning**

The project is designed to support a future fallback mode where multiple LLMs can evaluate and critique responses before a final answer is selected.

## Phase 2 Roadmap: LLM Council Fallback

The next planned version introduces a controlled fallback mechanism inspired by a multi-model evaluation approach.

When strict RAG cannot answer a question, the system can optionally route the query to multiple LLMs for reasoning and comparison.

Planned flow:

    Strict RAG Fails
          ↓
    Send Query to Multiple LLMs
          ↓
    Each LLM Generates an Answer
          ↓
    Models Critique and Score Responses
          ↓
    Chairman LLM Selects Best Final Answer
          ↓
    Final Response with Confidence Notes

**Potential models:**

OpenAI
Gemini
Grok
Ollama / Local LLMs

This fallback would be clearly separated from strict RAG mode so users know whether the answer came from uploaded documents or general reasoning.

## Future Improvements

Add page-level PDF citation extraction
Add support for DOCX, PPTX, and TXT files
Store document embeddings persistently
Add user authentication
Add multi-document filtering by course, topic, or module
Add evaluation metrics for retrieval quality
Add hallucination checks using a judge model
Add local embedding model support
Add Docker deployment
Add hosted demo using Streamlit Cloud or Render
Implement Phase 2 LLM Council fallback

## Business and Academic Value

This project demonstrates how RAG systems can be designed for trust-sensitive environments. Instead of building a general chatbot, it focuses on grounding, verification, and controlled response behavior.

This is especially useful in domains where correctness matters, such as:

Education
Enterprise knowledge management
Internal policy search
Research assistance
Legal and compliance workflows
Technical documentation support

## Author

**Shreevikas Bangalore Jagadish**
Graduate Student, Information Technology and Management
Illinois Institute of Technology

GitHub: Shreevikas-BJ
LinkedIn: shreevikasbj
Portfolio: datascienceportfol.io/shreevikasbj

## Repository

https://github.com/Shreevikas-BJ/ml-course-document-rag
