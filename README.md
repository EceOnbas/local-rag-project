# Local RAG AI Assistant (On-Device Syllabus Q&A)

A fully offline, privacy-first Retrieval-Augmented Generation (RAG) assistant designed to answer academic syllabus queries with strict adherence to source documents and verified page citations.

[![Watch the Demo]](https://youtu.be/qUy3_iyXVEE)
*Click the link to watch the live video demo.*
---

## 🎯 Project Goal

The primary goal of this project is to provide students with a reliable, **100% offline AI assistant** that answers questions regarding course syllabi (schedules, instructors, grading policies, prerequisites, and ECTS credits) using a **Retrieval-Augmented Generation (RAG)** architecture.

Unlike generic LLMs that rely solely on parametric memory, this system uses an augmented RAG pipeline:

* **Grounded Retrieval:** When a user asks a question, the system queries a local SQLite database using **Hybrid Search (BM25 + Cosine Similarity)** to retrieve the top 6 most relevant syllabus passages.
* **Augmented Generation:** The retrieved passages are passed as explicit context to a local **Phi-3.5-mini** LLM, which is strictly constrained to generate responses based *only* on the provided text.
* **Zero Hallucinations & Direct Citations:** If information is absent from the syllabus, the model explicitly states that it does not know. Every generated statement includes inline citations pointing directly to the source file and page number (`[Source: File | Page: X]`).
* **100% On-Device & Private:** Powered by the **Microsoft Foundry Local SDK**, all pipeline stages—ingestion, embedding, hybrid retrieval, and inference—execute locally on the user's hardware without cloud dependencies or API costs.

---

## ⚙️ How It Works (System Architecture)

The system operates across a 5-stage pipeline:

1. **Ingestion & Indexing (`ingestion.py`):**
* Source PDF files in the `docs/` directory are processed using `pdfplumber` to extract both narrative text and layout-sensitive table structures.
* Documents are split into semantic chunks (max 1,200 characters with a 200-character overlap) and tagged with metadata (Course Code, Title, Page Number).
* Chunks are embedded locally via `qwen3-embedding-0.6b` and indexed into a local SQLite database (`rag_database.db`). *This step runs only once or when source documents update.*


2. **Deterministic Structural Routing:**
* Before running standard semantic retrieval, query intent is checked for deterministic structural signals (e.g., exact course codes, faculty tags, or policy keywords). Direct database filters execute when exact matching yields complete accuracy over similarity thresholds.


3. **Hybrid Search & RRF Re-ranking (`query_app.py`):**
* If semantic search is required, the user query is embedded using `qwen3-embedding-0.6b`.
* The system runs **Sparse Search (BM25)** for keyword precision and **Dense Search (Cosine Similarity)** for semantic intent simultaneously.
* Scores are fused using **Reciprocal Rank Fusion (RRF)** ($k=60$) to select the optimal candidate passages (`top_k=6`). Candidate scores below strict relative threshold limits are dynamically pruned.


4. **Context Assembly & Strict Generation:**
* Retracted chunks are formatted and supplied to `Phi-3.5-mini` alongside a strict system prompt prohibiting out-of-context extrapolation.
* Small secondary models (if invoked for classification) handle task-specific tagging (e.g., mapping user inputs to predefined faculty aliases) rather than rewriting source text to prevent hallucination risks.


5. **Streaming Response Output:**
* The assistant streams the generated answer directly to the terminal or web interface token-by-token alongside metadata-backed citations (`[Source: File | Page: X]`).



---

## 💻 Installation & Usage Guide

### 📋 Prerequisites

* **Operating System:** Windows 10/11, macOS (12+), or Linux
* **Python:** Version **3.10** or higher
* **RAM:** Minimum **8 GB** (16 GB recommended)
* **Storage:** At least **10 GB** free space (for model weights and local database)
* **Microsoft Foundry Local:** Installed and accessible on your machine

---

### 📥 1. Clone the Repository

```bash
git clone https://github.com/your-username/local-rag-project.git
cd local-rag-project

```

---

### 📦 2. Install Dependencies

Install the required packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt

```

> **Windows Users:** If native C++ compilation errors occur, ensure **Microsoft C++ Build Tools** is installed on your system.

---

### 📄 3. Add Course Syllabi (PDFs)

Place your academic course syllabus PDFs inside the `docs/` directory:

```bash
mkdir -p docs

```

*Example files:* `docs/Syllabus-CS300.pdf`, `docs/Syllabus-CS306.pdf`

---

### ⚙️ 4. Data Ingestion & Indexing

Parse the PDFs, compute vector embeddings, and build the local SQLite database:

```bash
python ingestion.py

```

* Loads `qwen3-embedding-0.6b` via Foundry Local.
* Extracts text and tables, generates semantic chunks, computes embeddings, and populates `rag_database.db`.
* **Run this step only once** (or whenever PDFs in `docs/` are modified).

---

### 💬 5. Launch the Q&A Assistant

Start the interactive CLI interface:

```bash
python query_app.py

```

* **First Run:** Downloads `Phi-3.5-mini` weights via Microsoft Foundry Local (requires internet connection once).
* **Subsequent Runs:** Loads cached model weights directly from disk and runs **100% offline**.

Once initialized, ask your questions at the prompt:

```text
--- Optimized RAG System Ready (Phi-3.5-mini | top_k=6) ---
Question: On which days and at what times does CS300 take place, and in which classrooms?

```

Type **`quit`** to exit and safely unload models from RAM.

---

## 🎯 Architectural Decisions (Why This Stack?)

* **Hybrid Search (BM25 + Dense Vector) + RRF:** Pure vector search frequently fails on exact codes (e.g., `CS306`, `UC-G030`). BM25 guarantees keyword accuracy, Cosine Similarity provides semantic intent, and Reciprocal Rank Fusion (RRF) merges rankings fairly.
* **`Phi-3.5-mini` (Chat LLM):** At 3.8B parameters, it balances lightweight memory usage with exceptional prompt instruction compliance, guaranteeing citation formatting without cloud GPUs.
* **`qwen3-embedding-0.6b` (Embeddings):** A lightweight 0.6B parameter model delivering fast vector generation on local CPUs.
* **`pdfplumber` (Parsing):** Preserves table alignments and spatial structure better than standard PDF text extractors, avoiding corrupted class schedules.
* **SQLite (Storage):** Zero-configuration, serverless single-file storage that handles hundreds of local chunk embeddings with sub-millisecond retrieval times.

---

## ⚠️ Limitations & Weaknesses

* **In-Memory Vector Search:** Cosine similarity is computed in Python over in-memory arrays. Highly efficient for small document sets (hundreds of chunks), but requires switching to a dedicated extension (e.g., `sqlite-vec`) for thousands of files.
* **Regex Metadata Dependency:** Metadata extraction relies on standardized course code patterns. Non-standard document headers may default to `"UNKNOWN"`.
* **Fixed-Size Chunk Boundaries:** Splitting text strictly at 1,200 characters may occasionally split lengthy multi-page prerequisite policies across chunk boundaries.
* **Stylized PDF Parsing:** PDFs containing complex nested tables, merged cells, or scanned image text can cause extraction alignment artifacts.

---

## 📁 Project Structure

```text
local-rag-project/
├── docs/                      # Course syllabus PDFs directory
│   ├── Syllabus-CS300.pdf
│   └── Syllabus-CS306.pdf
├── .gitignore                 # Specifies intentionally untracked files to ignore
├── ingestion.py               # Extraction, table parsing, chunking, and vector embedding pipeline
├── query_app.py               # Hybrid search (BM25 + Dense) with RRF and Phi-3.5-mini inference
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies list

```

> **Note:** The local database (`rag_database.db`), Python virtual environment (`venv/`), and cache folders (`__pycache__/`) are untracked via `.gitignore`. The database is generated on-device when running `ingestion.py`.

---

## 📄 File Responsibilities

* **`docs/`:** Storage directory for raw PDF input files.
* **`ingestion.py`:** Parses PDFs via `pdfplumber`, creates overlapping chunks, computes dense embeddings, and writes to SQLite.
* **`query_app.py`:** Runs CLI interaction, executes BM25 + Vector Hybrid Search, applies RRF re-ranking, and streams answers via `Phi-3.5-mini`.
* **`requirements.txt`:** Specifies external Python dependencies (`foundry-local-sdk`, `pdfplumber`).
* **`.gitignore`:** Prevents local database files (`*.db`), virtual environments, and caches from being committed to source control.
* **`README.md`:** System architecture, setup instructions, and design trade-offs documentation.

---
📚 References & Resources
The technical architecture and local execution workflow of this project were built following the official Microsoft Learn documentation and technical guides:

📖 Microsoft Learn - Foundry Local Overview & Get Started — Official guide on setting up the Microsoft Foundry Local runtime, hardware acceleration, and SDK architecture.

🛠️ Tutorial: Build a RAG Application with Foundry Local — Step-by-step tutorial on building local Retrieval-Augmented Generation workflows on Windows.

💬 Tutorial: Build a Multi-Turn Chat Assistant with Foundry Local — Guide on building stateful conversational Python applications using local models.

🚀 Building Your First Local RAG Application with Foundry Local — Deep-dive technical blog post from Microsoft Tech Community on local embedding strategies and on-device LLM inference.

---
## ⚠️ Disclaimer

This project is developed independently as a non-commercial, personal student project.

* **Affiliation:** This project has no official affiliation, endorsement, sponsorship, or partnership with Sabanci University or any other academic institution.
* **Data Sources:** All data processed by this system is sourced strictly from publicly available documents. No proprietary, confidential, or internal institutional data is used.
* **Liability:** Outputs generated by this system are for educational, research, and proof-of-concept purposes only. They do not constitute official academic advice or legally binding information.
