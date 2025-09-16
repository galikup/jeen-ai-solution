# Project Report – AI Document Indexing & Search

## Project Objective

In this project, I developed a system that demonstrates how to:

- Store medical documents (PDF / Word) in a PostgreSQL database.  
- Split documents into smaller, readable chunks.  
- Generate vector embeddings for each chunk using Google Gemini.  
- Perform semantic search in natural language across all stored documents.  

**Goal:** Enable intelligent search within medical content – for example, identifying early warning signs of diseases or returning personalized answers directly from existing documents.

---

## Steps I Completed

### 1. Work Environment Setup
- Installed Python 3.12 and created a virtual environment (`venv`).  
- Installed required libraries:
  - `psycopg2-binary` – database connection  
  - `google-generativeai` – embeddings and LLM  
  - `python-docx` and `pypdf` – file parsing  
  - `numpy` – vector calculations  
  - `dotenv` – securely load environment variables  
- Created a `.env` file to store secrets:
  - `GEMINI_API_KEY` (Google Gemini key)  
  - `POSTGRES_URL` (database connection string)

### 2. Cloud Database (PostgreSQL on Neon)
- Created a PostgreSQL database hosted on **Neon**.  
- Defined a table `chunks` with these columns:
  - `id` – unique ID (primary key)  
  - `filename` – original document name  
  - `split_strategy` – how the text was split  
  - `chunk_text` – paragraph text  
  - `embedding` – numeric vector (`FLOAT8[]`)  
  - `created_at` – timestamp of insertion  
- Verified connection between Python and the database.

### 3. Document Indexing Pipeline
Script: **`index_files.py`**  
- Reads PDF/DOCX files.  
- Cleans and normalizes text.  
- Splits text into paragraph-based chunks.  
- Generates embeddings for each chunk using Gemini.  
- Inserts results into the PostgreSQL table.  
- Uses the **paragraph-based** split strategy.  

### 4. Semantic Search Pipeline
Script: **`query_engine.py`**  
- Takes a natural language query from the user.  
- Converts it into an embedding vector.  
- Fetches stored embeddings from the database.  
- Uses cosine similarity to compare query vs. stored chunks.  
- Returns the top 5 most relevant passages with metadata (filename, strategy, similarity score).

### 5. Testing with Sample Documents
- Created **20 synthetic medical test documents**.  
- Indexed them with `index_files.py`.  
- Queried the system (e.g., “What are early signs of diabetes?”) to validate search quality.  

---

## Function Explanations

### Indexing Pipeline (`index_files.py`)

- **`read_text(path: str) -> str`**  
  Reads a document (PDF or DOCX) and extracts plain text.  

- **`clean(t: str) -> str`**  
  Removes null characters, excess spaces, and repeated newlines.  

- **`split_paragraphs(t: str, max_len=1000) -> list[str]`**  
  Splits text into chunks (paragraph-based), respecting a max length.  

- **`embed_batch(texts: list[str]) -> list[list[float]]`**  
  Sends a batch of text chunks to Gemini and returns embeddings.  

- **`normalize_embeddings(resp)`**  
  Handles variations in Gemini API response formats, ensuring consistent embeddings.  

- **`ensure_table()`**  
  Creates the `chunks` table in PostgreSQL if not already present.  

- **`insert_rows(filename, strategy, chunks, embs)`**  
  Inserts chunks and their embeddings into the database.  

- **`index_file(path: str)`**  
  Full pipeline: read → clean → split → embed → store.

---

### Search Pipeline (`query_engine.py`)

- **`embed(text: str) -> np.ndarray`**  
  Converts a query into an embedding vector.  

- **`cosine(a, b) -> float`**  
  Computes cosine similarity between two vectors.  

- **`fetch_chunks() -> list[dict]`**  
  Retrieves chunks and embeddings from the database as numpy arrays.  

- **`top_k(query: str, k: int = 5)`**  
  Embeds the query, compares with stored embeddings, and returns top-k matches.  

jeen-ai-solution/
│── index_files.py       # Document indexing pipeline
│── query_engine.py      # Query + semantic search engine
│── search_documents.py  # Extended search with synthesized answer option
│── check_db.py          # Test DB connection
│── .env                 # Secrets (API keys, DB URL)
│── test_med_files/      # 20 test documents
│── README.md            # Project documentation


The system demonstrates a basic semantic search engine with embeddings + PostgreSQL.

The embeddings and queries are powered by Google Gemini.

The database is hosted on Neon (PostgreSQL).

The project shows how medical content can be indexed and searched intelligently.


Example 1
Enter your query: What are the best practices to prevent type 2 diabetes?


Output:

=== TOP RESULTS ===
[1] doc_01.pdf | paragraph_based | score=0.7961
Type 2 Diabetes Prevention [...]
[2] doc_06.pdf | paragraph_based | score=0.5802
Heart Failure Prevention [...]
[3] doc_16.pdf | paragraph_based | score=0.5766
Obesity and Weight Management [...]
[4] doc_08.pdf | paragraph_based | score=0.5693
Osteoporosis Prevention [...]
[5] doc_20.pdf | paragraph_based | score=0.5626
Cardiovascular Risk Reduction [...]

Example 2
Enter your query: How can I reduce my risk of developing heart failure?


Output:

=== TOP RESULTS ===
[1] doc_06.pdf | paragraph_based | score=0.7619
Heart Failure Prevention [...]
[2] doc_20.pdf | paragraph_based | score=0.6171
Cardiovascular Risk Reduction [...]
[3] doc_01.pdf | paragraph_based | score=0.5438
Type 2 Diabetes Prevention [...]
[4] doc_02.pdf | paragraph_based | score=0.5399
Hypertension Management [...]
[5] doc_05.pdf | paragraph_based | score=0.5383
Chronic Obstructive Pulmonary Disease (COPD) [...]

Example 3
Enter your query: What are the early warning signs of heart disease?


Output:

=== TOP RESULTS ===
[1] doc_06.pdf | paragraph_based | score=0.5303
Heart Failure Prevention [...]
[2] doc_20.pdf | paragraph_based | score=0.5235
Cardiovascular Risk Reduction [...]
[3] doc_03.pdf | paragraph_based | score=0.4786
Cholesterol Control [...]
[4] doc_01.pdf | paragraph_based | score=0.4423
Type 2 Diabetes Prevention [...]
[5] doc_17.pdf | paragraph_based | score=0.4349
Cancer Screening Awareness [...]