# Humbet AI Chatbot Backend

Backend FastAPI untuk chatbot edukasi Humbet dengan arsitektur Modular Retrieval-Augmented Generation (RAG). Backend ini hanya menyediakan REST API sehingga dapat dikonsumsi oleh berbagai frontend.

## Arsitektur Tingkat Tinggi

- Framework: FastAPI
- RAG: LangChain berbasis FAISS
- Embedding:
  - `text-embedding-ada-002` (OpenAI) atau
  - `BAAI/bge-large-en` (via Sentence Transformers)
- LLM: GPT-3.5 / GPT-4 (melalui OpenAI API)
- Mode RAG:
  - `naive`
  - `advanced`
  - `modular`

Struktur direktori utama:

```text
backend/
  app/
    main.py
    api/
      chat.py
      ingest.py
      evaluate.py
    core/
      config.py
      logging.py
    rag/
      query_refinement.py
      dynamic_decision.py
      iter_retgen.py
      retriever.py
      generator.py
    data/
      vector_store.py
    evaluation/
      retrieval_metrics.py
      ragas_eval.py
    schemas/
      chat_schema.py
  requirements.txt
```

## Konfigurasi

Konfigurasi menggunakan variabel lingkungan (dan dapat diatur melalui berkas `.env`):

- `RAG_MODE`: `naive` | `advanced` | `modular` (default: `modular`)
- `VECTOR_BACKEND`: `faiss` | `chroma` (saat ini diimplementasikan: `faiss`)
- `EMBEDDING_MODEL`: nama model embedding (default: `text-embedding-ada-002`)
- `USE_BGE`: `true`/`false`, aktifkan `BAAI/bge-large-en`
- `LLM_MODEL`: nama model LLM (misalnya `gpt-3.5-turbo`, `gpt-4`)
- `OPENAI_API_KEY`: kunci API OpenAI

Direktori penyimpanan:

- `storage/vectors`: penyimpanan indeks FAISS
- `storage/logs`: log eksperimen dan trace RAG

## Modul RAG

### 1. RQ – Query Refinement (`app/rag/query_refinement.py`)

- Menerima query mentah pengguna.
- Mendeteksi ambiguitas berbasis heuristik (panjang, kata ganti, dll.).
- Melakukan:
  - normalisasi / perluasan query kontekstual,
  - dekomposisi query menjadi beberapa sub-query paralel jika relevan.
- Keluaran:

```json
{
  "original_query": "...",
  "refined_query": "...",
  "sub_queries": ["...", "..."],
  "is_ambiguous": true
}
```

### 2. DRAGIN – Dynamic Retrieval Trigger (`app/rag/dynamic_decision.py`)

- Mengestimasi ketidakpastian berbasis panjang dan tipe query.
- Menghasilkan keputusan:

```json
{
  "retrieve": true,
  "confidence": 0.63,
  "reason": "Diperlukan retrieval untuk menurunkan halusinasi."
}
```

- Informasi ini digunakan dalam loop RAG untuk:
  - memutuskan kapan retrieval wajib,
  - mengukur confidence akhir jawaban.

### 3. ITER-RETGEN – Iterative Retrieve–Generate (`app/rag/iter_retgen.py`)

- Mengorkestrasi loop:
  1. Refinement query (RQ)
  2. Keputusan retrieval (DRAGIN)
  3. Retrieval dokumen (vector store FAISS)
  4. Generasi jawaban berbasis konteks
- Mode:
  - `naive`: satu iterasi langsung retrieve → generate.
  - `advanced`: satu iterasi dengan penyesuaian confidence.
  - `modular`: beberapa iterasi sampai:
    - query hasil refinement cukup spesifik,
    - confidence di atas ambang,
    - atau batas iterasi tercapai.
- Menyimpan jejak per iterasi (refined query, jumlah dokumen, keputusan DRAGIN) untuk keperluan tesis.

### 4. Retriever & Generator

- `retriever.py`
  - Mengambil dokumen top-k dari FAISS berdasarkan `similarity_search_with_score`.
  - Mencatat jumlah dokumen dan query yang digunakan.
- `generator.py`
  - Menggunakan `ChatOpenAI` (OpenAI GPT-3.5/4).
  - Membangun prompt sistem yang:
    - membatasi jawaban hanya pada konteks,
    - melarang halusinasi,
    - mewajibkan penyebutan sumber dokumen.
  - Mengembalikan:
    - teks jawaban,
    - daftar sumber (nama file, chunk id).

## Vector Store dan Ingestion

`app/data/vector_store.py` mengelola:

- Model embedding (OpenAI atau BGE).
- Indeks FAISS yang persisten (`storage/vectors/faiss_index`).
- Proses ingestion:
  - chunking teks dengan `RecursiveCharacterTextSplitter`,
  - penyimpanan metadata `source` dan `chunk_id`.

Endpoint ingestion (`/api/ingest`) mendukung:

- PDF (via `PyPDF2`)
- HTML (via `BeautifulSoup`)

Pipeline ingestion:

1. Upload dokumen.
2. Ekstraksi teks.
3. Chunking dan embedding.
4. Penyimpanan ke FAISS.

## Endpoint API

### 1. Chat – `POST /api/chat`

Input:

```json
{
  "query": "Jadwal ujian"
}
```

Output:

```json
{
  "answer": "...",
  "sources": [
    {
      "id": 0,
      "source": "kalender_akademik.pdf",
      "chunk_id": "0-3"
    }
  ],
  "iterations": 2,
  "confidence": 0.87,
  "trace": [
    {
      "iteration": 1,
      "refined_query": "...",
      "num_documents": 5,
      "retrieve": true,
      "retrieval_confidence": 0.8,
      "reason": "Diperlukan retrieval untuk menurunkan halusinasi."
    }
  ]
}
```

### 2. Ingestion – `POST /api/ingest`

- Form-data dengan field `file` (PDF/HTML).

Keluaran:

```json
{
  "source": "kalender_akademik.pdf",
  "num_chunks": 42
}
```

### 3. Evaluation – `POST /api/evaluate`

Payload:

```json
{
  "questions": ["..."],
  "ground_truth_answers": ["..."],
  "relevant_doc_ids": [["kalender_akademik.pdf-0-3"]]
}
```

Keluaran:

```json
{
  "hit_rate": 0.9,
  "mrr": 0.85,
  "ragas": {
    "faithfulness": 0.91,
    "answer_relevance": 0.88,
    "context_precision": 0.86
  }
}
```

## Evaluasi dan Eksperimen

- `retrieval_metrics.py`:
  - Hit Rate@k
  - MRR@k
- `ragas_eval.py`:
  - Faithfulness
  - Answer Relevance
  - Context Precision

Dengan struktur ini, backend siap digunakan untuk:

- eksperimen RAG sederhana (naive),
- RAG yang lebih canggih (advanced),
- RAG modular dengan loop iteratif (modular),
- logging rinci untuk pelaporan skripsi.

