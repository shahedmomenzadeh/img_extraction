# text_img_extraction

A lightweight pipeline to extract figures and captions from PDFs, generate concise descriptions using configurable LLM providers (Ollama, Groq, OpenAI, Gemini), validate and clean extracted text, and save images and CSV outputs for downstream use.

## Features
- Detects figure/image-caption pairs on PDF pages.
- Extracts full captions and context-aware descriptions.
- Uses configurable per-node LLM providers and models.
- Validates and discards low-quality extractions.
- Cleans and normalizes final descriptions.
- Saves extracted images and two CSVs: intermediate raw data and final cleaned descriptions.
- Batch processing of multiple PDFs in a directory.

## Architecture
- Core script: `img_extraction.py` — orchestrates the workflow using a StateGraph.
- Nodes (stages): prepare_page, pop_item, extract_caption, extract_description, validate_content, clean_description, save_item, increment_page.
- LLM call abstraction with support for multiple providers via environment variables and per-node overrides.

## Requirements
- Python 3.8+
- Recommended packages:
  - PyMuPDF (fitz)
  - requests
  - python-dotenv
  - langgraph
  - groq (optional)
  - openai (optional)
  - Any provider-specific client/library you intend to use

Example (create virtualenv then):
```
python -m venv .venv
source .venv/bin/activate      # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Environment variables
Configure credentials and defaults via environment variables (use a `.env` file or export in your shell).

Important variables:
- LLM_PROVIDER: global fallback provider (e.g. `ollama`, `groq`, `openai`, `gemini`). Default: `groq`.
- LLM_MODEL: fallback model string (example: `openai/gpt-oss-20b`).
- GROQ_API_KEY: API key for Groq client.
- OPENAI_API_KEY: API key for OpenAI (if used).
- OPENAI_API_BASE: optional OpenAI base URL for custom endpoints.
- OLLAMA_BASE_URL: base URL for Ollama (default `http://localhost:11434`).
- OLLAMA_API_KEY: Ollama API key (if required).
- GEMINI_API_KEY, GEMINI_API_URL: for Gemini provider requests.
- BASE_OUTPUT_DIR: where per-book output folders are written (default `books_output`).
- INTERMEDIATE_CSV_FILENAME: filename for intermediate CSV (default `intermediate_data_validated.csv`).
- FINAL_CSV_FILENAME: filename for final CSV (default `final_image_descriptions.csv`).

Per-node provider/model overrides can be set in the `LLM_NODE_CONFIG` mapping inside `img_extraction.py`. Example node keys:
- extract_caption
- extract_description
- validate_content
- clean_description

## Configuration examples
Set provider/model globally:
```
LLM_PROVIDER=ollama
LLM_MODEL=gemma3:4b
```
Or override per node by editing `LLM_NODE_CONFIG` in `img_extraction.py`:
```
"extract_caption": { "provider": "ollama", "model": "gemma3:4b" }
```

## Usage

Basic CLI:
```
python img_extraction.py /path/to/books_directory
```
- The script scans the directory for `.pdf` files and processes each PDF.
- Outputs are written to `BASE_OUTPUT_DIR/<book_name>/` by default.

What gets saved for each processed image:
- images/<figure_name>.<ext> — extracted image file
- intermediate CSV (`intermediate_data_validated.csv`) with rows: page_number, image_filename, caption_text, description_text
- final CSV (`final_image_descriptions.csv`) with rows: image_filename, final_description

Example run:
```
# Ensure environment variables are set (e.g., in .env) and dependencies installed
python img_extraction.py ./books
```

## Output layout
books_output/
  └─ <book_name>/
     ├─ images/
     │  └─ Figure_1.png
     ├─ intermediate_data_validated.csv
     └─ final_image_descriptions.csv

## Tips & Troubleshooting
- "No API key" or client import errors: ensure provider SDKs are installed and keys set in env.
- Mismatched image/caption counts on a page: code currently skips such pages and logs a warning.
- If LLM responses are unexpected, check `LLM_NODE_CONFIG` and provider/model selections; logs print resolved per-node config at start.
- Increase the `recursion_limit` passed to `app.invoke(...)` only if necessary for deep graphs; the script already uses `{"recursion_limit": 6000}`.
- For local Ollama use, confirm `OLLAMA_BASE_URL` and optional `OLLAMA_API_KEY`.
- If images fail to extract, ensure the PDF contains image streams PyMuPDF can access.

## Extending & Contributing
- Add support for other providers by extending `call_llm()` and setting environment variables.
- Improve figure-caption pairing heuristics in `prepare_page_node`.
- Add unit tests for node functions and response normalization.

Pull requests and issue reports are welcome.

