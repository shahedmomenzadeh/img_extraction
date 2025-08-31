import os
import re
import csv
import string
import argparse
import requests
from typing import List, Dict, TypedDict, Any, Optional

# PDF handling
import fitz  # PyMuPDF

# Graph orchestration
from langgraph.graph import StateGraph, END

# Environment
from dotenv import load_dotenv
load_dotenv()

# Optional clients (import when available)
try:
    from groq import Groq
except Exception:
    Groq = None

try:
    import openai
except Exception:
    openai = None

# --------------------------------------------------------------------------
# Environment variables / global defaults
# --------------------------------------------------------------------------

# Global provider used as a fallback when a node doesn't specify one
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
DEFAULT_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")

# Provider-specific credentials / endpoints
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")  # optional
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = os.getenv("GEMINI_API_URL")

# Initialize Groq client only if available and key provided
groq_client = None
if Groq and GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        print(f"⚠️ Warning: Failed to initialize Groq client: {e}")

# --------------------------------------------------------------------------
# Per-node LLM configuration
# Edit these values to choose provider+model per node.
# Valid provider strings: 'groq', 'openai', 'ollama', 'gemini'
# You may also set equivalent environment variables named like
# NODE_EXTRACT_CAPTION_PROVIDER / NODE_EXTRACT_CAPTION_MODEL etc.
# --------------------------------------------------------------------------
LLM_NODE_CONFIG = {
    "extract_caption": {
        "provider": "ollama",
        "model": "gemma3:4b",
    },
    "extract_description": {
        "provider": "ollama",
        "model": "gemma3:4b",
    },
    "validate_content": {
        "provider": "ollama",
        "model": "gemma3:4b",
    },
    "clean_description": {
        "provider": "ollama",
        "model": "gemma3:4b",
    },
}

# --------------------------------------------------------------------------
# Helper: normalize various provider response shapes to a single string
# --------------------------------------------------------------------------

def _extract_text_from_response(resp: Any) -> str:
    """Inspect common response shapes and extract assistant text.

    Accepts either parsed JSON (dict/list) or SDK objects that behave like dicts.
    Falls back to stringifying the response.
    """
    try:
        # Common OpenAI/Groq style: {'choices': [ { 'message': { 'content': '...'} } ] }
        if isinstance(resp, dict) and "choices" in resp:
            ch = resp["choices"][0]
            if isinstance(ch, dict):
                if "message" in ch and isinstance(ch["message"], dict):
                    content = ch["message"].get("content")
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        texts = []
                        for c in content:
                            if isinstance(c, dict) and "text" in c:
                                texts.append(c["text"])
                        if texts:
                            return "\n".join(texts)
                if "text" in ch and isinstance(ch["text"], str):
                    return ch["text"]
        # Ollama sometimes returns choices[0].content -> list of [ { 'text': '...' } ]
        if isinstance(resp, dict) and "choices" in resp:
            ch0 = resp["choices"][0]
            if isinstance(ch0, dict) and "content" in ch0:
                cont = ch0.get("content")
                if isinstance(cont, str):
                    return cont
                if isinstance(cont, list) and len(cont) > 0:
                    first = cont[0]
                    if isinstance(first, dict) and "text" in first:
                        return first["text"]
        # Google Generative: {'candidates': [ {'content': '...'} ] }
        if isinstance(resp, dict) and "candidates" in resp:
            cand = resp["candidates"][0]
            if isinstance(cand, dict) and "content" in cand:
                return cand["content"]
    except Exception:
        pass
    # fallback
    try:
        return str(resp)
    except Exception:
        return ""

# --------------------------------------------------------------------------
# Unified chat completion caller (supports groq, openai, ollama, gemini)
# --------------------------------------------------------------------------

def call_llm(provider: str, messages: List[Dict[str, str]], model: Optional[str] = None, temperature: float = 0.0, timeout: int = 60) -> str:
    provider = (provider or LLM_PROVIDER or "").lower()
    model = model or DEFAULT_MODEL
    try:
        if provider == "groq":
            if not groq_client:
                raise RuntimeError("Groq client not initialized (missing GROQ_API_KEY or groq package).")
            resp = groq_client.chat.completions.create(model=model, messages=messages, temperature=temperature)
            # Some SDKs return objects with attributes; convert to dict if possible
            try:
                return _extract_text_from_response(dict(resp)).strip()
            except Exception:
                return _extract_text_from_response(resp).strip()

        if provider == "openai":
            if not openai:
                raise RuntimeError("openai package not installed.")
            if OPENAI_API_KEY:
                openai.api_key = OPENAI_API_KEY
            if OPENAI_API_BASE:
                openai.api_base = OPENAI_API_BASE
            # Use ChatCompletion interface for broad compatibility
            resp = openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature)
            return _extract_text_from_response(resp).strip()

        if provider == "ollama":
            base = OLLAMA_BASE_URL.rstrip("/")
            url = f"{base}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            if OLLAMA_API_KEY:
                headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
            payload = {"model": model, "messages": messages, "temperature": temperature}
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)
            r.raise_for_status()
            return _extract_text_from_response(r.json()).strip()

        if provider == "gemini":
            if not GEMINI_API_KEY or not GEMINI_API_URL:
                raise RuntimeError("GEMINI_API_KEY and GEMINI_API_URL must be set to use 'gemini' provider.")
            headers = {"Authorization": f"Bearer {GEMINI_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": model, "messages": messages, "temperature": temperature}
            r = requests.post(GEMINI_API_URL, json=payload, headers=headers, timeout=timeout)
            r.raise_for_status()
            return _extract_text_from_response(r.json()).strip()

        raise RuntimeError(f"Unknown LLM provider: {provider}")
    except Exception as e:
        print(f"🔴 Error calling {provider}: {e}")
        return "Error."

# Wrapper used by nodes; allows per-call provider/model override
def call_llm_chat(prompt: str, system_prompt: str = "You are a helpful assistant.", model: Optional[str] = None, temperature: float = 0.0, provider: Optional[str] = None) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    resolved_provider = provider or LLM_PROVIDER
    resolved_model = model or DEFAULT_MODEL
    return call_llm(resolved_provider, messages, model=resolved_model, temperature=temperature)

# --------------------------------------------------------------------------
# Workflow state definition
# --------------------------------------------------------------------------
class WorkflowState(TypedDict):
    pdf_path: str
    intermediate_csv_writer: Any
    final_csv_writer: Any
    doc: Any  # opened PDF
    page_num: int
    unprocessed_pairs: List[Dict]
    output_dir: str
    current_item: Optional[Dict]
    extracted_caption: Optional[str]
    extracted_description: Optional[str]
    validation_decision: Optional[str]
    final_description: Optional[str]

# --------------------------------------------------------------------------
# Node implementations (these call the LLM via call_llm_chat and use LLM_NODE_CONFIG)
# --------------------------------------------------------------------------

def prepare_page_node(state: WorkflowState) -> WorkflowState:
    page_num = state["page_num"]
    doc = state["doc"]
    print(f"\n--- 📄 Preparing Page {page_num}/{len(doc)} ---")
    page = doc[page_num - 1]

    pattern = re.compile(r'^(Figure|Fig\.?|Image|Img\.?|Plate|Illustration|Diagram|Chart|Graph|Photo|Photograph|Scheme|Map)\s+(\d+(?:[.-]\d+)*)\.?', re.IGNORECASE)
    caption_anchors = []
    for block in page.get_text("blocks"):
        try:
            text_block = block[4].strip()
        except Exception:
            text_block = ""
        match = pattern.match(text_block)
        if match:
            figure_name = f"{match.group(1)} {match.group(2)}"
            caption_anchors.append({"name": figure_name, "y": block[1]})

    found_images = []
    try:
        for img in page.get_images(full=True):
            bbox = page.get_image_bbox(img)
            found_images.append({"xref": img[0], "y": bbox.y0})
    except Exception:
        pass

    caption_anchors.sort(key=lambda item: item["y"])
    found_images.sort(key=lambda item: item["y"])

    unprocessed_pairs = state.get("unprocessed_pairs", [])
    if len(caption_anchors) > 0 and len(caption_anchors) == len(found_images):
        print(f"   Found {len(caption_anchors)} image-caption pairs to process.")
        for img_data, cap_anchor in zip(found_images, caption_anchors):
            unprocessed_pairs.append({
                "image_xref": img_data["xref"],
                "figure_name": cap_anchor["name"],
                "page_num": page_num,
            })
    else:
        print(f"   ⚠️ Mismatch: {len(found_images)} images and {len(caption_anchors)} captions. Skipping page.")

    return {**state, "unprocessed_pairs": unprocessed_pairs}


def pop_item_from_queue_node(state: WorkflowState) -> WorkflowState:
    if not state["unprocessed_pairs"]:
        return {**state, "current_item": None}
    unprocessed_pairs = state["unprocessed_pairs"]
    next_item = unprocessed_pairs.pop(0)
    print(f"\n   ▶️ Processing '{next_item['figure_name']}' from page {next_item['page_num']}")
    return {**state, "unprocessed_pairs": unprocessed_pairs, "current_item": next_item}


def extract_caption_node(state: WorkflowState) -> WorkflowState:
    item = state.get("current_item")
    if not item:
        return state
    figure_name = item["figure_name"]
    page_num = item["page_num"]
    doc = state["doc"]
    page_text = doc[page_num - 1].get_text("text")
    print(f"     🤖 Caption Extractor running for '{figure_name}'...")
    prompt = f"""You are a document extraction assistant. Find and return the complete caption for \"{figure_name}\" from the following text. The caption may span multiple lines. Return ONLY the full caption text, starting with \"{figure_name}\".\n\n--- PAGE TEXT ---\n{page_text}\n--- END OF TEXT ---\n"""
    node_conf = LLM_NODE_CONFIG.get("extract_caption", {})
    clean_caption = call_llm_chat(prompt, system_prompt="You are a helpful assistant.", model=node_conf.get("model"), provider=node_conf.get("provider"))
    print(f"       ✅ Extracted Caption: '{(clean_caption or '')[:150]}...'")
    return {**state, "extracted_caption": clean_caption}


def extract_description_node(state: WorkflowState) -> WorkflowState:
    item = state.get("current_item")
    if not item:
        return state
    figure_name = item["figure_name"]
    page_num = item["page_num"]
    doc = state["doc"]
    print(f"     🧠 Description Finder searching context for '{figure_name}'...")
    context_text = ""
    for i in range(max(0, page_num - 2), min(len(doc), page_num + 1)):
        context_text += f"\n--- Page {i+1} ---\n" + doc[i].get_text("text")
    prompt = f"""You are a research assistant. Find any sentences in the provided text that refer to and describe '{figure_name}'. Combine them into a single paragraph. If none are found, return \"No specific description found.\".\n\n--- TEXT CONTEXT ---\n{context_text}\n--- END OF TEXT ---\n"""
    node_conf = LLM_NODE_CONFIG.get("extract_description", {})
    description = call_llm_chat(prompt, system_prompt="You are a research assistant.", model=node_conf.get("model"), provider=node_conf.get("provider"))
    print(f"       ✅ Found Description: '{(description or '')[:150]}...'")
    return {**state, "extracted_description": description}


def validate_content_node(state: WorkflowState) -> WorkflowState:
    caption = state.get("extracted_caption")
    description = state.get("extracted_description")
    item = state.get("current_item")
    figure_name = item["figure_name"] if item else "(unknown)"
    if not caption or not description:
        return {**state, "validation_decision": "DISCARD"}
    print(f"     🔍 Quality Validator checking '{figure_name}'...")
    prompt = f"""You are a data quality specialist. A GOOD entry is a clear, complete sentence. A BAD entry contains garbled text, is fragmented, or is just a list. Return ONLY ONE WORD: KEEP or DISCARD.\n\n--- DATA TO VALIDATE ---\n**Caption:** \"{caption}\"\n**Description:** \"{description}\"\n--- END OF DATA ---\n"""
    node_conf = LLM_NODE_CONFIG.get("validate_content", {})
    decision = call_llm_chat(prompt, system_prompt="You are a data quality specialist.", model=node_conf.get("model"), provider=node_conf.get("provider"))
    if decision:
        decision = decision.strip().upper()
    if decision not in ("KEEP", "DISCARD"):
        decision = "DISCARD"
    print(f"       ➡️ Validation Result: {decision}")
    return {**state, "validation_decision": decision}


def clean_description_node(state: WorkflowState) -> WorkflowState:
    if state.get("validation_decision") != "KEEP":
        return state
    caption = state.get("extracted_caption")
    description = state.get("extracted_description")
    item = state.get("current_item")
    figure_name = item["figure_name"] if item else "(unknown)"
    print(f"     ✨ Final Cleaner processing '{figure_name}'...")
    combined_text = f"Caption: {caption}\n\nDescription from text: {description}"
    prompt = f"""You are a medical text editor. Create a single, clean, concise description based on caption and text.\n- Remove the figure number itself.\n- Remove citations like (1), [2], (A, B).\n- Start with \"This image shows...\" or similar.\nReturn ONLY the final cleaned description.\n\n--- COMBINED TEXT ---\n{combined_text}\n--- END OF TEXT ---\n"""
    node_conf = LLM_NODE_CONFIG.get("clean_description", {})
    final_desc = call_llm_chat(prompt, system_prompt="You are a medical text editor.", model=node_conf.get("model"), provider=node_conf.get("provider"))
    print(f"       ✅ Final Description: '{(final_desc or '')[:150]}...'")
    return {**state, "final_description": final_desc}


def save_item_node(state: WorkflowState) -> WorkflowState:
    if state.get("validation_decision") != "KEEP":
        print("     ❌ Item discarded by validator. Skipping save.")
        return state
    item = state.get("current_item")
    if not item:
        return state
    figure_name = item["figure_name"]
    output_dir = state["output_dir"]
    print(f"     💾 Saving all results for '{figure_name}'")
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    base_filename = ''.join(c for c in figure_name if c in valid_chars).replace(' ', '_')
    try:
        base_image = state["doc"].extract_image(item["image_xref"])
    except Exception as e:
        print(f"     ⚠️ Failed to extract image: {e}")
        return state
    image_filename = f"{base_filename}.{base_image.get('ext','png')}"
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    image_save_path = os.path.join(images_dir, image_filename)
    try:
        with open(image_save_path, "wb") as img_file:
            img_file.write(base_image["image"])
    except Exception as e:
        print(f"     ⚠️ Failed to save image: {e}")

    try:
        state["intermediate_csv_writer"].writerow([
            item["page_num"], image_filename, state.get("extracted_caption"), state.get("extracted_description")
        ])
        state["final_csv_writer"].writerow([image_filename, state.get("final_description")])
    except Exception as e:
        print(f"     ⚠️ Failed to write CSV row: {e}")
    return state


def increment_page_node(state: WorkflowState) -> WorkflowState:
    return {**state, "page_num": state["page_num"] + 1}

# --------------------------------------------------------------------------
# Routing functions
# --------------------------------------------------------------------------

def route_after_page_prep(state: WorkflowState) -> str:
    if state.get("unprocessed_pairs"):
        return "process_item"
    if state["page_num"] < len(state["doc"]):
        return "prepare_next_page"
    return "end"


def route_after_item_processing(state: WorkflowState) -> str:
    if state.get("unprocessed_pairs"):
        return "process_item"
    if state["page_num"] < len(state["doc"]):
        return "prepare_next_page"
    return "end"

# --------------------------------------------------------------------------
# Build and compile the workflow graph
# --------------------------------------------------------------------------
workflow = StateGraph(WorkflowState)
workflow.add_node("prepare_page", prepare_page_node)
workflow.add_node("pop_item", pop_item_from_queue_node)
workflow.add_node("extract_caption", extract_caption_node)
workflow.add_node("extract_description", extract_description_node)
workflow.add_node("validate_content", validate_content_node)
workflow.add_node("clean_description", clean_description_node)
workflow.add_node("save_item", save_item_node)
workflow.add_node("increment_page", increment_page_node)
workflow.set_entry_point("prepare_page")
workflow.add_conditional_edges("prepare_page", route_after_page_prep, {"process_item": "pop_item", "prepare_next_page": "increment_page", "end": END})
workflow.add_edge("pop_item", "extract_caption")
workflow.add_edge("extract_caption", "extract_description")
workflow.add_edge("extract_description", "validate_content")
workflow.add_edge("validate_content", "clean_description")
workflow.add_edge("clean_description", "save_item")
workflow.add_conditional_edges("save_item", route_after_item_processing, {"process_item": "pop_item", "prepare_next_page": "increment_page", "end": END})
workflow.add_edge("increment_page", "prepare_page")
app = workflow.compile()

# --------------------------------------------------------------------------
# Orchestrator for a single book
# --------------------------------------------------------------------------
BASE_OUTPUT_DIR = os.getenv("BASE_OUTPUT_DIR", "books_output")
INTERMEDIATE_CSV_FILENAME = os.getenv("INTERMEDIATE_CSV_FILENAME", "intermediate_data_validated.csv")
FINAL_CSV_FILENAME = os.getenv("FINAL_CSV_FILENAME", "final_image_descriptions.csv")


def process_single_book(pdf_path: str, book_output_dir: str):
    print(f"\n{'='*80}\n🚀 Starting Workflow for Book: {os.path.basename(pdf_path)}\n{'='*80}")
    os.makedirs(book_output_dir, exist_ok=True)
    images_dir = os.path.join(book_output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    intermediate_csv_path = os.path.join(book_output_dir, INTERMEDIATE_CSV_FILENAME)
    final_csv_path = os.path.join(book_output_dir, FINAL_CSV_FILENAME)
    with open(intermediate_csv_path, 'w', newline='', encoding='utf-8') as intermediate_f, open(final_csv_path, 'w', newline='', encoding='utf-8') as final_f:
        intermediate_writer = csv.writer(intermediate_f)
        final_writer = csv.writer(final_f)
        intermediate_writer.writerow(["page_number", "image_filename", "caption_text", "description_text"])
        final_writer.writerow(["image_filename", "final_description"])
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"🔴 Error opening PDF file '{pdf_path}': {e}")
            return
        initial_state = {
            "pdf_path": pdf_path,
            "output_dir": book_output_dir,
            "intermediate_csv_writer": intermediate_writer,
            "final_csv_writer": final_writer,
            "doc": doc,
            "page_num": 1,
            "unprocessed_pairs": [],
        }
        app.invoke(initial_state, {"recursion_limit": 6000})
    print(f"\n--- ✅ Workflow Complete for: {os.path.basename(pdf_path)} ---")
    print(f"   Results saved in: '{os.path.abspath(book_output_dir)}'")

# --------------------------------------------------------------------------
# Batch entry point
# --------------------------------------------------------------------------

def main(books_directory: str):
    print(f"--- Starting Batch Processing in Directory: {books_directory} ---")
    if not os.path.isdir(books_directory):
        print(f"🔴 Error: The directory '{books_directory}' was not found.")
        return
    pdf_files = [f for f in os.listdir(books_directory) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"🟡 Warning: No PDF files found in '{books_directory}'.")
        return
    print(f"Found {len(pdf_files)} PDF books to process.")
    for pdf_filename in pdf_files:
        pdf_path = os.path.join(books_directory, pdf_filename)
        book_name = os.path.splitext(pdf_filename)[0]
        book_output_dir = os.path.join(BASE_OUTPUT_DIR, book_name)
        process_single_book(pdf_path, book_output_dir)
    print(f"\n{'='*80}\n🎉 All books have been processed. Batch job finished. \n{'='*80}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process PDFs in a directory to extract, describe, validate, and clean image data.")
    parser.add_argument("books_dir", type=str, help="Path to the directory containing input PDF files (e.g., 'books').")
    args = parser.parse_args()
    # Print resolved node-level provider/model mapping for verification
    print("Resolved per-node LLM configuration:")
    for node_name, conf in LLM_NODE_CONFIG.items():
        print(f"  - {node_name}: provider={conf.get('provider')}, model={conf.get('model')}")
    main(args.books_dir)
