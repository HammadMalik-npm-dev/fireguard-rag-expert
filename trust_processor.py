import os
import sys
import gc
import warnings
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# -------------------- LangChain Imports (Corrected) --------------------
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

# UPDATED: Using LCEL helper functions
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# -------------------- CONFIG --------------------
PDF_DIR = "./pdfs"
OUTPUT_DIR = "./output"
DB_PATH = "./faiss_index_spanish"

MODEL_NAME = "tinyllama"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# -------------------- 1. LOAD & SPLIT PDFs --------------------
def load_pdfs():
    print("--- 1. Escaneando Manuales (PDFs) ---")
    if not os.path.exists(PDF_DIR):
        print(f"ERROR: No existe la carpeta '{PDF_DIR}'")
        sys.exit(1)

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("ERROR: No se encontraron PDFs.")
        sys.exit(1)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_chunks = []

    for pdf in tqdm(pdf_files, desc="Leyendo PDFs"):
        try:
            loader = PyMuPDFLoader(os.path.join(PDF_DIR, pdf))
            pages = loader.load()
            chunks = splitter.split_documents(pages)
            all_chunks.extend(chunks)
            del pages
            gc.collect()
        except Exception as e:
            print(f"Advertencia leyendo {pdf}: {e}")

    if not all_chunks:
        print("ERROR CRÍTICO: No se extrajo texto.")
        sys.exit(1)

    return all_chunks

# -------------------- 2. BUILD VECTOR DATABASE --------------------
def build_database():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.exists(DB_PATH):
        import shutil
        shutil.rmtree(DB_PATH)

    chunks = load_pdfs()
    print(f"Construyendo FAISS con {len(chunks)} fragmentos...")

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_PATH)
    return db

# -------------------- 3. MAIN PROCESS --------------------
def run_process():
    print("--- 2. Conectando con la IA (Phi-3) ---")
    try:
        llm = OllamaLLM(model=MODEL_NAME)
    except Exception:
        print("ERROR: Ollama no está corriendo.")
        return

    db = build_database()

    # -------------------- PROMPT --------------------
    # Note: '{input}' is required for create_retrieval_chain to map the question automatically
    prompt_template = """
Eres un asistente experto en bomberos.
Usa SOLO el contexto del manual para explicar la respuesta correcta.

Contexto:
{context}

Pregunta:
{input}

Respuesta correcta:
{answer_option}

INSTRUCCIONES:
1. Explica por qué la respuesta es correcta usando SOLO el contexto.
2. Usa citas textuales si es posible.
3. Si no está en el contexto, di:
"No encontrado en el manual provisto."

Explicación:
"""

    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # FIX: Use create_stuff_documents_chain (Modern way)
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    retriever = db.as_retriever(search_kwargs={"k": 2})
    
    # FIX: Use create_retrieval_chain (Modern way)
    qa_chain = create_retrieval_chain(retriever, document_chain)

    # -------------------- LOAD QUESTIONS --------------------
    print("--- 3. Cargando preguntas ---")
    if os.path.exists("questions.xlsx"):
        df = pd.read_excel("questions.xlsx")
    elif os.path.exists("questions.csv"):
        df = pd.read_csv("questions.csv", encoding="latin-1")
    else:
        print("ERROR: No se encontró questions.xlsx o questions.csv")
        return

    if "Question" not in df.columns or "Correct Answer" not in df.columns:
        print("ERROR: El archivo debe tener columnas 'Question' y 'Correct Answer'")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df["explanation"] = ""

    # -------------------- PROCESS QUESTIONS --------------------
    print("--- 4. Procesando Preguntas ---")
    
    # NOTE: .head(10) is used for testing. Remove it to process the whole file.
    rows_to_process = df.head(10) 
    
    for idx, row in tqdm(rows_to_process.iterrows(), total=len(rows_to_process), desc="Analizando"):
        question_text = row["Question"]
        answer_text = str(row["Correct Answer"])

        try:
            # FIX: Pass 'input' for the retrieval query, and 'answer_option' for the prompt
            response = qa_chain.invoke({
                "input": question_text, 
                "answer_option": answer_text
            })
            df.at[idx, "explanation"] = response["answer"]
        except Exception as e:
            print(f"Error procesando fila {idx}: {e}")
            df.at[idx, "explanation"] = "Error generando explicación"

    # -------------------- SAVE OUTPUT --------------------
    output_path = os.path.join(OUTPUT_DIR, "Resultado_Final.xlsx")
    df.to_excel(output_path, index=False)
    print("\n¡ÉXITO! Archivo generado:", output_path)

# -------------------- RUN --------------------
if __name__ == "__main__":
    run_process()