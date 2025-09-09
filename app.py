# app.py
import os
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# LangChain (versioni recenti)
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# =========================
# ====== CONFIG BASE ======
# =========================
DOCS_ROOT = "docs"                 # struttura: /docs/<Materia>/*.pdf
USERS_FILE = "users.json"
VALID_CODES = {"2","3","5","8","13","21","34","55","89","144"}  # set = lookup veloce
MAX_QUESTIONS_PER_DAY = 5
MODEL_NAME = "gpt-4o-mini"         # più economico/veloce di 3.5-turbo (deprecato)

# =========================
# === FUNZIONI SUPPORTO ===
# =========================
def local_css(file_name: str):
    if not os.path.exists(file_name):
        return
    with open(file_name, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def ensure_users_file():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)

def load_users():
    ensure_users_file()
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_users(users: dict):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)

def check_quota(user_code: str) -> int:
    users = load_users()
    today = datetime.today().strftime("%Y-%m-%d")
    if user_code not in users:
        users[user_code] = {"count": 0, "last_access": today}
    elif users[user_code].get("last_access") != today:
        users[user_code]["count"] = 0
        users[user_code]["last_access"] = today
    save_users(users)
    return users[user_code]["count"]

def increment_quota(user_code: str):
    users = load_users()
    users[user_code]["count"] = users[user_code].get("count", 0) + 1
    save_users(users)

def list_materie(root: str) -> list[str]:
    """Ritorna l'elenco delle cartelle (materie) presenti in DOCS_ROOT."""
    p = Path(root)
    if not p.exists():
        return []
    return sorted([f.name for f in p.iterdir() if f.is_dir()])

def load_documents_from_folder(folder_path: str):
    """Carica e splitta tutti i PDF di una cartella (ricorsivo)."""
    docs = []
    folder = Path(folder_path)
    if not folder.exists():
        return docs
    for pdf_path in folder.rglob("*.pdf"):
        try:
            loader = PyPDFLoader(str(pdf_path))
            # load_and_split di PyPDFLoader ritorna pagine già spezzate
            docs.extend(loader.load_and_split())
        except Exception as e:
            st.warning(f"Impossibile leggere: {pdf_path.name} ({e})")
    return docs

@st.cache_resource(show_spinner=False)
def build_vectorstore(materia: str):
    """
    Crea (e cache) un FAISS vectorstore per:
      - "Tutte le materie" -> scansione completa di DOCS_ROOT
      - nome materia       -> solo quella cartella
    """
    if materia == "Tutte le materie":
        folder = DOCS_ROOT
    else:
        folder = os.path.join(DOCS_ROOT, materia)

    docs = load_documents_from_folder(folder)
    if not docs:
        # Lascio che il chiamante gestisca l'errore in UI
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        return None

    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(chunks, embeddings)

def make_retriever(vectorstore):
    # puoi affinare: search_kwargs={"k": 4}
    return vectorstore.as_retriever()

# =========================
# ===== STREAMLIT APP =====
# =========================
load_dotenv()
st.set_page_config(page_title="Chatbot Materie", page_icon="📚", layout="centered")
local_css("style.css")

st.markdown("<div class='logo-container' style='text-align:center;margin-bottom:12px'>"
            "<img src='https://i.postimg.cc/5NqjQ63K/images.jpg' height='80'>"
            "</div>", unsafe_allow_html=True)
st.title("📚 Chatbot Educativo")
st.markdown("Interroga i documenti caricati per materia. **Max 5 domande al giorno per codice.**")

# --- Codice corsista ---
user_code = st.text_input("🔐 Inserisci il tuo codice corsista:")
if user_code:
    if user_code not in VALID_CODES:
        st.error("❌ Codice non valido.")
        st.stop()
    count = check_quota(user_code)
    remaining = MAX_QUESTIONS_PER_DAY - count
    st.caption(f"Domande residue oggi: **{max(0, remaining)}**")
    if count >= MAX_QUESTIONS_PER_DAY:
        st.warning("⛔ Hai raggiunto il limite giornaliero. Riprova domani.")
        st.stop()

    # --- Scelta materia ---
    materie_fs = list_materie(DOCS_ROOT)
    # Etichette “carine” solo se esistono davvero
    label_map = {
        "Educazione cinofila": "📘 Educazione Cinofila",
        "Istruzione cinofila": "📗 Istruzione Cinofila",
    }
    # Mappa nome-cartella -> etichetta visuale (se definita), altrimenti nome puro
    nice_labels = [label_map.get(m, m) for m in materie_fs]
    # Reverse map etichetta -> cartella
    reverse_label = {label_map.get(m, m): m for m in materie_fs}

    materia_scelta = st.selectbox(
        "📁 Scegli la materia:",
        options=["Tutte le materie"] + nice_labels,
        index=0
    )

    # Normalizzo alla cartella effettiva
    materia_folder = reverse_label.get(materia_scelta, materia_scelta)

    # --- Box domanda in form (invio con Enter oppure bottone) ---
    with st.form("qa_form", clear_on_submit=False):
        user_question = st.text_input("✍️ Fai la tua domanda:", value="", placeholder="Es. Spiega la differenza tra X e Y…")
        submitted = st.form_submit_button("Cerca nei materiali")

    if submitted and user_question.strip():
        # Costruzione (o recupero cache) del vectorstore
        with st.spinner("🔎 Sto analizzando i materiali…"):
            vectorstore = build_vectorstore(materia_folder)
            if vectorstore is None:
                if materia_scelta == "Tutte le materie":
                    st.error("⚠️ Nessun documento trovato in **docs/**. Aggiungi dei PDF per iniziare.")
                else:
                    st.error(f"⚠️ Nessun documento trovato per **{materia_folder}** in `docs/`.")
                st.stop()

            retriever = make_retriever(vectorstore)

            try:
                llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True
                )
                result = qa({"query": user_question})
                answer = result["result"]
                sources = result.get("source_documents", []) or []

                increment_quota(user_code)

            except Exception as e:
                st.error(f"Si è verificato un errore durante la risposta del modello: {e}")
                st.stop()

        # --- Output risposta ---
        st.success(answer)

        # --- Sorgenti (facoltativo, utile per trasparenza) ---
        if sources:
            with st.expander("Mostra estratti fonti"):
                for i, doc in enumerate(sources, start=1):
                    src_path = doc.metadata.get("source", "Sorgente sconosciuta")
                    page = doc.metadata.get("page", None)
                    page_str = f" (pag. {page+1})" if isinstance(page, int) else ""
                    st.markdown(f"**Fonte {i}:** `{src_path}`{page_str}")
                    # Mostro un piccolo estratto del contenuto
                    snippet = doc.page_content.strip()
                    if len(snippet) > 500:
                        snippet = snippet[:500] + "…"
                    st.code(snippet)

# =========================
# ==== NOTE DI UTILIZZO ===
# =========================
# 1) Metti i PDF in cartelle dentro /docs, es.:
#    docs/
#      Educazione cinofila/
#        lezione1.pdf
#      Istruzione cinofila/
#        moduloA.pdf
#
# 2) Aggiungi la tua API key in .env:
#    OPENAI_API_KEY=sk-xxxxx
#
# 3) Avvia:
#    streamlit run app.py
