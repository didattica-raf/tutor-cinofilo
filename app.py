# app.py
import streamlit as st
import os
import io
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from pypdf import PdfReader  # per leggere PDF da bytes
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# === IMPORT MODERNI PER OPENAI (migrazione consigliata) ===
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# --- CONFIG (immutata dove possibile) ---
DOCS_ROOT = "docs"  # non più usata per caricare, ma la lascio per compatibilità
USERS_FILE = "users.json"
VALID_CODES = ["2", "3", "5", "8", "13", "21", "34", "55", "89", "144"]
MAX_QUESTIONS_PER_DAY = 5

# --- UTIL: secrets/env ---
def _get_secret(key: str, default: str | None = None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

# --- FTP helpers (lettura in memoria, niente scritture su disco) ---
def _ftp_connect():
    """Ritorna una connessione FTP o FTPS (TLS) usando i secrets."""
    from ftplib import FTP, FTP_TLS

    host = _get_secret("FTP_HOST")
    port = int(_get_secret("FTP_PORT", "21"))
    user = _get_secret("FTP_USER")
    password = _get_secret("FTP_PASS")
    if not all([host, user, password]):
        raise RuntimeError("FTP non configurato (FTP_HOST/FTP_USER/FTP_PASS).")

    # tenta FTPS esplicito, poi fallback a FTP
    try:
        ftps = FTP_TLS()
        ftps.connect(host, port, timeout=30)
        ftps.login(user=user, passwd=password)
        try:
            ftps.prot_p()
        except Exception:
            pass
        return ftps
    except Exception:
        ftp = FTP()
        ftp.connect(host, port, timeout=30)
        ftp.login(user=user, passwd=password)
        return ftp

def _ftp_list(ftp, path):
    """Ritorna lista di (name, kind) dove kind in {'file','dir'}."""
    items = []
    try:
        lines = []
        ftp.retrlines(f"MLSD {path}", lines.append)
        for e in lines:
            facts, _, fname = e.partition(" ")
            if not fname or fname in (".", ".."):
                continue
            factmap = dict(f.split("=", 1) for f in facts.rstrip(";").split(";") if "=" in f)
            kind = factmap.get("type", "file")
            items.append((fname, "dir" if kind == "dir" else "file"))
    except Exception:
        lines = []
        ftp.retrlines(f"LIST {path}", lines.append)
        for line in lines:
            parts = line.split(maxsplit=8)
            if not parts:
                continue
            kind = "dir" if parts[0].startswith("d") else "file"
            fname = parts[-1]
            if fname in (".", ".."):
                continue
            items.append((fname, kind))
    return items

def _ftp_list_dirs(ftp, base_dir):
    """Elenca solo le sottocartelle di base_dir (le 'materie')."""
    try:
        ftp.cwd(base_dir)
    except Exception:
        # prova qualche root comune se base_dir non esiste
        for r in ["/public_html", "/www", "/web", "/htdocs", "/"]:
            try:
                ftp.cwd(r)
                base_dir = r
                break
            except Exception:
                continue
    dirs = []
    for name, kind in _ftp_list(ftp, base_dir):
        if kind == "dir":
            dirs.append(name)
    return base_dir, sorted(dirs)

def _ftp_iter_pdfs(ftp, path):
    """Genera (remote_path, filename) per i PDF sotto 'path' (non ricorsivo)."""
    pdfs = []
    for name, kind in _ftp_list(ftp, path):
        if kind == "file" and name.lower().endswith(".pdf"):
            pdfs.append((path.rstrip("/") + "/" + name, name))
    return pdfs

def _ftp_read_file_bytes(ftp, remote_path):
    """Scarica un file in memoria (bytes) senza scriverlo su disco."""
    buf = io.BytesIO()
    ftp.retrbinary(f"RETR {remote_path}", buf.write)
    buf.seek(0)
    return buf

# --- FUNZIONI DI SUPPORTO (tue) ---
def local_css(file_name):
    if not os.path.exists(file_name):
        return
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def ensure_users_file():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump({}, f)

def load_users():
    ensure_users_file()
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

def check_quota(user_code):
    users = load_users()
    today = datetime.today().strftime("%Y-%m-%d")
    if user_code not in users:
        users[user_code] = {"count": 0, "last_access": today}
    elif users[user_code]["last_access"] != today:
        users[user_code]["count"] = 0
        users[user_code]["last_access"] = today
    save_users(users)
    return users[user_code]["count"]

def increment_quota(user_code):
    users = load_users()
    users[user_code]["count"] += 1
    save_users(users)

# --- COSTRUZIONE DOCUMENTI (da FTP, in RAM) ---
def _build_documents_from_ftp(materia: str | None):
    """
    Legge i PDF direttamente da FTP in RAM e costruisce una lista di Document.
    - Se materia == "Tutte le materie" o None: indicizza tutte le sottocartelle.
    - Altrimenti solo la sottocartella selezionata.
    """
    base_dir = _get_secret("FTP_DIR", "/")
    ftp = _ftp_connect()

    # individua la root reale e le materie disponibili (sottocartelle)
    base_dir, remote_materie = _ftp_list_dirs(ftp, base_dir)

    targets = []
    if not materia or materia == "Tutte le materie":
        # tutte le sottocartelle
        for m in remote_materie:
            targets.append(base_dir.rstrip("/") + "/" + m)
        # in più: anche eventuali PDF direttamente dentro base_dir
        targets.append(base_dir)
    else:
        # solo la sottocartella scelta (se esiste), fallback a base_dir/materia
        chosen = base_dir.rstrip("/") + "/" + materia
        targets = [chosen]

    documents = []
    for path in targets:
        try:
            pdfs = _ftp_iter_pdfs(ftp, path)
        except Exception:
            continue
        for remote_path, name in pdfs:
            try:
                buf = _ftp_read_file_bytes(ftp, remote_path)
                reader = PdfReader(buf)
                text_pages = []
                for page in reader.pages:
                    try:
                        text_pages.append(page.extract_text() or "")
                    except Exception:
                        text_pages.append("")
                text = "\n".join(text_pages).strip()
                if not text:
                    continue
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": remote_path, "filename": name, "materia": materia or "Tutte"}
                    )
                )
            except Exception as e:
                st.warning(f"Impossibile leggere {name} da FTP: {e}")

    try:
        ftp.quit()
    except Exception:
        pass

    return documents, remote_materie, base_dir

# --- VECTORSTORE (on-demand, senza cache disco) ---
def create_vectorstore_from_ftp(materia="Tutte le materie"):
    docs, _, _ = _build_documents_from_ftp(materia)
    if not docs:
        st.error("⚠️ Nessun documento trovato nella materia selezionata (via FTP).")
        st.stop()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    if not chunks:
        st.error("⚠️ Nessun contenuto testuale utile trovato nei documenti (via FTP).")
        st.stop()

    # API key per embeddings/LLM
    api_key = _get_secret("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY mancante nei secrets.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = api_key

    # === Embeddings modello aggiornato ===
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.from_documents(chunks, embeddings)

# --- STREAMLIT APP ---
load_dotenv()
local_css("style.css")

st.set_page_config(page_title="Chatbot Materie", page_icon="📚")
st.markdown("<div class='logo-container'><img src='https://i.postimg.cc/5NqjQ63K/images.jpg'></div>", unsafe_allow_html=True)
st.title("📚 Chatbot Educativo")
st.markdown("Interroga i documenti **remoti via FTP** per materia. Max 5 domande al giorno per codice.")

# Diagnostica rapida
#with st.expander("Diagnostica"):
#    st.write("FTP_HOST:", _get_secret("FTP_HOST", "(non impostato)"))
#    st.write("FTP_DIR:", _get_secret("FTP_DIR", "/"))
#    st.write("OPENAI_API_KEY presente:", bool(_get_secret("OPENAI_API_KEY")))

# Login semplice con codice
user_code = st.text_input("🔐 Inserisci il tuo codice corsista:")
if user_code and user_code not in VALID_CODES:
    st.error("❌ Codice non valido.")
    st.stop()

if user_code:
    count = check_quota(user_code)
    if count >= MAX_QUESTIONS_PER_DAY:
        st.warning("⛔ Hai raggiunto il limite giornaliero. Riprova domani.")
        st.stop()

    # Elenco materie dal server FTP
    try:
        ftp = _ftp_connect()
        base_dir, materie_remoto = _ftp_list_dirs(ftp, _get_secret("FTP_DIR", "/"))
        try:
            ftp.quit()
        except Exception:
            pass
    except Exception as e:
        st.error(f"Errore FTP: {e}")
        st.stop()

    # UI materie (SOLO le sottocartelle di Docs, niente "Tutte le materie")
    label_overrides = {
        "Educazione cinofila": "📘 Educazione Cinofila",
        "Istruzione cinofila": "📗 Istruzione Cinofila",
    }
    
    pretty_labels = [label_overrides.get(m, m) for m in materie_remoto]
    display_to_real = {label_overrides.get(m, m): m for m in materie_remoto}
    
            if pretty_labels:
            st.markdown("<div class='fade-in-1'>", unsafe_allow_html=True)
            materia_scelta_display = st.selectbox(
                "📁 Scegli la materia:",
                pretty_labels,
                index=None,
                placeholder="Seleziona una materia..."
            )
            st.markdown("</div>", unsafe_allow_html=True)
        
            if materia_scelta_display:
                materia_scelta = display_to_real[materia_scelta_display]
            else:
                st.stop()
        else:
            st.error("Nessuna materia trovata nella cartella Docs.")
            st.stop()
        
        # Domanda (con fade-in ritardato)
        st.markdown("<div class='fade-in-2'>", unsafe_allow_html=True)
        user_question = st.text_input("✍️ Fai la tua domanda:")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if user_question:
            with st.spinner("Sto cercando nei materiali remoti..."):
                # Vectorstore
                vectorstore = create_vectorstore_from_ftp(materia_scelta)


            # === Retriever limitato per prompt più corto ===
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

            # === LLM modello aggiornato ===
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever
            )

            try:
                response = qa.run(user_question)
            except Exception as e:
                st.error(f"Errore del modello (controlla modello/API key/segreti): {e}")
                st.stop()

            increment_quota(user_code)
            st.success(response)
