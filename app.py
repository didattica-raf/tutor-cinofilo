
import streamlit as st
import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# --- CONFIG ---
DOCS_ROOT = "docs"
USERS_FILE = "users.json"
VALID_CODES = ["2", "3", "5", "8", "13", "21", "34", "55", "89", "144"]
MAX_QUESTIONS_PER_DAY = 5

# --- FUNZIONI DI SUPPORTO ---
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def load_users():
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
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

def load_documents_from_folder(folder_path):
    docs = []
    for pdf_path in Path(folder_path).rglob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load_and_split())
    return docs

@st.cache_resource
def create_vectorstore(materiale="Tutte le materie"):
    folder = DOCS_ROOT if materiale == "Tutte le materie" else os.path.join(DOCS_ROOT, materiale)
    docs = load_documents_from_folder(folder)
    if not docs:
        st.error("⚠️ Nessun documento trovato nella materia selezionata.")
        st.stop()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    if not chunks:
        st.error("⚠️ Nessun contenuto testuale utile trovato nei documenti.")
        st.stop()
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(chunks, embeddings)

# --- STREAMLIT APP ---
load_dotenv()
local_css("style.css")

st.set_page_config(page_title="Chatbot Materie", page_icon="📚")
st.markdown("<div class='logo-container'><img src='https://github.com/didattica-raf/tutor-cinofilo/blob/main/logo.jpg'></div>", unsafe_allow_html=True)
st.title("📚 Chatbot Educativo")
st.markdown("Interroga i documenti caricati per materia. Max 5 domande al giorno per codice.")

user_code = st.text_input("🔐 Inserisci il tuo codice corsista:")
if user_code and user_code not in VALID_CODES:
    st.error("❌ Codice non valido.")
    st.stop()

if user_code:
    count = check_quota(user_code)
    if count >= MAX_QUESTIONS_PER_DAY:
        st.warning("⛔ Hai raggiunto il limite giornaliero. Riprova domani.")
        st.stop()

    # Selezione materia

    materie_raw = sorted([f.name for f in Path(DOCS_ROOT).iterdir() if f.is_dir()])
    materia_labels = {name: name for name in materie_raw}
    materia_labels.update({
        "Educazione cinofila": "📘 Educazione Cinofila",
        "Istruzione cinofila": "📗 Istruzione Cinofila"
    })
    materie = ["Tutte le materie"] + [materia_labels[m] for m in materia_labels]

    materia_scelta = st.selectbox("📁 Scegli la materia:", materie)
    label_to_folder = {v: k for k, v in materia_labels.items()}
    materia_folder = label_to_folder.get(materia_scelta, materia_scelta)

    user_question = st.text_input("✍️ Fai la tua domanda:")
    if user_question:
        with st.spinner("Sto cercando nei materiali..."):
            vectorstore = create_vectorstore(materia_folder)
            qa = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
                chain_type="stuff",
                retriever=vectorstore.as_retriever()
            )
            response = qa.run(user_question)
            increment_quota(user_code)
            st.success(response)
