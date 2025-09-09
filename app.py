
import streamlit as st
from dotenv import load_dotenv
import os
import json
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Carica variabili ambiente
load_dotenv()
local_css("style.css")

# Mostra logo
st.markdown(
    """
    <div class='logo-container' style='text-align: center;' width=139>
        <img src='https://www.istitutoleopardi.it/wp-content/uploads/LEO-logo.gif' width='139'>
    </div>
    """,
    unsafe_allow_html=True
)

st.set_page_config(page_title="Chat del Corso", page_icon="https://www.istitutoleopardi.it/wp-content/uploads/Leo-favicon-1.gif")
st.title("💬 Chatbot dei POF dell'Istituto Leopardi")
st.markdown("❓ Fai una domanda basata sul materiale caricato nei POF dei vari licei. Max 5 domande al giorno per corsista.")

user_code = st.text_input("🔐 Inserisci il tuo codice corsista (es. numero personale):")

# Codici validi
VALID_CODES = ['2', '3', '5', '8', '13', '21', '34', '55', '89', '144']

if user_code and user_code not in VALID_CODES:
    st.error("⚠️ Codice non valido. Contatta l'amministratore.")
    st.stop()

USERS_FILE = "users.json"
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, "w") as f:
        json.dump({}, f)

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
    return users[user_code]["count"], today

def increment_quota(user_code):
    users = load_users()
    users[user_code]["count"] += 1
    save_users(users)

@st.cache_resource
def load_data():
    pdf_files = [
        "POFisttecnicoeconomico2024_25.pdf",
        "POFliceoeuropeo2024_25.pdf",
        "POFliceolinguistico2024_25.pdf",
        "POFliceoscienze_umane2024_25.pdf",
        "POFliceoscientifico2024_25.pdf"
    ]
    pages = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        pages.extend(loader.load_and_split())
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

vectorstore = load_data()
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

if user_code:
    count, today = check_quota(user_code)
    if count >= 5:
        st.warning("⛔ Hai già fatto 5 domande oggi. Riprova domani.")
        st.stop()

    st.text_input(
        "Scrivi qui la tua domanda:",
        key="user_question",
        on_change=lambda: st.session_state.pop("asked", None)
    )

    if "user_question" in st.session_state and st.session_state.user_question and "asked" not in st.session_state:
        with st.spinner("Sto cercando nei materiali del corso..."):
            result = qa.run(st.session_state.user_question)
            increment_quota(user_code)
            st.success(result)
        st.session_state.asked = True
        st.session_state.user_question = ""
