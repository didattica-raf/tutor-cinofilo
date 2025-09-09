
import os
from pathlib import Path
import streamlit as st

# ---- CONFIG ----
DOCS_ROOT = "docs"
Path(DOCS_ROOT).mkdir(parents=True, exist_ok=True)

# ---- SECRETS/ENV HELPER ----
def _get_secret(key: str, default: str | None = None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)

# ---- FTP SYNC (FTPS with fallback to FTP) ----
def sync_from_ftp():
    """
    Sync recursively from FTP/FTPS into DOCS_ROOT.
    Supported files: .pdf, .docx, .txt, .md
    """
    host = _get_secret("FTP_HOST")
    port = int(_get_secret("FTP_PORT", "21"))
    user = _get_secret("FTP_USER")
    password = _get_secret("FTP_PASS")
    remote_dir = _get_secret("FTP_DIR", "/")

    if not all([host, user, password]):
        raise RuntimeError("Missing FTP config: FTP_HOST / FTP_USER / FTP_PASS.")

    exts = {".pdf", ".docx", ".txt", ".md"}

    def should_get(name: str) -> bool:
        return Path(name).suffix.lower() in exts

    # Walk helper
    def ftp_walk(ftp, root):
        entries = []
        try:
            ftp.retrlines(f"MLSD {root}", entries.append)
            parsed = []
            for e in entries:
                facts, _, fname = e.partition(" ")
                factmap = dict(f.split("=", 1) for f in facts.rstrip(";").split(";") if "=" in f)
                parsed.append((fname, factmap.get("type", "file")))
        except Exception:
            lines = []
            ftp.retrlines(f"LIST {root}", lines.append)
            parsed = []
            for line in lines:
                parts = line.split(maxsplit=8)
                kind = "dir" if parts and parts[0].startswith("d") else "file"
                fname = parts[-1] if parts else ""
                if fname:
                    parsed.append((fname, kind))

        files, dirs = [], []
        for fname, kind in parsed:
            if fname in (".", ".."):
                continue
            if kind.lower() == "dir":
                dirs.append(fname)
            else:
                files.append(fname)

        yield (root, files)
        for d in dirs:
            sub = root.rstrip("/") + "/" + d
            yield from ftp_walk(ftp, sub)

    # Connect: try FTPS first
    def connect_ftps_then_ftp():
        try:
            from ftplib import FTP_TLS
            ftps = FTP_TLS()
            ftps.connect(host, port, timeout=30)
            ftps.login(user=user, passwd=password)
            try:
                ftps.prot_p()
            except Exception:
                pass
            return ftps
        except Exception:
            from ftplib import FTP
            ftp = FTP()
            ftp.connect(host, port, timeout=30)
            ftp.login(user=user, passwd=password)
            return ftp

    ftp = connect_ftps_then_ftp()

    # Ensure starting dir
    try:
        ftp.cwd(remote_dir)
        start_dir = remote_dir
    except Exception:
        start_dir = "/"

    count = 0
    for dirpath, files in ftp_walk(ftp, start_dir):
        rel = dirpath[len(start_dir):].lstrip("/") if dirpath.startswith(start_dir) else dirpath.strip("/")
        local_dir = Path(DOCS_ROOT) / rel if rel else Path(DOCS_ROOT)
        local_dir.mkdir(parents=True, exist_ok=True)

        for name in files:
            if not should_get(name):
                continue
            remote_path = dirpath.rstrip("/") + "/" + name
            local_path = local_dir / name
            with open(local_path, "wb") as f:
                ftp.retrbinary(f"RETR " + remote_path, f.write)
            count += 1

    try:
        ftp.quit()
    except Exception:
        pass

    if count == 0:
        raise RuntimeError("FTP sync finished but no eligible files were downloaded.")
    return count

# ---- RAG PIPELINE (minimal) ----
@st.cache_resource(show_spinner=False)
def build_vectorstore():
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS

    docs = []
    # Load PDFs (extend to other types if needed)
    for p in Path(DOCS_ROOT).rglob("*.pdf"):
        try:
            docs.extend(PyPDFLoader(str(p)).load())
        except Exception as e:
            st.warning(f"Errore nel leggere {p.name}: {e}")

    if not docs:
        raise RuntimeError("Nessun documento caricato in docs/.")

    splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    # Embeddings (needs OPENAI_API_KEY)
    api_key = _get_secret("OPENAI_API_KEY")
    if not api_key:
        st.stop()
    os.environ["OPENAI_API_KEY"] = api_key

    embeddings = OpenAIEmbeddings()

    vs = FAISS.from_documents(chunks, embeddings)
    return vs

def main():
    st.set_page_config(page_title="Tutor Cinofilo – RAG FTP", page_icon="🐶")
    st.title("Tutor Cinofilo – Documenti da Hosting FTP")

    with st.spinner("Sincronizzo i materiali dal tuo hosting..."):
        try:
            n = sync_from_ftp()
            st.success(f"Sincronizzati {n} file dal tuo hosting.")
        except Exception as e:
            st.error(f"Errore di sincronizzazione: {e}")
            st.stop()

    with st.spinner("Creo l'indice dei documenti..."):
        try:
            vs = build_vectorstore()
            st.success("Indice pronto.")
        except Exception as e:
            st.error(f"Errore nella creazione dell'indice: {e}")
            st.stop()

    query = st.text_input("Fai una domanda sui materiali:", "")
    if query.strip():
        from langchain.chains import RetrievalQA
        from langchain_community.chat_models import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

        with st.spinner("Sto cercando la risposta..."):
            out = qa({"query": query})

        st.write(out["result"])
        with st.expander("Fonti"):
            for d in out.get("source_documents", []):
                st.write("•", d.metadata.get("source", "sconosciuto"))

if __name__ == "__main__":
    main()
