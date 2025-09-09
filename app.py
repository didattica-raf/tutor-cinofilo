
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

# ---- FTP SYNC (diagnostica): prova più root tipiche e mostra info utili ----
def sync_from_ftp():
    """
    Sincronizza ricorsivamente i file dal tuo hosting FTP/FTPS dentro DOCS_ROOT.
    - Prova la cartella indicata in FTP_DIR e, se vuota/non valida, scandisce anche
      root comuni dei provider: /public_html, /www, /web, /htdocs, /
    - Scarica solo estensioni utili (PDF in primis). Le altre sono opzionali.
    - Mostra in UI un riepilogo dei percorsi testati e, se necessario, esempi di file ignorati.
    """
    from ftplib import FTP, FTP_TLS

    host = _get_secret("FTP_HOST")
    port = int(_get_secret("FTP_PORT", "21"))
    user = _get_secret("FTP_USER")
    password = _get_secret("FTP_PASS")
    ftp_dir = (_get_secret("FTP_DIR", "/") or "/").rstrip("/") or "/"

    if not all([host, user, password]):
        raise RuntimeError("Config FTP mancante: FTP_HOST / FTP_USER / FTP_PASS.")

    # Estensioni che vale la pena sincronizzare
    exts = {".pdf", ".docx", ".txt", ".md", ".pptx", ".xlsx", ".csv", ".rtf", ".odt"}

    def should_get(name: str) -> bool:
        return Path(name).suffix.lower() in exts

    # Connessione: tenta FTPS (TLS esplicito) e fallback a FTP plain
    def connect_ftps_then_ftp():
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

    # Listing con MLSD se disponibile, altrimenti LIST
    def list_dir(ftp, path):
        items = []
        try:
            lines = []
            ftp.retrlines(f"MLSD {path}", lines.append)
            for e in lines:
                facts, _, fname = e.partition(" ")
                factmap = dict(f.split("=", 1) for f in facts.rstrip(";").split(";") if "=" in f)
                kind = factmap.get("type", "file")
                items.append((fname, kind))
        except Exception:
            lines = []
            ftp.retrlines(f"LIST {path}", lines.append)
            for line in lines:
                parts = line.split(maxsplit=8)
                if not parts:
                    continue
                kind = "dir" if parts[0].startswith("d") else "file"
                fname = parts[-1]
                items.append((fname, kind))
        return items

    def ftp_walk(ftp, root):
        yield (root, [f for f, k in list_dir(ftp, root) if k != "dir"])
        for name, kind in list_dir(ftp, root):
            if kind == "dir" and name not in (".", ".."):
                sub = root.rstrip("/") + "/" + name
                yield from ftp_walk(ftp, sub)

    ftp = connect_ftps_then_ftp()

    # 1) Prova la cartella indicata
    candidate_roots = []
    try:
        ftp.cwd(ftp_dir)
        candidate_roots.append(ftp_dir)
    except Exception:
        pass

    # 2) Aggiungi root tipiche dei provider
    for r in ["/public_html", "/www", "/web", "/htdocs", "/"]:
        try:
            ftp.cwd(r)
            if r not in candidate_roots:
                candidate_roots.append(r)
        except Exception:
            continue

    st.caption(f"[FTP] Percorsi testati: {', '.join(candidate_roots)}")

    total_found = 0
    total_downloaded = 0
    ignored_examples = []

    for root in candidate_roots:
        try:
            for dirpath, files in ftp_walk(ftp, root):
                # Costruisci path locale relativo
                if dirpath == "/":
                    rel = ""
                elif dirpath.startswith(root):
                    rel = dirpath[len(root):].lstrip("/")
                else:
                    rel = dirpath.strip("/")

                local_dir = Path(DOCS_ROOT) / rel if rel else Path(DOCS_ROOT)
                local_dir.mkdir(parents=True, exist_ok=True)

                for name in files:
                    total_found += 1
                    remote_path = dirpath.rstrip("/") + "/" + name
                    if should_get(name):
                        local_path = local_dir / name
                        with open(local_path, "wb") as f:
                            ftp.retrbinary(f"RETR " + remote_path, f.write)
                        total_downloaded += 1
                    else:
                        if len(ignored_examples) < 10:
                            ignored_examples.append(remote_path)
        except Exception as e:
            st.warning(f"[FTP] Ignorato {root} per errore: {e}")

    try:
        ftp.quit()
    except Exception:
        pass

    if total_downloaded == 0:
        if ignored_examples:
            st.info("Esempi di file ignorati (estensioni non incluse):\n- " + "\n- ".join(ignored_examples))
        raise RuntimeError(
            f"Nessun file scaricato. Trovati {total_found} elementi totali. "
            "Controlla che i PDF siano in /public_html (o imposta correttamente FTP_DIR)."
        )

    return total_downloaded

# ---- RAG PIPELINE (minimal) ----
@st.cache_resource(show_spinner=False)
def build_vectorstore():
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS

    docs = []
    # Carichiamo SOLO PDF per l'indicizzazione (estendi se ti serve)
    for p in Path(DOCS_ROOT).rglob("*.pdf"):
        try:
            docs.extend(PyPDFLoader(str(p)).load())
        except Exception as e:
            st.warning(f"Errore nel leggere {p.name}: {e}")

    if not docs:
        raise RuntimeError("Nessun documento PDF trovato in docs/.")

    splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    api_key = _get_secret("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY mancante nei secrets.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = api_key

    embeddings = OpenAIEmbeddings()
    vs = FAISS.from_documents(chunks, embeddings)
    return vs

def main():
    st.set_page_config(page_title="Tutor Cinofilo – RAG FTP", page_icon="🐶")
    st.title("Tutor Cinofilo – Documenti da Hosting FTP")

    # Pannello diagnostico
    with st.expander("Diagnostica FTP"):
        st.write("Host:", _get_secret("FTP_HOST"))
        st.write("Porta:", _get_secret("FTP_PORT", "21"))
        st.write("User:", _get_secret("FTP_USER"))
        st.write("Dir preferita (FTP_DIR):", _get_secret("FTP_DIR", "/"))

    with st.spinner("Sincronizzo i materiali dal tuo hosting..."):
        try:
            n = sync_from_ftp()
            st.success(f"Sincronizzati {n} file dal tuo hosting.")
        except Exception as e:
            st.error(f"Errore di sincronizzazione: {e}")
            st.stop()

    with st.spinner("Creo l'indice dei documenti (PDF)..."):
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
