import streamlit as st
import os
import pickle
import re

import numpy as np
import faiss

from pypdf import PdfReader
from docx import Document

from sentence_transformers import SentenceTransformer

from groq import Groq

import markdown
from bs4 import BeautifulSoup


# ==========================
# CONFIG
# ==========================

st.set_page_config(
    page_title="AI Document Assistant",
    layout="wide"
)

st.title("📄 AI Document Assistant")

DB_FILE = "document_database.pkl"


# ==========================
# MODEL LOADING
# ==========================

@st.cache_resource
def get_embedding_model():

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    )


# ==========================
# EXTRACTION FUNCTIONS
# ==========================


def extract_pdf(file):

    data=[]

    reader = PdfReader(file)

    for page_no,page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text:

            data.append({

                "text": text,

                "filename": file.name,

                "page": page_no

            })


    return data



def extract_docx(file):

    doc = Document(file)

    text="\n".join(
        p.text for p in doc.paragraphs
    )


    return [{

        "text":text,

        "filename":file.name,

        "page":None

    }]



def extract_txt(file):

    text=file.read().decode(
        "utf-8"
    )


    return [{

        "text":text,

        "filename":file.name,

        "page":None

    }]



def extract_md(file):

    raw=file.read().decode(
        "utf-8"
    )

    html=markdown.markdown(raw)

    text=BeautifulSoup(
        html,
        "html.parser"
    ).get_text()


    return [{

        "text":text,

        "filename":file.name,

        "page":None

    }]



def extract_document(file):

    name=file.name.lower()


    if name.endswith(".pdf"):
        return extract_pdf(file)

    if name.endswith(".docx"):
        return extract_docx(file)

    if name.endswith(".txt"):
        return extract_txt(file)

    if name.endswith(".md"):
        return extract_md(file)


    return []



# ==========================
# CHUNKING
# ==========================


def create_chunks(
        documents,
        chunk_size=400,
        overlap=80):


    chunks=[]


    for doc in documents:

        words=doc["text"].split()

        start=0


        while start < len(words):

            end=start+chunk_size


            chunks.append({

                "text":
                " ".join(words[start:end]),

                "filename":
                doc["filename"],

                "page":
                doc["page"]

            })


            start += chunk_size-overlap


    return chunks



# ==========================
# EMBEDDINGS + FAISS
# ==========================


def create_database(chunks):

    model=get_embedding_model()


    texts=[
        c["text"]
        for c in chunks
    ]


    embeddings=model.encode(
        texts,
        show_progress_bar=False
    )


    embeddings=np.array(
        embeddings
    ).astype("float32")


    index=faiss.IndexFlatL2(
        embeddings.shape[1]
    )


    index.add(
        embeddings
    )


    return {

        "index":index,

        "chunks":chunks

    }



def save_database(database):

    with open(
        DB_FILE,
        "wb"
    ) as f:

        pickle.dump(
            database,
            f
        )



def load_database():

    if os.path.exists(DB_FILE):

        with open(
            DB_FILE,
            "rb"
        ) as f:

            return pickle.load(f)


    return None



# ==========================
# SEARCH
# ==========================


def semantic_search(
        question,
        database,
        k=8):


    model=get_embedding_model()


    query_embedding=model.encode(
        [question]
    )


    query_embedding=np.array(
        query_embedding
    ).astype("float32")


    distances,ids = database["index"].search(
        query_embedding,
        k
    )


    results=[]


    for i in ids[0]:

        results.append(
            database["chunks"][i]
        )


    return results



def keyword_score(
        question,
        text):

    keywords=re.findall(
        r"\w+",
        question.lower()
    )


    score=0


    text=text.lower()


    for word in keywords:

        if len(word)>3 and word in text:

            score +=1


    return score



def hybrid_search(
        question,
        database):


    results=semantic_search(
        question,
        database
    )


    ranked=[]


    for r in results:

        score=keyword_score(
            question,
            r["text"]
        )


        ranked.append(
            (
                score,
                r
            )
        )


    ranked.sort(
        key=lambda x:x[0],
        reverse=True
    )


    return [
        item[1]
        for item in ranked[:5]
    ]



# ==========================
# GROQ
# ==========================


def ask_groq(
        question,
        context):


    api_key=st.secrets.get(
        "GROQ_API_KEY"
    )


    if not api_key:

        st.error(
            "GROQ_API_KEY missing"
        )

        return



    client=Groq(
        api_key=api_key
    )


    prompt=f"""

You are a document assistant.

Answer ONLY from the context below.

If the answer is not present,
say:

"I could not find this information
in the provided documents."


CONTEXT:

{context}


QUESTION:

{question}

"""


    response=client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]

    )


    return response.choices[0].message.content



# ==========================
# STREAMLIT UI
# ==========================


uploaded_files=st.file_uploader(

    "Upload Documents",

    type=[
        "pdf",
        "docx",
        "txt",
        "md"
    ],

    accept_multiple_files=True

)



if st.button("Process Documents"):


    documents=[]


    for file in uploaded_files:

        documents.extend(
            extract_document(file)
        )


    st.success(
        f"{len(documents)} sections extracted"
    )


    chunks=create_chunks(
        documents
    )


    st.success(
        f"{len(chunks)} chunks created"
    )


    database=create_database(
        chunks
    )


    save_database(
        database
    )


    st.session_state.database=database



database=st.session_state.get(
    "database"
)


if database is None:

    database=load_database()



if database:


    st.success(
        "Database ready"
    )


    question=st.text_input(
        "Ask your question"
    )


    if question:


        results=hybrid_search(
            question,
            database
        )


        context="\n\n".join(
            r["text"]
            for r in results
        )


        answer=ask_groq(
            question,
            context
        )


        st.subheader(
            "Answer"
        )

        st.write(
            answer
        )


        st.subheader(
            "Sources"
        )


        for r in results:

            st.write(
                f"📄 {r['filename']} | Page {r['page']}"
            )

            st.caption(
                r["text"]
            )
