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


# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="AI Document Assistant",
    layout="wide"
)


st.title("📄 AI Document Assistant")


# -----------------------------
# LOAD EMBEDDING MODEL
# -----------------------------

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2",
        device="cpu"
    )


# -----------------------------
# TEXT EXTRACTION
# -----------------------------

def extract_pdf(file):

    results=[]

    reader=PdfReader(file)

    for page_no,page in enumerate(
        reader.pages,
        start=1
    ):

        text=page.extract_text()

        if text:

            results.append({

                "text":text,

                "filename":file.name,

                "page":page_no

            })

    return results



def extract_docx(file):

    doc=Document(file)

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

    elif name.endswith(".docx"):
        return extract_docx(file)

    elif name.endswith(".txt"):
        return extract_txt(file)

    elif name.endswith(".md"):
        return extract_md(file)

    return []



# -----------------------------
# CHUNKING
# -----------------------------

def create_chunks(
    documents,
    size=400,
    overlap=80
):

    chunks=[]


    for doc in documents:


        words=doc["text"].split()


        start=0


        while start < len(words):

            end=start+size


            chunk=" ".join(
                words[start:end]
            )


            chunks.append({

                "text":chunk,

                "filename":
                doc["filename"],

                "page":
                doc["page"]

            })


            start += size-overlap


    return chunks



# -----------------------------
# CREATE DATABASE
# -----------------------------

def build_database(chunks):


    model=load_embedding_model()


    texts=[
        c["text"]
        for c in chunks
    ]


    vectors=model.encode(
        texts,
        show_progress_bar=False
    )


    vectors=np.array(
        vectors
    ).astype("float32")


    index=faiss.IndexFlatL2(
        vectors.shape[1]
    )


    index.add(vectors)


    return index



def save_database(index,chunks):

    with open(
        "database.pkl",
        "wb"
    ) as f:

        pickle.dump(
            {
                "index":index,
                "chunks":chunks
            },
            f
        )



def load_database():

    if os.path.exists(
        "database.pkl"
    ):

        with open(
            "database.pkl",
            "rb"
        ) as f:

            return pickle.load(f)

    return None



# -----------------------------
# SEARCH
# -----------------------------

def semantic_search(
    question,
    database,
    k=5
):

    model=load_embedding_model()


    vector=model.encode(
        [question]
    )


    vector=np.array(
        vector
    ).astype("float32")


    distances,ids=database["index"].search(
        vector,
        k
    )


    results=[]


    for i in ids[0]:

        results.append(
            database["chunks"][i]
        )


    return results



def keyword_score(question,text):

    keywords=re.findall(
        r"\w+",
        question.lower()
    )


    score=0


    text=text.lower()


    for word in keywords:

        if len(word)>3 and word in text:

            score+=1


    return score



def hybrid_search(
    question,
    database
):


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
            (score,r)
        )


    ranked.sort(
        reverse=True,
        key=lambda x:x[0]
    )


    return [
        x[1]
        for x in ranked
    ]



# -----------------------------
# GROQ
# -----------------------------


def ask_groq(question,context):


    key=st.secrets.get(
        "GROQ_API_KEY"
    )


    if not key:

        st.error(
            "GROQ_API_KEY missing"
        )

        st.stop()



    client=Groq(
        api_key=key
    )


    prompt=f"""

Answer only using this context.

If the answer is not available,
say:
"I could not find this information
in the provided documents."


CONTEXT:

{context}


QUESTION:

{question}

"""


    response=client.chat.completions.create(

        model="llama-3.1-8b-instant",

        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ]

    )


    return response.choices[0].message.content



# -----------------------------
# USER INTERFACE
# -----------------------------


files=st.file_uploader(
    "Upload PDF, DOCX, TXT or MD",
    type=[
        "pdf",
        "docx",
        "txt",
        "md"
    ],
    accept_multiple_files=True
)



if st.button("Process Documents"):


    docs=[]


    for file in files:

        docs.extend(
            extract_document(file)
        )


    st.success(
        f"{len(docs)} document sections extracted"
    )


    chunks=create_chunks(
        docs
    )


    st.success(
        f"{len(chunks)} chunks created"
    )


    index=build_database(
        chunks
    )


    save_database(
        index,
        chunks
    )


    st.session_state.database={
        "index":index,
        "chunks":chunks
    }



database=st.session_state.get(
    "database"
)


if database:


    question=st.text_input(
        "Ask a question"
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
            "Retrieved Sources"
        )


        for r in results:

            st.write(
                f"📄 {r['filename']} | Page: {r['page']}"
            )

            st.caption(
                r["text"]
            )
