import streamlit as st
import os
import pickle
import re

import numpy as np

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

DATABASE_FILE = "document_database.pkl"


# ==========================
# EMBEDDING MODEL
# ==========================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )



# ==========================
# DOCUMENT EXTRACTION
# ==========================


def extract_pdf(file):

    data=[]

    reader = PdfReader(file)


    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text:

            data.append({

                "text": text,

                "filename": file.name,

                "page": page_number

            })


    return data



def extract_docx(file):

    doc = Document(file)

    text="\n".join(
        p.text for p in doc.paragraphs
    )


    return [{
        "text": text,
        "filename": file.name,
        "page": None
    }]



def extract_txt(file):

    text=file.read().decode(
        "utf-8"
    )


    return [{
        "text": text,
        "filename": file.name,
        "page": None
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
        "text": text,
        "filename": file.name,
        "page": None
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
# EMBEDDINGS DATABASE
# ==========================


def create_database(chunks):


    model=load_embedding_model()


    texts=[
        c["text"]
        for c in chunks
    ]


    embeddings=model.encode(
        texts,
        normalize_embeddings=True
    )


    return {

        "chunks": chunks,

        "embeddings": embeddings

    }



def save_database(database):

    with open(
        DATABASE_FILE,
        "wb"
    ) as f:

        pickle.dump(
            database,
            f
        )



def load_database():

    if os.path.exists(
        DATABASE_FILE
    ):

        with open(
            DATABASE_FILE,
            "rb"
        ) as f:

            return pickle.load(f)

    return None



# ==========================
# HYBRID SEARCH
# ==========================


def semantic_search(
        question,
        database,
        top_k=8):


    model=load_embedding_model()


    query_embedding=model.encode(
        [question],
        normalize_embeddings=True
    )[0]


    scores=np.dot(
        database["embeddings"],
        query_embedding
    )


    best_indexes=np.argsort(
        scores
    )[::-1][:top_k]


    return [
        database["chunks"][i]
        for i in best_indexes
    ]



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


    for item in results:


        score=keyword_score(
            question,
            item["text"]
        )


        ranked.append(
            (
                score,
                item
            )
        )


    ranked.sort(
        key=lambda x:x[0],
        reverse=True
    )


    return [
        x[1]
        for x in ranked[:5]
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

        return None



    client=Groq(
        api_key=api_key
    )


    prompt=f"""

You are an AI document assistant.

Answer ONLY from the provided context.

If the answer is not available,
say:

"I could not find this information
in the provided documents."


DOCUMENT CONTEXT:

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

    "Upload PDF, DOCX, TXT, MD",

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
        "Document database ready"
    )


    question=st.text_input(
        "Ask your question"
    )


    if question:


        sources=hybrid_search(
            question,
            database
        )


        context="\n\n".join(
            s["text"]
            for s in sources
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


        for source in sources:

            st.write(
                f"📄 {source['filename']} | Page: {source['page']}"
            )

            st.caption(
                source["text"]
            )
