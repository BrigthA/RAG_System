import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv

# ---------------------------------------------------------
# LangChain imports
# ---------------------------------------------------------

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from langchain_openai import ChatOpenAI

from langsmith import traceable


# =========================================================
# CONFIGURATION
# =========================================================

load_dotenv()

APP_TITLE = "College Course RAG System"

VECTOR_DB_PATH = "faiss_college_index"

UPLOAD_DIR = "uploaded_documents"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

TOP_K = 8


# =========================================================
# STREAMLIT CONFIG
# =========================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎓",
    layout="wide",
)


# =========================================================
# CREATE DIRECTORIES
# =========================================================

os.makedirs(UPLOAD_DIR, exist_ok=True)


# =========================================================
# LANGSMITH CONFIGURATION
# =========================================================

# These can also be configured through .env

if os.getenv("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_TRACING"] = "true"

if os.getenv("LANGSMITH_PROJECT"):
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")


# =========================================================
# SESSION STATE
# =========================================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "indexed_documents" not in st.session_state:
    st.session_state.indexed_documents = []

if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    return embeddings


# =========================================================
# LLM
# =========================================================

@st.cache_resource
def load_llm():

    if not os.getenv("OPENAI_API_KEY"):

        st.error(
            "OPENAI_API_KEY is not configured. "
            "Add it to your .env file."
        )

        st.stop()

    llm = ChatOpenAI(
        model=os.getenv(
            "OPENAI_MODEL",
            "gpt-4o-mini"
        ),
        temperature=0,
    )

    return llm


# =========================================================
# DOCUMENT LOADER
# =========================================================

def load_single_file(
    file_path: str,
    original_filename: str
) -> List[Document]:

    extension = Path(file_path).suffix.lower()

    documents = []

    # -----------------------------------------------------
    # PDF
    # -----------------------------------------------------

    if extension == ".pdf":

        loader = PyPDFLoader(
            file_path,
            mode="page"
        )

        pages = loader.load()

        for page_number, doc in enumerate(pages, start=1):

            content = doc.page_content.strip()

            if not content:
                continue

            doc.metadata["source"] = original_filename

            doc.metadata["file_name"] = original_filename

            doc.metadata["page"] = page_number

            doc.metadata["file_type"] = "PDF"

            documents.append(doc)

    # -----------------------------------------------------
    # TXT
    # -----------------------------------------------------

    elif extension == ".txt":

        loader = TextLoader(
            file_path,
            encoding="utf-8"
        )

        docs = loader.load()

        for doc in docs:

            doc.metadata["source"] = original_filename
            doc.metadata["file_name"] = original_filename
            doc.metadata["page"] = 1
            doc.metadata["file_type"] = "TXT"

            documents.append(doc)

    # -----------------------------------------------------
    # DOCX
    # -----------------------------------------------------

    elif extension == ".docx":

        loader = Docx2txtLoader(file_path)

        docs = loader.load()

        for doc in docs:

            doc.metadata["source"] = original_filename
            doc.metadata["file_name"] = original_filename
            doc.metadata["page"] = 1
            doc.metadata["file_type"] = "DOCX"

            documents.append(doc)

    else:

        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    return documents


# =========================================================
# LOAD MULTIPLE DOCUMENTS
# =========================================================

def load_uploaded_files(uploaded_files):

    all_documents = []

    progress = st.progress(
        0,
        text="Loading documents..."
    )

    total_files = len(uploaded_files)

    for index, uploaded_file in enumerate(
        uploaded_files
    ):

        file_path = os.path.join(
            UPLOAD_DIR,
            uploaded_file.name
        )

        with open(
            file_path,
            "wb"
        ) as f:

            f.write(
                uploaded_file.getbuffer()
            )

        try:

            documents = load_single_file(
                file_path,
                uploaded_file.name
            )

            all_documents.extend(
                documents
            )

        except Exception as e:

            st.error(
                f"Failed to process "
                f"{uploaded_file.name}: {e}"
            )

        progress.progress(
            (index + 1) / total_files,
            text=(
                f"Processing "
                f"{uploaded_file.name}"
            )
        )

    progress.empty()

    return all_documents


# =========================================================
# CHUNK DOCUMENTS
# =========================================================

def create_chunks(
    documents: List[Document]
) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
    )

    chunks = splitter.split_documents(
        documents
    )

    # -----------------------------------------------------
    # Add chunk number
    # -----------------------------------------------------

    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        chunk.metadata["chunk_index"] = index

        chunk.metadata["chunk_id"] = (
            f"{chunk.metadata.get('file_name', 'unknown')}"
            f"_chunk_{index}"
        )

    return chunks


# =========================================================
# CREATE FAISS INDEX
# =========================================================

@traceable(
    name="create_college_vector_index"
)
def create_vector_index(
    chunks: List[Document]
):

    embeddings = load_embeddings()

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    vectorstore.save_local(
        VECTOR_DB_PATH
    )

    return vectorstore


# =========================================================
# LOAD EXISTING FAISS INDEX
# =========================================================

def load_vector_index():

    index_file = os.path.join(
        VECTOR_DB_PATH,
        "index.faiss"
    )

    if not os.path.exists(
        index_file
    ):

        return None

    try:

        embeddings = load_embeddings()

        vectorstore = FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        return vectorstore

    except Exception as e:

        st.warning(
            f"Could not load existing "
            f"FAISS index: {e}"
        )

        return None


# =========================================================
# CLEAR VECTOR DATABASE
# =========================================================

def clear_vector_database():

    if os.path.exists(
        VECTOR_DB_PATH
    ):

        shutil.rmtree(
            VECTOR_DB_PATH
        )

    if os.path.exists(
        UPLOAD_DIR
    ):

        shutil.rmtree(
            UPLOAD_DIR
        )

    os.makedirs(
        UPLOAD_DIR,
        exist_ok=True
    )

    st.session_state.vectorstore = None

    st.session_state.indexed_documents = []

    st.session_state.total_chunks = 0


# =========================================================
# RETRIEVE DOCUMENTS
# =========================================================

@traceable(
    name="college_rag_retrieval"
)
def retrieve_documents(
    question: str,
    top_k: int = TOP_K
):

    vectorstore = (
        st.session_state.vectorstore
    )

    if vectorstore is None:

        return []

    documents = (
        vectorstore.similarity_search(
            question,
            k=top_k
        )
    )

    return documents


# =========================================================
# FORMAT SOURCES
# =========================================================

def format_sources(
    documents: List[Document]
):

    formatted_sources = []

    for doc in documents:

        source = doc.metadata.get(
            "file_name",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            "N/A"
        )

        chunk_index = doc.metadata.get(
            "chunk_index",
            "N/A"
        )

        formatted_sources.append(
            f"""
SOURCE:
Document: {source}
Page: {page}
Chunk Index: {chunk_index}

Content:
{doc.page_content}
"""
        )

    return "\n\n".join(
        formatted_sources
    )


# =========================================================
# RAG PROMPT
# =========================================================

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [

        (
            "system",
            """
You are a college admission and course
recommendation assistant.

Your job is to recommend courses and colleges
using ONLY the retrieved knowledge provided
below.

The knowledge base contains college admission
rules, courses, eligibility criteria, minimum
marks, subjects, fees, duration and other
course information.

IMPORTANT RULES:

1. Do not invent colleges or courses.
2. Do not invent eligibility criteria.
3. Use the student's marks when recommending courses.
4. Compare the student's marks against the retrieved
   eligibility criteria.
5. Clearly identify whether the student appears
   eligible based on the retrieved information.
6. If the retrieved documents do not contain enough
   information, say that the information is unavailable.
7. Mention the source document and page number.
8. Prefer courses for which the student satisfies
   the minimum eligibility.
9. If several courses are suitable, rank them.
10. Explain why each recommended course is suitable.

Return the answer in a structured format:

### Student Assessment

Student marks:
- Overall:
- Relevant subjects:

### Recommended Courses

| Rank | College | Course | Eligibility | Student Marks | Status |
|------|---------|--------|-------------|---------------|--------|

### Explanation

Explain why the courses are recommended.

### Source

For each important recommendation mention:

- Document
- Page
- Chunk

Retrieved information:

{context}
"""
        ),

        (
            "human",
            """
Student information:

{student_information}

Question:

{question}
"""
        )
    ]
)


# =========================================================
# RAG ANSWER
# =========================================================

@traceable(
    name="college_course_rag"
)
def generate_rag_answer(
    question: str,
    student_information: str
) -> Tuple[str, List[Document]]:

    documents = retrieve_documents(
        question,
        TOP_K
    )

    if not documents:

        return (
            "I could not find relevant information "
            "in the uploaded college documents.",
            []
        )

    context = format_sources(
        documents
    )

    prompt = RAG_PROMPT.format_messages(
        context=context,
        student_information=student_information,
        question=question
    )

    llm = load_llm()

    response = llm.invoke(
        prompt
    )

    return (
        response.content,
        documents
    )


# =========================================================
# DISPLAY SOURCE DOCUMENTS
# =========================================================

def display_sources(
    documents: List[Document]
):

    if not documents:

        return

    st.subheader(
        "📚 Retrieved Sources"
    )

    for index, doc in enumerate(
        documents,
        start=1
    ):

        file_name = doc.metadata.get(
            "file_name",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            "N/A"
        )

        chunk_index = doc.metadata.get(
            "chunk_index",
            "N/A"
        )

        chunk_id = doc.metadata.get(
            "chunk_id",
            "N/A"
        )

        title = (
            f"Source {index} | "
            f"{file_name} | "
            f"Page {page} | "
            f"Chunk {chunk_index}"
        )

        with st.expander(title):

            st.write(
                f"**Document:** {file_name}"
            )

            st.write(
                f"**Page:** {page}"
            )

            st.write(
                f"**Chunk Index:** {chunk_index}"
            )

            st.write(
                f"**Chunk ID:** {chunk_id}"
            )

            st.divider()

            st.write(
                doc.page_content
            )


# =========================================================
# UPLOAD TAB
# =========================================================

def upload_tab():

    st.header(
        "📚 College DataSet"
    )

    st.write(
        """
Upload college brochures, admission guides,
course catalogs, eligibility documents and
other PDF/DOCX/TXT files.
"""
    )

    uploaded_files = st.file_uploader(
        "Upload multiple documents",
        type=[
            "pdf",
            "docx",
            "txt"
        ],
        accept_multiple_files=True
    )

    col1, col2 = st.columns(2)

    with col1:

        build_index = st.button(
            "🚀 Build / Update FAISS Index",
            use_container_width=True
        )

    with col2:

        clear_index = st.button(
            "🗑️ Clear Knowledge Base",
            use_container_width=True
        )

    if clear_index:

        clear_vector_database()

        st.success(
            "Knowledge base cleared."
        )

        st.rerun()

    if build_index:

        if not uploaded_files:

            st.warning(
                "Please upload at least one document."
            )

            return

        with st.spinner(
            "Loading and processing documents..."
        ):

            documents = load_uploaded_files(
                uploaded_files
            )

            if not documents:

                st.error(
                    "No readable content found."
                )

                return

            st.info(
                f"Loaded {len(documents)} pages/documents."
            )

            chunks = create_chunks(
                documents
            )

            st.info(
                f"Created {len(chunks)} chunks."
            )

            vectorstore = create_vector_index(
                chunks
            )

            st.session_state.vectorstore = (
                vectorstore
            )

            st.session_state.total_chunks = (
                len(chunks)
            )

            st.session_state.indexed_documents = [
                doc.metadata.get(
                    "file_name",
                    "Unknown"
                )
                for doc in documents
            ]

        st.success(
            "✅ FAISS index created successfully."
        )

        st.subheader(
            "Index Statistics"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Documents",
                len(
                    set(
                        st.session_state
                        .indexed_documents
                    )
                )
            )

        with col2:

            st.metric(
                "Pages",
                len(documents)
            )

        with col3:

            st.metric(
                "Chunks",
                len(chunks)
            )

        st.subheader(
            "Indexed Documents"
        )

        for document in sorted(
            set(
                st.session_state
                .indexed_documents
            )
        ):

            st.write(
                f"📄 {document}"
            )


# =========================================================
# CHAT TAB
# =========================================================

def chat_tab():

    st.header(
        "🎓 Student Course Recommendation"
    )

    # -----------------------------------------------------
    # Load existing vector index automatically
    # -----------------------------------------------------

    if st.session_state.vectorstore is None:

        existing_index = load_vector_index()

        if existing_index:

            st.session_state.vectorstore = (
                existing_index
            )

            st.info(
                "Loaded existing FAISS knowledge base."
            )

    if st.session_state.vectorstore is None:

        st.warning(
            """
No knowledge base is available.

Please go to the **Upload Documents** tab
and build the FAISS index first.
"""
        )

        return

    # -----------------------------------------------------
    # Student Information
    # -----------------------------------------------------

    st.subheader(
        "👨‍🎓 Student Information"
    )

    col1, col2 = st.columns(2)

    with col1:

        student_name = st.text_input(
            "Student Name",
            placeholder="Example: Rahul"
        )

        total_marks = st.number_input(
            "Overall / Total Marks",
            min_value=0.0,
            max_value=1000.0,
            value=0.0,
            step=1.0
        )

        percentage = st.number_input(
            "Percentage",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.1
        )

    with col2:

        maths = st.number_input(
            "Mathematics Mark",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0
        )

        physics = st.number_input(
            "Physics Mark",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0
        )

        chemistry = st.number_input(
            "Chemistry Mark",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0
        )

    # -----------------------------------------------------
    # Chat question
    # -----------------------------------------------------

    st.subheader(
        "💬 Ask the College Assistant"
    )

    question = st.text_area(
        "Question",
        placeholder=(
            "Example:\n"
            "Which engineering courses can I apply for?\n\n"
            "Recommend the best courses based on "
            "my marks.\n\n"
            "Which colleges am I eligible for?"
        ),
        height=130
    )

    ask_button = st.button(
        "🔍 Find Suitable Courses",
        type="primary",
        use_container_width=True
    )

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

            return

        student_information = f"""
Student Name: {student_name}

Overall Marks: {total_marks}

Percentage: {percentage}%

Mathematics: {maths}/100

Physics: {physics}/100

Chemistry: {chemistry}/100
"""

        with st.spinner(
            "Searching college documents..."
        ):

            answer, sources = (
                generate_rag_answer(
                    question,
                    student_information
                )
            )

        st.subheader(
            "🎯 Recommendation"
        )

        st.markdown(
            answer
        )

        display_sources(
            sources
        )


# =========================================================
# MAIN APPLICATION
# =========================================================

def main():

    st.title(
        "🎓 College Course Recommendation RAG"
    )

    st.caption(
        "LangChain + FAISS + Hugging Face Embeddings "
        "+ LangSmith + Streamlit"
    )

    # -----------------------------------------------------
    # Sidebar
    # -----------------------------------------------------

    with st.sidebar:

        st.header(
            "⚙️ RAG Configuration"
        )

        st.write(
            f"**Embedding Model:**"
        )

        st.code(
            EMBEDDING_MODEL
        )

        st.write(
            f"**Chunk Size:** {CHUNK_SIZE}"
        )

        st.write(
            f"**Chunk Overlap:** {CHUNK_OVERLAP}"
        )

        st.write(
            f"**Top K:** {TOP_K}"
        )

        st.divider()

        if st.session_state.vectorstore:

            st.success(
                "🟢 FAISS Index Loaded"
            )

            st.write(
                f"Chunks: "
                f"{st.session_state.total_chunks}"
            )

        else:

            st.warning(
                "🔴 No FAISS Index"
            )

        st.divider()

        st.write(
            "LangSmith tracing:"
        )

        if os.getenv(
            "LANGSMITH_API_KEY"
        ):

            st.success(
                "Enabled"
            )

        else:

            st.info(
                "Disabled"
            )

    # -----------------------------------------------------
    # Tabs
    # -----------------------------------------------------

    upload_container, chat_container = st.tabs(
        [
            "📚 Upload Documents",
            "🎓 Student Course Assistant"
        ]
    )

    with upload_container:

        upload_tab()

    with chat_container:

        chat_tab()


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    main()