import os
import time
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate

from langsmith import traceable


# ============================================================
# 1. ENVIRONMENT
# ============================================================

load_dotenv()

PDF_PATH = "Data/Documents.pdf"
FAISS_PATH = "faiss_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Change this through .env if required
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ============================================================
# 2. TIMER UTILITY
# ============================================================

class Timer:

    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        return time.perf_counter() - self.start_time


# ============================================================
# 3. LOAD PDF
# ============================================================

@traceable(name="Load PDF")
def load_pdf():

    timer = Timer()
    timer.start()

    print("\n========== PDF LOADING ==========")

    pdf_path = Path(PDF_PATH)
    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path.resolve()}"
        )

    with pdf_path.open("rb") as pdf_file:
        header = pdf_file.read(1024)

    if b"%PDF-" not in header:
        raise ValueError(
            f"{pdf_path.resolve()} does not appear to be a PDF "
            "(missing the %PDF- header). Set PDF_PATH to a valid PDF file."
        )

    loader = PyPDFLoader(str(pdf_path))

    documents = loader.load()

    elapsed = timer.stop()

    print(f"Pages loaded : {len(documents)}")
    print(f"Time         : {elapsed:.4f} sec")

    return documents, elapsed


# ============================================================
# 4. SPLIT DOCUMENT
# ============================================================

@traceable(name="Split Documents")
def split_documents(documents):

    timer = Timer()
    timer.start()

    print("\n========== CHUNKING ==========")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    elapsed = timer.stop()

    print(f"Chunks created : {len(chunks)}")
    print(f"Time           : {elapsed:.4f} sec")

    return chunks, elapsed


# ============================================================
# 5. EMBEDDING MODEL
# ============================================================

def create_embeddings():

    print("\n========== EMBEDDING MODEL ==========")

    print(f"Model: {EMBEDDING_MODEL}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embeddings


# ============================================================
# 6. CREATE / LOAD FAISS
# ============================================================

@traceable(name="Create FAISS Index")
def create_faiss(chunks, embeddings):

    timer = Timer()
    timer.start()

    print("\n========== FAISS INDEX ==========")

    index_path = Path(FAISS_PATH)

    if index_path.exists():

        print("Existing FAISS index found.")

        vector_store = FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

    else:

        print("Creating new FAISS index...")

        vector_store = FAISS.from_documents(
            chunks,
            embeddings
        )

        vector_store.save_local(FAISS_PATH)

    elapsed = timer.stop()

    print(f"Time : {elapsed:.4f} sec")

    return vector_store, elapsed


# ============================================================
# 7. LLM
# ============================================================

def create_llm():

    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0
    )


# ============================================================
# 8. BASIC RETRIEVAL
# ============================================================

@traceable(name="Basic Retrieval")
def retrieve_documents(
    vector_store,
    query: str,
    k: int = 5
):

    timer = Timer()
    timer.start()

    documents = vector_store.similarity_search(
        query,
        k=k
    )

    elapsed = timer.stop()

    print("\n========== RETRIEVAL ==========")
    print(f"Query       : {query}")
    print(f"Documents   : {len(documents)}")
    print(f"Time        : {elapsed:.4f} sec")

    return documents, elapsed


# ============================================================
# 9. FORMAT CONTEXT
# ============================================================

def format_context(documents):

    context = []

    for i, document in enumerate(documents):

        page = document.metadata.get(
            "page",
            "unknown"
        )

        context.append(
            f"""
--- SOURCE {i + 1} ---
Page: {page}

{document.page_content}
"""
        )

    return "\n".join(context)


# ============================================================
# 10. BASIC RAG
# ============================================================

@traceable(name="Basic RAG")
def basic_rag(
    query,
    vector_store,
    llm
):

    total_timer = Timer()
    total_timer.start()

    documents, retrieval_time = retrieve_documents(
        vector_store,
        query,
        k=5
    )

    context = format_context(documents)

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful RAG assistant.

Answer the question ONLY using the supplied context.

If the answer is not available in the context,
say:

"I could not find this information in the document."

Question:
{question}

Context:
{context}

Provide a concise answer and mention the relevant
source pages when possible.
"""
    )

    generation_timer = Timer()
    generation_timer.start()

    response = llm.invoke(
        prompt.format_messages(
            question=query,
            context=context
        )
    )

    generation_time = generation_timer.stop()

    total_time = total_timer.stop()

    answer = response.content

    print("\n========== BASIC RAG ==========")

    print(answer)

    print("\nTiming:")
    print(f"Retrieval  : {retrieval_time:.4f} sec")
    print(f"Generation : {generation_time:.4f} sec")
    print(f"Total      : {total_time:.4f} sec")

    return {
        "answer": answer,
        "documents": documents,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time,
        "total_time": total_time
    }


# ============================================================
# 11. SELF-RAG - RELEVANCE GRADER
# ============================================================

@traceable(name="Self RAG Relevance Grading")
def grade_documents(
    query,
    documents,
    llm
):

    relevant_documents = []

    grading_prompt = ChatPromptTemplate.from_template(
        """
You are a document relevance grader.

Determine whether the following document chunk
contains information useful for answering the question.

Question:
{question}

Document:
{document}

Return ONLY:

YES

or

NO
"""
    )

    for document in documents:

        response = llm.invoke(
            grading_prompt.format_messages(
                question=query,
                document=document.page_content
            )
        )

        result = response.content.strip().upper()

        if "YES" in result:

            relevant_documents.append(document)

    return relevant_documents


# ============================================================
# 12. QUERY REWRITING
# ============================================================

@traceable(name="Query Rewriting")
def rewrite_query(
    query,
    llm
):

    prompt = ChatPromptTemplate.from_template(
        """
Rewrite the user's question to make it better
for semantic document retrieval.

Original question:
{question}

Return only the rewritten question.
"""
    )

    response = llm.invoke(
        prompt.format_messages(
            question=query
        )
    )

    return response.content.strip()


# ============================================================
# 13. SELF-RAG
# ============================================================

@traceable(name="Self RAG")
def self_rag(
    query,
    vector_store,
    llm
):

    total_timer = Timer()
    total_timer.start()

    print("\n======================================")
    print("              SELF-RAG")
    print("======================================")

    # --------------------------------------------------------
    # Step 1 - Initial retrieval
    # --------------------------------------------------------

    documents, retrieval_time = retrieve_documents(
        vector_store,
        query,
        k=5
    )

    # --------------------------------------------------------
    # Step 2 - Grade retrieved documents
    # --------------------------------------------------------

    grade_timer = Timer()
    grade_timer.start()

    relevant_documents = grade_documents(
        query,
        documents,
        llm
    )

    grade_time = grade_timer.stop()

    print(
        f"Relevant documents: "
        f"{len(relevant_documents)}/{len(documents)}"
    )

    # --------------------------------------------------------
    # Step 3 - Self correction
    # --------------------------------------------------------

    rewrite_time = 0
    second_retrieval_time = 0

    if len(relevant_documents) < 2:

        print("\nSelf-RAG decided retrieval was weak.")

        rewrite_timer = Timer()
        rewrite_timer.start()

        new_query = rewrite_query(
            query,
            llm
        )

        rewrite_time = rewrite_timer.stop()

        print(f"Rewritten query: {new_query}")

        second_documents, second_retrieval_time = (
            retrieve_documents(
                vector_store,
                new_query,
                k=5
            )
        )

        second_relevant = grade_documents(
            new_query,
            second_documents,
            llm
        )

        if len(second_relevant) > len(
            relevant_documents
        ):
            relevant_documents = second_relevant

    # --------------------------------------------------------
    # Step 4 - Generate answer
    # --------------------------------------------------------

    context = format_context(
        relevant_documents
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a Self-RAG assistant.

Answer the question using ONLY the
retrieved context.

If the context is insufficient,
say that the document does not contain
enough information.

Question:
{question}

Context:
{context}

Answer with source page references.
"""
    )

    generation_timer = Timer()
    generation_timer.start()

    response = llm.invoke(
        prompt.format_messages(
            question=query,
            context=context
        )
    )

    generation_time = generation_timer.stop()

    total_time = total_timer.stop()

    answer = response.content

    print("\nAnswer:")
    print(answer)

    print("\nTiming:")
    print(f"Initial retrieval : {retrieval_time:.4f} sec")
    print(f"Grading           : {grade_time:.4f} sec")
    print(f"Query rewriting   : {rewrite_time:.4f} sec")
    print(
        f"Second retrieval  : "
        f"{second_retrieval_time:.4f} sec"
    )
    print(f"Generation        : {generation_time:.4f} sec")
    print(f"Total             : {total_time:.4f} sec")

    return {
        "answer": answer,
        "documents": relevant_documents,
        "retrieval_time": retrieval_time,
        "grading_time": grade_time,
        "rewrite_time": rewrite_time,
        "second_retrieval_time": second_retrieval_time,
        "generation_time": generation_time,
        "total_time": total_time
    }


# ============================================================
# 14. CORRECTIVE RAG
# ============================================================

@traceable(name="Corrective RAG")
def corrective_rag(
    query,
    vector_store,
    llm
):

    total_timer = Timer()
    total_timer.start()

    print("\n======================================")
    print("           CORRECTIVE RAG")
    print("======================================")

    # --------------------------------------------------------
    # Step 1 - Retrieve
    # --------------------------------------------------------

    documents, retrieval_time = retrieve_documents(
        vector_store,
        query,
        k=5
    )

    # --------------------------------------------------------
    # Step 2 - Evaluate retrieval
    # --------------------------------------------------------

    grading_timer = Timer()
    grading_timer.start()

    relevant_documents = grade_documents(
        query,
        documents,
        llm
    )

    grading_time = grading_timer.stop()

    # --------------------------------------------------------
    # Step 3 - Correct retrieval
    # --------------------------------------------------------

    correction_time = 0
    corrected_retrieval_time = 0

    if len(relevant_documents) < 3:

        print("\nCorrective RAG:")
        print("Initial retrieval is insufficient.")

        correction_timer = Timer()
        correction_timer.start()

        corrected_query = rewrite_query(
            query,
            llm
        )

        correction_time = correction_timer.stop()

        print(
            f"Corrected query: "
            f"{corrected_query}"
        )

        corrected_documents, corrected_retrieval_time = (
            retrieve_documents(
                vector_store,
                corrected_query,
                k=8
            )
        )

        corrected_relevant = grade_documents(
            corrected_query,
            corrected_documents,
            llm
        )

        # Combine useful documents

        combined = (
            relevant_documents
            + corrected_relevant
        )

        # Remove duplicates

        unique_documents = []

        seen = set()

        for document in combined:

            content_hash = hash(
                document.page_content
            )

            if content_hash not in seen:

                seen.add(content_hash)

                unique_documents.append(
                    document
                )

        relevant_documents = unique_documents

    # --------------------------------------------------------
    # Step 4 - Generate
    # --------------------------------------------------------

    context = format_context(
        relevant_documents
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a Corrective RAG system.

Use the corrected retrieval context
to answer the question.

Do not invent information.

Question:
{question}

Context:
{context}

If information is missing,
clearly say that it is not available.

Include source pages.
"""
    )

    generation_timer = Timer()
    generation_timer.start()

    response = llm.invoke(
        prompt.format_messages(
            question=query,
            context=context
        )
    )

    generation_time = generation_timer.stop()

    total_time = total_timer.stop()

    answer = response.content

    print("\nAnswer:")
    print(answer)

    print("\nTiming:")
    print(f"Retrieval          : {retrieval_time:.4f} sec")
    print(f"Grading            : {grading_time:.4f} sec")
    print(f"Correction         : {correction_time:.4f} sec")
    print(
        f"Corrected retrieval: "
        f"{corrected_retrieval_time:.4f} sec"
    )
    print(f"Generation         : {generation_time:.4f} sec")
    print(f"Total              : {total_time:.4f} sec")

    return {
        "answer": answer,
        "documents": relevant_documents,
        "retrieval_time": retrieval_time,
        "grading_time": grading_time,
        "correction_time": correction_time,
        "corrected_retrieval_time":
            corrected_retrieval_time,
        "generation_time": generation_time,
        "total_time": total_time
    }


# ============================================================
# 15. FUSION RAG - QUERY GENERATION
# ============================================================

@traceable(name="Fusion Query Generation")
def generate_queries(
    query,
    llm
):

    prompt = ChatPromptTemplate.from_template(
        """
Generate 4 different search queries for the
following user question.

Each query should approach the question
from a different perspective.

Original question:
{question}

Return exactly 4 queries,
one per line.
"""
    )

    response = llm.invoke(
        prompt.format_messages(
            question=query
        )
    )

    queries = [
        q.strip()
        for q in response.content.split("\n")
        if q.strip()
    ]

    return queries[:4]


# ============================================================
# 16. RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(
    result_lists,
    k=60
):

    scores = {}
    documents = {}

    for result_list in result_lists:

        for rank, document in enumerate(
            result_list,
            start=1
        ):

            document_id = hash(
                document.page_content
            )

            documents[document_id] = document

            if document_id not in scores:
                scores[document_id] = 0

            scores[document_id] += (
                1 / (k + rank)
            )

    ranked_ids = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    return [
        documents[document_id]
        for document_id in ranked_ids
    ]


# ============================================================
# 17. FUSION RAG
# ============================================================

@traceable(name="Fusion RAG")
def fusion_rag(
    query,
    vector_store,
    llm
):

    total_timer = Timer()
    total_timer.start()

    print("\n======================================")
    print("              FUSION RAG")
    print("======================================")

    # --------------------------------------------------------
    # Step 1 - Generate multiple queries
    # --------------------------------------------------------

    query_timer = Timer()
    query_timer.start()

    queries = generate_queries(
        query,
        llm
    )

    query_generation_time = (
        query_timer.stop()
    )

    print("\nGenerated queries:")

    for q in queries:
        print(f"- {q}")

    # --------------------------------------------------------
    # Step 2 - Search each query
    # --------------------------------------------------------

    retrieval_timer = Timer()
    retrieval_timer.start()

    result_lists = []

    for q in queries:

        docs = vector_store.similarity_search(
            q,
            k=5
        )

        result_lists.append(docs)

    retrieval_time = retrieval_timer.stop()

    # --------------------------------------------------------
    # Step 3 - Reciprocal Rank Fusion
    # --------------------------------------------------------

    fusion_timer = Timer()
    fusion_timer.start()

    fused_documents = reciprocal_rank_fusion(
        result_lists
    )

    fused_documents = fused_documents[:8]

    fusion_time = fusion_timer.stop()

    print(
        f"\nFused documents: "
        f"{len(fused_documents)}"
    )

    # --------------------------------------------------------
    # Step 4 - Generate answer
    # --------------------------------------------------------

    context = format_context(
        fused_documents
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a Fusion RAG assistant.

The context was created by combining
multiple retrieval strategies.

Answer the question using only the
provided context.

Question:
{question}

Context:
{context}

Give a clear answer and mention
source pages.
"""
    )

    generation_timer = Timer()
    generation_timer.start()

    response = llm.invoke(
        prompt.format_messages(
            question=query,
            context=context
        )
    )

    generation_time = generation_timer.stop()

    total_time = total_timer.stop()

    answer = response.content

    print("\nAnswer:")
    print(answer)

    print("\nTiming:")
    print(
        f"Query generation : "
        f"{query_generation_time:.4f} sec"
    )
    print(
        f"Retrieval        : "
        f"{retrieval_time:.4f} sec"
    )
    print(
        f"Fusion            : "
        f"{fusion_time:.4f} sec"
    )
    print(
        f"Generation        : "
        f"{generation_time:.4f} sec"
    )
    print(
        f"Total             : "
        f"{total_time:.4f} sec"
    )

    return {
        "answer": answer,
        "documents": fused_documents,
        "query_generation_time":
            query_generation_time,
        "retrieval_time": retrieval_time,
        "fusion_time": fusion_time,
        "generation_time": generation_time,
        "total_time": total_time
    }


# ============================================================
# 18. COMPARISON
# ============================================================

def print_comparison(results):

    print("\n")
    print("=" * 75)
    print("                 RAG PERFORMANCE")
    print("=" * 75)

    print(
        f"{'RAG Type':<25}"
        f"{'Total Time':<20}"
        f"{'Documents':<15}"
    )

    print("-" * 75)

    for name, result in results.items():

        print(
            f"{name:<25}"
            f"{result['total_time']:<20.4f}"
            f"{len(result['documents']):<15}"
        )

    print("=" * 75)


# ============================================================
# 19. MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 75)
    print("              RAG SYSTEM COMPARISON")
    print("=" * 75)

    total_timer = Timer()
    total_timer.start()

    # --------------------------------------------------------
    # INGESTION
    # --------------------------------------------------------

    documents, load_time = load_pdf()

    chunks, chunk_time = split_documents(
        documents
    )

    embeddings = create_embeddings()

    vector_store, faiss_time = create_faiss(
        chunks,
        embeddings
    )

    ingestion_time = total_timer.stop()

    print("\n======================================")
    print("           INGESTION COMPLETE")
    print("======================================")

    print(f"PDF loading : {load_time:.4f} sec")
    print(f"Chunking    : {chunk_time:.4f} sec")
    print(f"FAISS       : {faiss_time:.4f} sec")
    print(f"Total       : {ingestion_time:.4f} sec")

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    llm = create_llm()

    # --------------------------------------------------------
    # QUESTION
    # --------------------------------------------------------

    query = input(
        "\nEnter your question: "
    )

    # --------------------------------------------------------
    # BASIC RAG
    # --------------------------------------------------------

    basic_result = basic_rag(
        query,
        vector_store,
        llm
    )

    # --------------------------------------------------------
    # SELF RAG
    # --------------------------------------------------------

    self_result = self_rag(
        query,
        vector_store,
        llm
    )

    # --------------------------------------------------------
    # CORRECTIVE RAG
    # --------------------------------------------------------

    corrective_result = corrective_rag(
        query,
        vector_store,
        llm
    )

    # --------------------------------------------------------
    # FUSION RAG
    # --------------------------------------------------------

    fusion_result = fusion_rag(
        query,
        vector_store,
        llm
    )

    # --------------------------------------------------------
    # COMPARE
    # --------------------------------------------------------

    results = {

        "Basic RAG":
            basic_result,

        "Self-RAG":
            self_result,

        "Corrective RAG":
            corrective_result,

        "Fusion RAG":
            fusion_result
    }

    print_comparison(results)

    print("\n")
    print("=" * 75)
    print("Open LangSmith to inspect:")
    print("1. Retrieval")
    print("2. LLM calls")
    print("3. Grading")
    print("4. Query rewriting")
    print("5. Fusion")
    print("6. Execution time")
    print("=" * 75)


if __name__ == "__main__":
    main()