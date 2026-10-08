
===========================================================================
              RAG SYSTEM COMPARISON
===========================================================================

========== PDF LOADING ==========
Pages loaded : 2
Time         : 0.2012 sec

========== CHUNKING ==========
Chunks created : 3
Time           : 0.0005 sec

========== EMBEDDING MODEL ==========
Model: sentence-transformers/all-MiniLM-L6-v2
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 103/103 [00:00<00:00, 3634.95it/s]

========== FAISS INDEX ==========
Creating new FAISS index...
Time : 0.4833 sec

======================================
           INGESTION COMPLETE
======================================
PDF loading : 0.2012 sec
Chunking    : 0.0005 sec
FAISS       : 0.4833 sec
Total       : 44.4360 sec

Enter your question: What is microflow?

========== RETRIEVAL ==========
Query       : What is microflow?
Documents   : 3
Time        : 0.0245 sec

========== BASIC RAG ==========
Microflows in Mendix execute primarily on the server and are commonly used for business logic, database operations, integrations, and server-side processing (Source 1, Page 0).

Timing:
Retrieval  : 0.0245 sec
Generation : 2.8842 sec
Total      : 2.9107 sec

======================================
              SELF-RAG
======================================

========== RETRIEVAL ==========
Query       : What is microflow?
Documents   : 3
Time        : 0.0098 sec
Relevant documents: 1/3

Self-RAG decided retrieval was weak.
Rewritten query: What is the concept of microflow in the context of process management?

========== RETRIEVAL ==========
Query       : What is the concept of microflow in the context of process management?
Documents   : 3
Time        : 0.0185 sec

Answer:
Microflows in Mendix execute primarily on the server and are commonly used for business logic, database operations, integrations, and server-side processing (Source 1, Page 0).

Timing:
Initial retrieval : 0.0098 sec
Grading           : 3.7599 sec
Query rewriting   : 0.7930 sec
Second retrieval  : 0.0185 sec
Generation        : 1.6361 sec
Total             : 9.7076 sec

======================================
           CORRECTIVE RAG
======================================

========== RETRIEVAL ==========
Query       : What is microflow?
Documents   : 3
Time        : 0.0087 sec

Corrective RAG:
Initial retrieval is insufficient.
Corrected query: What is the definition and purpose of microflow in the context of process management?

========== RETRIEVAL ==========
Query       : What is the definition and purpose of microflow in the context of process management?
Documents   : 3
Time        : 0.0115 sec

Answer:
Microflows in Mendix are processes that execute primarily on the server. They are commonly used for business logic, database operations, integrations, and server-side processing. 

(Source: Page 0)

Timing:
Retrieval          : 0.0087 sec
Grading            : 2.1088 sec
Correction         : 0.8111 sec
Corrected retrieval: 0.0115 sec
Generation         : 1.1936 sec
Total              : 6.8130 sec

======================================
              FUSION RAG
======================================

Generated queries:
- 1. What are the key features and benefits of microflow in software development?
- 2. How does microflow differ from traditional workflow processes?
- 3. What are some practical applications of microflow in business automation?
- 4. Can you provide examples of tools or platforms that utilize microflow technology?

Fused documents: 3

Answer:
Microflows are processes in Mendix that primarily execute on the server. They are used for business logic, database operations, integrations, and server-side processing (Source 1, Page 0).

Timing:
Query generation : 1.1308 sec
Retrieval        : 0.0495 sec
Fusion            : 0.0000 sec
Generation        : 0.9203 sec
Total             : 2.1017 sec


===========================================================================
                 RAG PERFORMANCE
===========================================================================
RAG Type                 Total Time          Documents      
---------------------------------------------------------------------------
Basic RAG                2.9107              3              
Self-RAG                 9.7076              1              
Corrective RAG           6.8130              1              
Fusion RAG               2.1017              3              
===========================================================================


===========================================================================
Open LangSmith to inspect:
1. Retrieval
2. LLM calls
3. Grading
4. Query rewriting
5. Fusion
6. Execution time
===========================================================================
