import ollama
from typing import List, Dict, Tuple
from collections import defaultdict

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

# -----------------------------
# CONFIG
# -----------------------------
CHROMA_DB_DIR = "../vector_store_claude/chroma_profile_db"
COLLECTION_NAME = "profile"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# Local Llama model
OLLAMA_MODEL = "llama3.1:latest"

# -----------------------------
# EMBEDDINGS
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)

# -----------------------------
# VECTOR DB
# -----------------------------
vector_db = Chroma(
    persist_directory=CHROMA_DB_DIR,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)

# -----------------------------
# BM25 RETRIEVER
# -----------------------------
def load_all_docs_from_chroma(vector_db) -> List[Document]:
    raw_data = vector_db._collection.get(include=["documents", "metadatas"])
    docs = raw_data["documents"]
    metadatas = raw_data.get("metadatas", [{}] * len(docs))
    return [
        Document(page_content=text, metadata=meta if meta else {})
        for text, meta in zip(docs, metadatas)
    ]

all_docs = load_all_docs_from_chroma(vector_db)
bm25_retriever = BM25Retriever.from_documents(all_docs)
bm25_retriever.k = 8

# Vector retriever
vector_retriever = vector_db.as_retriever(search_kwargs={"k": 8})

# -----------------------------
# ADVANCED QUERY EXPANSION
# -----------------------------
DOMAIN_SYNONYMS = {
    # Education keywords
    'education': ['degree', 'university', 'academic', 'study', 'studied', 'graduate', 
                  'graduation', 'school', 'college', 'bachelor', 'bsc', 'cgpa', 'gpa'],
    'graduate': ['graduation', 'graduated', 'degree', 'university', 'completed'],
    'degree': ['education', 'bachelor', 'bsc', 'qualification', 'university'],
    'university': ['college', 'school', 'institution', 'academic', 'iut', 'islamic university'],
    
    # Work keywords
    'work': ['job', 'company', 'employer', 'position', 'role', 'employment', 'career'],
    'company': ['employer', 'organization', 'firm', 'workplace'],
    'current': ['now', 'present', 'currently', 'latest', 'today'],
    'experience': ['work history', 'career', 'background', 'employment', 'worked at', 'jobs'],
    # Skills keywords
    'skills': ['technologies', 'programming', 'languages', 'tools', 'expertise', 
               'knowledge', 'proficiency'],
    'programming': ['coding', 'development', 'software', 'languages'],
    'language': ['c++', 'python', 'javascript', 'java', 'c#', 'sql', 'php'],

     'about': ['who is', 'tell me about', 'background', 'introduction', 'profile', 'overview'],
    'contact': ['email', 'phone', 'reach', 'number', 'address', 'connect'],
    'interest': ['hobby', 'hobbies', 'likes', 'enjoys', 'personal', 'free time', 'passion'],
    'philosophy': ['approach', 'mindset', 'thinking', 'belief', 'engineering mindset', 'values'],
}

def expand_query_advanced(query: str) -> List[str]:
    """Advanced query expansion with domain knowledge"""
    query_lower = query.lower()
    variations = [query]
    
    relevant_synonyms = set()
    for keyword, synonyms in DOMAIN_SYNONYMS.items():
        if keyword in query_lower:
            relevant_synonyms.update(synonyms[:5])
    
    if relevant_synonyms:
        expanded = f"{query} {' '.join(list(relevant_synonyms)[:8])}"
        variations.append(expanded)
        
        keywords = query_lower.split()
        keyword_query = ' '.join(keywords + list(relevant_synonyms)[:4])
        variations.append(keyword_query)
    
    return variations[:3]

# -----------------------------
# QUERY VARIATIONS (NO LLM!)
# -----------------------------
def generate_query_variations(question: str) -> List[str]:
    """Generate query variations using rules (instant!)"""
    variations = expand_query_advanced(question)
    question_lower = question.lower()
    
    # Pattern matching
    if question_lower.startswith('what is'):
        variations.append(question_lower.replace('what is', '').strip())
    
    if 'where' in question_lower and 'work' in question_lower:
        variations.append("current job company employer position")
    # Remove duplicates
    seen = set()
    unique = []
    for v in variations:
        if v.lower() not in seen:
            seen.add(v.lower())
            unique.append(v)
    
    return unique[:5]

# -----------------------------
# SECTION-AWARE INTENT DETECTION
# -----------------------------
def detect_query_intent(query: str) -> str:
    """Detect which section the query is asking about"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['education', 'degree', 'university', 'graduate', 'cgpa', 'study']):
        return 'EDUCATION'
    elif any(word in query_lower for word in ['work', 'job', 'company', 'employer', 'position']):
        return 'WORK EXPERIENCE'
    elif any(word in query_lower for word in ['skill', 'programming', 'language', 'technology', 'tool']):
        return 'SKILLS'
    elif any(word in query_lower for word in ['project', 'built', 'created', 'developed']):
        return 'PROJECTS'
    return None

# -----------------------------
# RRF WITH SECTION BOOSTING
# -----------------------------
def reciprocal_rank_fusion_with_boost(
    doc_lists: List[List[Document]], 
    query_intent: str = None,
    k: int = 60
) -> List[Tuple[Document, float]]:
    """RRF with intelligent section boosting"""
    doc_scores = defaultdict(float)
    doc_objects = {}
    
    for doc_list in doc_lists:
        for rank, doc in enumerate(doc_list, start=1):
            doc_id = doc.page_content
            
            # Base RRF score
            base_score = 1.0 / (k + rank)
            
            # Apply section boost
            section = doc.metadata.get('section', '')
            boost = 1.0
            
            if query_intent and section == query_intent:
                boost = 2.0  # Strong boost for matching section
            # elif section in SECTION_BOOST:
            #     boost = SECTION_BOOST[section]
            
            final_score = base_score * boost
            doc_scores[doc_id] += final_score
            doc_objects[doc_id] = doc
    
    sorted_items = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_objects[doc_id], score) for doc_id, score in sorted_items]

# -----------------------------
# HYBRID RETRIEVER
# -----------------------------
def hybrid_retriever(question: str, top_k: int = 5) -> List[Document]:
    """Ultimate hybrid retrieval"""
    
    # Detect intent
    query_intent = detect_query_intent(question)
    
    # Generate variations
    queries = generate_query_variations(question)
    
    # Retrieve from both methods
    all_doc_lists = []
    
    for query in queries:
        try:
            vector_docs = vector_retriever.invoke(query)
            all_doc_lists.append(vector_docs)
        except Exception as e:
            pass
        
        try:
            bm25_docs = bm25_retriever.invoke(query)
            all_doc_lists.append(bm25_docs)
        except Exception as e:
            pass
    
    # Fuse with RRF + boosting
    fused_results = reciprocal_rank_fusion_with_boost(
        all_doc_lists, 
        query_intent=query_intent,
        k=60
    )
    
    return [doc for doc, score in fused_results[:top_k]]

# -----------------------------
# DIRECT OLLAMA ANSWER GENERATION
# -----------------------------
SYSTEM_PROMPT = """You are a knowledgeable assistant answering questions about ##NAME## Hasan.

STRICT RULES:
1. Answer ONLY using the provided context documents
2. If the question is not about ##NAME## Hasan, respond: "Not related to ##NAME## Hasan."
3. If the answer is not in the context, respond: "I don't have information about that."
4. Be specific but do NOT reference document numbers or mention any documents in your answer
5. Keep answers concise but complete
6. Create Complete Sentences Or Statements
6. NEVER make up information
7. Do not mention that you are an AI or reference any source documents in your response"""

def format_docs(docs: List[Document]) -> str:
    """Format documents for context"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        section = doc.metadata.get('section', 'Unknown')
        formatted.append(f"[Document {i} - Section: {section}]\n{doc.page_content}")
    return "\n\n".join(formatted)

def answer_question(question: str) -> Tuple[str, List[Document]]:
    """
    Main RAG pipeline using direct Ollama call
    
    Uses ollama.chat() instead of server API
    """
    
    # Retrieve documents
    docs = hybrid_retriever(question, top_k=5)
    
    if not docs:
        return "I don't have information about that.", []
    
    # Format context
    context = format_docs(docs)

    
    # Direct Ollama call - NO SERVER OVERHEAD!
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    'role': 'system',
                    'content': SYSTEM_PROMPT
                },
                {
                    'role': 'user',
                    'content': f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                }
            ],
            options={
                'temperature': 0.0,
                'num_predict': 512,  # Max tokens
            }
        )
        
        answer = response['message']['content']
        return answer, docs
        
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        print("Make sure Ollama is installed and the model is downloaded:")
        print(f"  ollama pull {OLLAMA_MODEL}")
        return f"Error: {e}", docs

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    print("="*80)
    print("="*80)
    print(f"Model: {OLLAMA_MODEL}")
    print("="*80)
  
    
    # Interactive mode
    print("\n" + "="*80)
    print("Commands:")
    print("  - Type your question for normal response")
    print("  - Type 'stream <question>' for streaming response")
    print("  - Type 'exit' to quit")
    print("="*80)
    
    while True:
        user_input = input("\nYour input: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        question = user_input
        print("\n" + "-"*80)
        answer, _ = answer_question(question)
        print(f"\n{answer}")
        print("-"*80)
