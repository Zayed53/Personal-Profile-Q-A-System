
import re
from typing import List, Dict
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "/data/aboutme.txt"
CHROMA_DB_DIR = "/vector_store/chroma_profile_db"
COLLECTION_NAME = "profile"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# -----------------------------
# SMART SECTION-AWARE CHUNKING
# -----------------------------
def extract_sections(text: str) -> List[Dict[str, str]]:
    # Pattern to match section headers (all caps, often standalone line)
    section_pattern = r'^([A-Z][A-Z\s/&-]+)$'
    
    lines = text.split('\n')
    sections = []
    current_section = None
    current_content = []
    
    for line in lines:
        line_stripped = line.strip()
        
        # Check if this line is a section header
        if re.match(section_pattern, line_stripped) and len(line_stripped) > 2:
            # Save previous section if exists
            if current_section is not None:
                sections.append({
                    'section': current_section,
                    'content': '\n'.join(current_content).strip()
                })
            
            # Start new section
            current_section = line_stripped
            current_content = [line_stripped]
        else:
            # Add to current section
            if current_content:  # Only add if we're in a section
                current_content.append(line)
    
   
    if current_section is not None and current_content:
        sections.append({
            'section': current_section,
            'content': '\n'.join(current_content).strip()
        })
    
    return sections


def create_smart_chunks(sections: List[Dict[str, str]], max_chunk_size: int = 1000) -> List[Document]:
    chunks = []
    
    for section_data in sections:
        section_name = section_data['section']
        content = section_data['content']
        
        # If section is small enough, keep it as one chunk
        if len(content) <= max_chunk_size:
            chunks.append(Document(
                page_content=content,
                metadata={
                    'section': section_name,
                    'source': DATA_PATH
                }
            ))
        else:
            
            subsection_pattern = r'\n\n(?=[A-Z])'  
            subsections = re.split(subsection_pattern, content)
            
            if len(subsections) > 1:
                
                current_chunk = section_name + "\n\n"  
                
                for subsection in subsections:
                    subsection = subsection.strip()
                    if not subsection:
                        continue
                    
                    
                    if len(current_chunk) + len(subsection) > max_chunk_size and len(current_chunk) > len(section_name) + 2:
                        chunks.append(Document(
                            page_content=current_chunk.strip(),
                            metadata={
                                'section': section_name,
                                'source': DATA_PATH
                            }
                        ))
                        current_chunk = f"{section_name}\n\n"  
                    
                    current_chunk += subsection + "\n\n"
                
               
                if len(current_chunk.strip()) > len(section_name):
                    chunks.append(Document(
                        page_content=current_chunk.strip(),
                        metadata={
                            'section': section_name,
                            'source': DATA_PATH
                        }
                    ))
            else:
               
                chunks.append(Document(
                    page_content=content,
                    metadata={
                        'section': section_name,
                        'source': DATA_PATH,
                        'note': 'Large single section'
                    }
                ))
    
    return chunks


# -----------------------------
# MAIN INDEXING PROCESS
# -----------------------------
if __name__ == "__main__":
    print("="*70)
    print("SMART PROFILE INDEXING")
    print("="*70)
    
    
    print(f"\n Loading document from: {DATA_PATH}")
    loader = TextLoader(DATA_PATH, encoding="utf-8")
    raw_docs = loader.load()
    text = raw_docs[0].page_content
    
    print(f"Loaded {len(text)} characters")
    
   
    print("\n Extracting sections...")
    sections = extract_sections(text)
    print(f"Found {len(sections)} sections:")
    for i, section in enumerate(sections, 1):
        print(f"   {i}. {section['section']} ({len(section['content'])} chars)")
    
    
    print("\n  Creating smart chunks...")
    chunks = create_smart_chunks(sections, max_chunk_size=1000)
    print(f" Created {len(chunks)} chunks")
    
    # Create embeddings
    print("\nCreating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )
    print("Embeddings ready")
    
   
    print(f"\nCreating Chroma vector store at: {CHROMA_DB_DIR}")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name=COLLECTION_NAME
    )
    
    vector_db.persist()
    
    print("\n" + "="*70)
    print("INDEXING COMPLETE!")
    print("="*70)
    print(f"Total chunks: {len(chunks)}")
    print(f"Database location: {CHROMA_DB_DIR}")
    print(f"Collection name: {COLLECTION_NAME}")
    
   