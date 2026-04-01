from typing import List
import re


def split_into_sentences(text: str) -> List[str]:
    # simple sentence splitter
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def split_by_sections(text: str) -> List[str]:
    sections = re.split(r'\n\s*\d+\.\s+', text)
    return [s.strip() for s in sections if s.strip()]


def chunk_text(
    text: str,
    max_tokens: int = 200,
    overlap: int = 2
) -> List[str]:

    sentences = split_into_sentences(text)

    chunks = []
    current_chunk = []
    start = 0

    for sentence in sentences:

        current_chunk.append(sentence)

        # approximate token count by word count
        token_count = sum(len(s.split()) for s in current_chunk)

        if token_count >= max_tokens:
            chunks.append(" ".join(current_chunk))

            # keep overlap sentences
            current_chunk = current_chunk[-overlap:]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def chunk_document(text: str) -> List[str]:

    sections = split_by_sections(text)

    all_chunks = []

    for section in sections:
        chunks = chunk_text(section)
        all_chunks.extend(chunks)

    return all_chunks