import json
import re
import random

# Configuration for chunking
MAX_WORDS = 200            # target max words per chunk
MIN_WORDS = 50             # optional: min words to consider a valid chunk
OVERLAP_SENTENCES = 2      # number of sentences to overlap between chunks

# Improved Patterns for splitting headings at line start
CHAPTER_REGEX = re.compile(r"(?m)^(Chương\s+[IVX]+)\b")
ARTICLE_REGEX = re.compile(r"(?m)^(Điều\s+\d+)\b")


def split_by_heading(text, pattern):
    """
    Split text by headings matching pattern at start of line, keeping the heading with each section.
    Returns list of (heading, content) tuples.
    """
    parts = pattern.split(text)
    if len(parts) < 3:
        return [(None, text.strip())]

    sections = []
    for i in range(1, len(parts), 2):
        heading = parts[i].strip()
        content = parts[i+1].strip()
        sections.append((heading, content))
    return sections


def split_sentences(text):
    """
    Naive sentence splitter that ends on ., ?, !, ; or newline.
    Keeps delimiters.
    """
    text = text.replace("\r\n", "\n")
    sentence_end = re.compile(r'(?<=[\.\?\!;])\s+|\n+')
    parts = sentence_end.split(text)
    return [s.strip() for s in parts if s.strip()]


def chunk_sentences(sentences, max_words, overlap_sents):
    """
    Group list of sentences into chunks that end at sentence boundary,
    each chunk around max_words, with overlap by sentences.
    Returns list of text chunks.
    """
    chunks = []
    n = len(sentences)
    idx = 0

    while idx < n:
        start_idx = idx
        word_count = 0
        while idx < n and word_count + len(sentences[idx].split()) <= max_words:
            word_count += len(sentences[idx].split())
            idx += 1
        if idx == start_idx:
            idx += 1
        chunk_sents = sentences[start_idx:idx]
        chunk_text = ' '.join(chunk_sents)
        if len(chunk_text.split()) < MIN_WORDS and chunks:
            chunks[-1] += ' ' + chunk_text
        else:
            chunks.append(chunk_text)
        next_idx = start_idx + max(1, len(chunk_sents) - overlap_sents)
        idx = next_idx

    return chunks


def process_document(doc):
    text = doc.get("Nội dung văn bản", "")
    doc_id = doc.get("Số hiệu")
    effective_date = doc.get("Ngày có hiệu lực")
    field = doc.get("Lĩnh vực")

    chapters = split_by_heading(text, CHAPTER_REGEX)
    all_chunks = []

    for chap_heading, chap_content in chapters:
        chap_title = chap_heading or ''
        articles = split_by_heading(chap_content, ARTICLE_REGEX)
        for art_heading, art_content in articles:
            art_title = art_heading or ''
            full_text = f"{chap_title}\n{art_heading}\n{art_content}" if chap_title else f"{art_heading}\n{art_content}"
            sentences = split_sentences(full_text)
            for i, chunk in enumerate(chunk_sentences(sentences, MAX_WORDS, OVERLAP_SENTENCES)):
                all_chunks.append({
                    "id": random.randint(100000, 999999),
                    "doc_id": doc_id,
                    "effective_date": effective_date,
                    "field": field,
                    "chapter": chap_title,
                    "article": art_title,
                    "chunk_index": i,
                    "text": chunk
                })

    if not chapters:
        sentences = split_sentences(text)
        for i, chunk in enumerate(chunk_sentences(sentences, MAX_WORDS, OVERLAP_SENTENCES)):
            all_chunks.append({
                "id": random.randint(100000, 999999),
                "doc_id": doc.get("Số hiệu"),
                "effective_date": doc.get("Ngày có hiệu lực"),
                "field": doc.get("Lĩnh vực"),
                "chapter": None,
                "article": None,
                "chunk_index": i,
                "text": chunk
            })

    return all_chunks

# === Entry point for direct usage without terminal arguments ===
def run_chunking(input_path, output_path):
    """
    Run the chunking process for given input/output paths without using argparse.
    """
    all_chunks = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            all_chunks.extend(process_document(doc))

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for chunk in all_chunks:
            f_out.write(json.dumps(chunk, ensure_ascii=False) + '\n')


# Example usage:
if __name__ == '__main__':
    # Simply set your file paths here and run the script in any IDE or double-click
    input_file = 'data_all.jsonl'         # <-- chỉnh đường dẫn file đầu vào
    output_file = 'new_chunked_data.jsonl' # <-- chỉnh đường dẫn file đầu ra
    run_chunking(input_file, output_file)
    print(f"Chunking completed: {output_file}")
