import os
import re
import json
import sqlite3
import pdfplumber
from foundry_local_sdk import Configuration, FoundryLocalManager


DB_NAME = "rag_database.db"
DOCS_FOLDER = "docs"
COURSE_REGEX = r'\b([a-zA-Z]{2,4})[\s_\-]*(\d{3}[a-zA-Z]?)\b'


def extract_course_info(text, file_name):
    """Belgenin başından veya dosya adından Ders Kodu ve Ders Adını tespit eder."""
    code_match = re.search(COURSE_REGEX, text[:1000]) or re.search(COURSE_REGEX, file_name)
    course_code = f"{code_match.group(1).upper()}{code_match.group(2)}" if code_match else "UNKNOWN"


    title_match = re.search(r'\b[a-zA-Z]{2,4}[\s_\-]*\d{3}[a-zA-Z]?[\s_\-:\–]*([A-Za-z\s]{3,40})', text[:1000])
    course_title = "Unknown Course"
    if title_match:
        raw_title = title_match.group(1).split('\n')[0].strip()
        if len(raw_title) > 3:
            course_title = raw_title


    return course_code, course_title


# ingestion.py içindeki clean_and_format_page fonksiyonunu şu şekilde güncelleyin:


def clean_and_format_page(page):
    # Metni doğrudan sayfa üzerindeki fiziksel pozisyonlarına göre çıkar
    text = page.extract_text(layout=True, x_tolerance=2, y_tolerance=2) or ""
   
    # Tabloları çizgisiz yapılar için esnek stratejiyle okuyalım
    tables = page.extract_tables({
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
        "snap_tolerance": 3
    })
   
    table_str = ""
    for table in tables:
        for row in table:
            if row:
                clean_row = [str(cell).replace('\n', ' ').strip() if cell else "" for cell in row]
                # Boş hücreleri eleyip araya net ayraç koyuyoruz
                if any(clean_row):
                    table_str += " | ".join(clean_row) + "\n"
        table_str += "\n"


    full_page_content = f"{text}\n\n--- TABLE DATA ---\n{table_str}" if table_str else text
    return full_page_content


def semantic_chunk_text(text, max_chars=1200, overlap=200):
    """Kelimeleri ve cümleleri bölmeden parçalar."""
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = ""


    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= max_chars:
            current_chunk += paragraph + "\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            overlap_str = current_chunk[-overlap:] if len(current_chunk) >= overlap else current_chunk
            current_chunk = overlap_str + paragraph + "\n"


    if current_chunk.strip():
        chunks.append(current_chunk.strip())


    return chunks


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            content TEXT,
            embedding TEXT
        )
    """)
    conn.commit()
    conn.close()


def main():
    init_db()
    config = Configuration(app_name="foundry_local_rag")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance


    print("Loading embedding model...")
    embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")
    embedding_model.load()
    embedding_client = embedding_model.get_embedding_client()


    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()


    for file_name in os.listdir(DOCS_FOLDER):
        file_path = os.path.join(DOCS_FOLDER, file_name)
        if not os.path.isfile(file_path) or not file_name.endswith(".pdf"):
            continue


        print(f"Processing PDF with pdfplumber: {file_name}")
       
        with pdfplumber.open(file_path) as pdf:
            if len(pdf.pages) == 0:
                continue
           
            # İlk sayfadan metadata çıkar
            first_page_text = pdf.pages[0].extract_text() or ""
            course_code, course_title = extract_course_info(first_page_text, file_name)


            for page_idx, page in enumerate(pdf.pages, start=1):
                cleaned_text = clean_and_format_page(page)
                if not cleaned_text.strip():
                    continue


                chunks = semantic_chunk_text(cleaned_text, max_chars=1200, overlap=200)


                for chunk in chunks:
                    tagged_content = (
                        f"--- METADATA ---\n"
                        f"CourseCode: {course_code}\n"
                        f"CourseTitle: {course_title}\n"
                        f"SourceFile: {file_name}\n"
                        f"PageNumber: {page_idx}\n"
                        f"--- CONTENT ---\n"
                        f"{chunk}"
                    )


                    response = embedding_client.generate_embedding(tagged_content)
                    embedding = response.data[0].embedding


                    cursor.execute(
                        "INSERT INTO documents (file_name, content, embedding) VALUES (?, ?, ?)",
                        (file_name, tagged_content, json.dumps(embedding))
                    )


    conn.commit()
    conn.close()
    embedding_model.unload()
    print("\nIngestion complete with Table Extraction!")


if __name__ == "__main__":
    main()
