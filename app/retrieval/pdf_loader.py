from pypdf import PdfReader


def load_pdf(path: str) -> str:

    reader = PdfReader(path)

    text = []

    for page in reader.pages:
        text.append(page.extract_text())

    return "\n".join(text)


def load_pdf_with_pages(path: str):

    reader = PdfReader(path)

    pages = []

    for i, page in enumerate(reader.pages):
        print(f"Processing page {i + 1} of {len(reader.pages)}")
        text = page.extract_text()
        pages.append({
            "page": i + 1,
            "text": text
        })

    return pages