import pymupdf

def extract_text_from_pdf(pdf_bytes:bytes)->str:
    """
    Extract embedded text from a PDF.

    Args:
        pdf_bytes: The contents of a PDF file.

    Returns:
        Extracted text as a string.
    """
    
    extracted_pages=[]
    
    with pymupdf.open(stream=pdf_bytes,filetype="pdf")as pdf:
        for page in pdf:
            text=page.get_text("text")
            extracted_pages.append(text)
            
    return "\n\n".join(extracted_pages).strip()