from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.logging import get_logger
from app.data.vector_store import vector_store_manager
from app.schemas.chat_schema import IngestResponse


logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["ingest"])


async def _extract_text_from_pdf(file: UploadFile) -> str:
    import io

    from PyPDF2 import PdfReader

    content = io.BytesIO(await file.read())
    reader = PdfReader(content)
    texts: List[str] = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts)


async def _extract_text_from_html(file: UploadFile) -> str:
    from bs4 import BeautifulSoup

    raw = (await file.read()).decode("utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")
    return soup.get_text(separator="\n")


@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(file: UploadFile = File(...)) -> IngestResponse:
    filename = file.filename or "uploaded"
    if filename.lower().endswith(".pdf"):
        text = await _extract_text_from_pdf(file)
    elif filename.lower().endswith((".html", ".htm")):
        text = await _extract_text_from_html(file)
    else:
        raise HTTPException(
            status_code=400,
            detail="Format file tidak didukung. Gunakan PDF atau HTML.",
        )

    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="Tidak ada teks yang dapat diekstrak dari dokumen.",
        )

    num_chunks = vector_store_manager.ingest_texts(
        texts=[text],
        source=filename,
    )

    logger.info(
        "Document ingested",
        extra={
            "document_filename": filename,
            "document_num_chunks": num_chunks,
        },
    )

    return IngestResponse(source=filename, num_chunks=num_chunks)
