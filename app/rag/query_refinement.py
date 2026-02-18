from typing import List, TypedDict, Optional, Tuple, Set
import json
import replicate
import numpy as np
import re

from app.core.config import get_settings
from app.core.logging import get_logger, broadcast_event
from app.data.vector_store import vector_store_manager

logger = get_logger(__name__)


class RefinedQuery(TypedDict):
    original_query: str
    refined_query: str
    sub_queries: List[str]
    refinement_type: str


def _distance_to_conf(distance: float) -> float:
    return max(0.0, 1.0 - min(distance / 2.0, 1.0))


def _embed(text: str) -> List[float]:
    return vector_store_manager.embedding_model.embed_query(text)


def _cosine(a: List[float], b: List[float]) -> float:
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0
    return float(np.dot(va, vb) / denom)


def _top_distance_and_sources(q: str, k: int) -> Tuple[float, Set[str]]:
    results = vector_store_manager.similarity_search_with_scores(query=q, top_k=k)
    if not results:
        return 1.0, set()
    dists = [s for _, s in results]
    min_d = min(dists) if dists else 1.0
    srcs: Set[str] = set()
    for doc, _ in results:
        src = doc.metadata.get("source")
        if src:
            srcs.add(str(src))
    return float(min_d), srcs


def _norm_text(t: str) -> str:
    s = (t or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _retrieval_sims(q: str, k: int) -> List[float]:
    results = vector_store_manager.similarity_search_with_scores(query=q, top_k=k)
    if not results:
        return []
    sims: List[float] = []
    for _, dist in results:
        s = 1.0 - min(float(dist) / 2.0, 1.0)
        sims.append(max(0.0, min(1.0, s)))
    return sims


def _scan_json_objects(text: str) -> List[str]:
    objs: List[str] = []
    if not text:
        return objs
    s = text
    start = s.find("{")
    while start != -1:
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == "\"":
                    in_str = False
            else:
                if ch == "\"":
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        objs.append(s[start:i + 1])
                        start = s.find("{", i + 1)
                        break
        else:
            break
    return objs


def _extract_json_candidates(text: str) -> List[str]:
    if not text:
        return []
    s = text.strip()
    candidates: List[str] = []
    for m in re.finditer(r"```(?:json)?\s*([\s\S]*?)```", s, re.IGNORECASE):
        block = m.group(1).strip()
        candidates.extend(_scan_json_objects(block))
    candidates.extend(_scan_json_objects(s))
    seen: Set[str] = set()
    uniq: List[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def refine_query(query: str, draft_answer: Optional[str] = None, user_role: Optional[str] = None) -> RefinedQuery:
    settings = get_settings()
    bypass_conf_threshold = settings.rq_bypass_confidence_threshold
    enable_bypass = getattr(settings, "rq_enable_bypass", True)
    sim_block_threshold = settings.rq_similarity_block_threshold
    post_margin = settings.rq_postvalidate_margin

    top_d_orig, src_orig = _top_distance_and_sources(query, settings.similarity_top_k)
    broadcast_event(
        stage="rq_rag",
        action="start",
        summary="Memulai Query Refinement",
        details={"original_query": query},
    )

    anchor_docs = vector_store_manager.get_documents_by_source(next(iter(src_orig), "")) if src_orig else []
    anchor_context = ""
    if anchor_docs:
        chunks = []
        char_budget = 800
        used = 0
        for _, doc in anchor_docs[:5]:
            txt = (doc.page_content or "").strip()
            if not txt:
                continue
            take = min(len(txt), max(0, char_budget - used))
            if take <= 0:
                break
            chunks.append(txt[:take])
            used += take
        anchor_context = "\n\n".join(chunks)

    sims = _retrieval_sims(query, settings.similarity_top_k)
    if sims:
        s1 = max(sims)
        mean_s = float(np.mean(sims))
        std_s = float(np.std(sims))
        concentration = max(0.0, s1 - mean_s)
        conc_scale = settings.rq_concentration_scale or 0.3
        conc_weight = max(0.0, min(1.0, concentration / conc_scale))
        len_tokens = max(1, len((query or "").split()))
        length_norm = max(1, int(settings.rq_length_norm or 6))
        length_weight = max(0.3, min(1.0, len_tokens / length_norm))
        conf_orig = s1 * conc_weight * length_weight
    else:
        top_d_orig, _ = _top_distance_and_sources(query, settings.similarity_top_k)
        conf_orig = _distance_to_conf(top_d_orig)
    if enable_bypass and conf_orig >= bypass_conf_threshold:
        broadcast_event(
            stage="rq_rag",
            action="bypass",
            summary="Bypass rewrite karena confidence tinggi",
            details={
                "confidence": conf_orig,
                "sims_stats": {
                    "top": float(max(sims) if sims else 0.0),
                    "mean": float(np.mean(sims) if sims else 0.0),
                    "std": float(np.std(sims) if sims else 0.0),
                },
            },
        )
        return {
            "original_query": query,
            "refined_query": query,
            "sub_queries": [query],
            "refinement_type": "BYPASS_HIGH_CONF",
        }

    context_instruction = ""
    if draft_answer:
        context_instruction = f"""
    KONTEKS ITERASI SEBELUMNYA:
    Sistem telah menghasilkan jawaban sementara berikut, namun tingkat
    kepastian model masih rendah (entropy tinggi):

    "{draft_answer}"

    Tugasmu:
    1. Identifikasi bagian jawaban yang KURANG SPESIFIK atau TIDAK DIDUKUNG data.
    2. Rumuskan kueri baru yang menargetkan celah informasi tersebut.
    3. Jika ada klaim tanpa sumber, buat sub_query untuk memverifikasinya.
    """

    prompt = f"""
    Kamu adalah pakar optimasi kueri RAG untuk sistem Kelas Digital Huma Betang.
    Peran pengguna: {user_role or "-"}.
    Gunakan hanya konteks berikut sebagai acuan domain:
    ---
    {anchor_context}
    ---

    Tugasmu adalah melakukan Query Refinement berdasarkan kategori RQ-RAG berikut:

    1. DECOMPOSE
       - Tugasmu adalah memecah pertanyaan kompleks atau multihop menjadi beberapa
         sub-pertanyaan atau tugas yang lebih sederhana dan dapat dijawab secara mandiri.
       - Langkah:
         a) Analisis pertanyaan untuk mengidentifikasi komponen dan informasi yang dibutuhkan.
         b) Susun sub-pertanyaan sehingga setiap sub-pertanyaan bersifat self-contained,
            mengikuti urutan penalaran yang logis, dan bisa dijawab sendiri-sendiri.

    2. DISAMBIGUATE
       - Tugasmu adalah mengidentifikasi dan menghilangkan ambiguitas pada kueri, sehingga
         hanya ada satu interpretasi yang jelas.
       - Langkah:
         a) Cari bagian kueri yang bisa ditafsirkan lebih dari satu cara.
         b) Reformulasikan kueri dengan menambahkan detail, mempersempit istilah, atau
            memberi konteks tambahan agar makna tunggal menjadi jelas.

    3. REWRITE
       - Jika kueri sebenarnya sederhana namun kurang eksplisit, tulis ulang menjadi
         satu kueri yang lebih spesifik dan jelas tanpa mengubah maksud pengguna.

    Hindari topik di luar konteks di atas dan tetap jaga agar makna asli pengguna
    tidak berubah.

    Sebagai keluaran, kamu harus:
    - Mengembalikan SATU kueri utama yang paling optimal pada field "refined_query".
    - Jika tipe "DECOMPOSE", isi "sub_queries" dengan beberapa sub-pertanyaan, satu
      untuk setiap langkah yang saling berurutan dan dapat dijawab sendiri.
    - Jika tipe lain, "sub_queries" boleh berisi satu atau beberapa kueri pendukung
      yang masih konsisten dengan kueri utama.

    {context_instruction}

    Kueri Asli: "{query}"

    Berikan output dalam format JSON:
    {{
      "refinement_type": "REWRITE|DECOMPOSE|DISAMBIGUATE",
      "refined_query": "kueri tunggal yang paling optimal",
      "sub_queries": ["langkah 1", "langkah 2"]
    }}
    """

    client = replicate.Client(api_token=settings.replicate_api_token)

    try:
        output = client.run(
            settings.llm_model,
            input={
                "prompt": prompt,
                "temperature": 0.1,
            },
        )

        text = (
            "".join(str(part) for part in output)
            if isinstance(output, list)
            else str(output)
        )

        candidates = _extract_json_candidates(text)
        parsed = None
        for c in reversed(candidates):
            try:
                obj = json.loads(c)
                if isinstance(obj, dict) and "refined_query" in obj:
                    parsed = obj
                    break
            except Exception:
                continue
        if parsed is None:
            raise ValueError("json_not_found")

        candidate = parsed.get("refined_query", query) or query
        subqs = parsed.get("sub_queries", [candidate]) or [candidate]
        rtype = parsed.get("refinement_type", "REWRITE")
        noop = _norm_text(candidate) == _norm_text(query)
        rtype_out = "NO_CHANGE" if noop else rtype

        sim = 0.0
        try:
            e1 = _embed(query)
            e2 = _embed(candidate)
            sim = _cosine(e1, e2)
        except Exception:
            sim = 0.0

        top_d_cand, src_cand = _top_distance_and_sources(candidate, settings.similarity_top_k)
        jacc = 0.0
        _, src_orig = _top_distance_and_sources(query, settings.similarity_top_k)
        u = len(src_orig | src_cand) or 1
        jacc = len(src_orig & src_cand) / u

        keep = True
        reason = "OK"
        if sim < sim_block_threshold:
            keep = False
            reason = "LOW_SIMILARITY"
        elif top_d_cand > top_d_orig + post_margin or jacc < 0.2:
            keep = False
            reason = "POST_VALIDATE_FAIL"

        broadcast_event(
            stage="rq_rag",
            action="rewrite_validation",
            summary="Validasi rewrite",
            details={
                "similarity": sim,
                "top_d_orig": top_d_orig,
                "top_d_cand": top_d_cand,
                "jaccard": jacc,
                "decision": "keep" if keep else "block",
                "reason": reason,
            },
        )

        if not keep:
            return {
                "original_query": query,
                "refined_query": query,
                "sub_queries": [query],
                "refinement_type": f"BLOCKED_{reason}",
            }

        logger.info(f"RQ-RAG Refinement Success: {rtype}")
        broadcast_event(
            stage="rq_rag",
            action="complete",
            summary=f"Refinement {rtype_out}",
            details={
                "refined_query": candidate,
                "sub_queries_count": len(subqs),
            },
        )
        return {
            "original_query": query,
            "refined_query": candidate,
            "sub_queries": subqs,
            "refinement_type": rtype_out,
        }

    except Exception as exc:
        logger.error(f"failed: {str(exc)}")

    return {
        "original_query": query,
        "refined_query": query,
        "sub_queries": [query],
        "refinement_type": "ERROR_FALLBACK",
    }
