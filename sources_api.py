"""
sources_api.py
--------------
إضافة مصدر من رابط ويب — يعمل **بدون أي مفتاح ذكاء اصطناعي**.

    POST   /api/sources/add-url        جلب الرابط + تلخيصه + توثيق IEEE + حفظه
    GET    /api/sources               قائمة مصادر تقرير
    DELETE /api/sources/{source_id}   حذف مصدر

الخطوات التي تحدث على الخادم (وهذا ما يتجاوز قيود CORS في المتصفح):
  1) جلب الصفحة عبر httpx (موجودة أصلًا في requirements.txt).
  2) استخراج النص النظيف: يستخدم trafilatura إن كانت مثبّتة، وإلا يستخدم
     مستخرجًا مدمجًا يعتمد على وسوم <p> ويحذف السكربتات والقوائم.
  3) تلخيص استخراجي حقيقي (ترجيح تكرار الكلمات) — لا اختراع محتوى.
  4) نقاط رئيسية (اقتراحات للاستخدام في التقرير).
  5) توليد توثيق IEEE من عنوان الصفحة واسم الموقع والسنة وتاريخ الوصول.
  6) حفظ المصدر في جدول sources ليظهر في قائمة المصادر.

للحصول على استخراج أدق (اختياري):
    pip install trafilatura
"""

import html as html_lib
import ipaddress
import re
import socket
from datetime import datetime, timezone
from typing import List, Optional
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.orm import Session

from database import Base, engine, get_db
from writing_assistant import split_sentences, summarize_extractive, term_freq, tokenize

router = APIRouter(prefix="/api/sources", tags=["المصادر"])

MAX_BYTES = 2_000_000      # سقف حجم الصفحة
FETCH_TIMEOUT = 15.0       # ثانية
SUMMARY_SENTENCES = 4
KEY_POINTS = 5

# ترويسة متصفح واقعية: كثير من المواقع (ومنها ويكيبيديا) ترفض الترويسات الآلية بـ 403
_BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36")
# ترويسة بديلة تُجرَّب عند الرفض (سياسة ويكيميديا تقبل معرّفًا وصفيًا فيه وسيلة تواصل)
_ALT_UA = "ScribeAI/1.0 (academic source collector; +https://example.edu/contact)"

_BASE_HEADERS = {
    "User-Agent": _BROWSER_UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ar,en-US;q=0.9,en;q=0.8",
    "Cache-Control": "no-cache",
}

# أسماء مواقع أكاديمية معروفة لتوثيق أنظف
SITE_NAMES = {
    "ieeexplore.ieee.org": "IEEE Xplore",
    "dl.acm.org": "ACM Digital Library",
    "arxiv.org": "arXiv",
    "link.springer.com": "SpringerLink",
    "www.sciencedirect.com": "ScienceDirect",
    "sciencedirect.com": "ScienceDirect",
    "pubmed.ncbi.nlm.nih.gov": "PubMed",
    "www.jstor.org": "JSTOR",
    "jstor.org": "JSTOR",
    "en.wikipedia.org": "Wikipedia",
    "ar.wikipedia.org": "ويكيبيديا",
    "scholar.google.com": "Google Scholar",
}


# ---------------------------------------------------------------------------
# جدول المصادر
# ---------------------------------------------------------------------------
class Source(Base):
    """مصدر بحثي مرتبط بتقرير."""

    __tablename__ = "sources"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(String(64), index=True, nullable=False, default="default")
    ref_number = Column(Integer, default=1)          # رقم التوثيق [n]

    title = Column(String(500), nullable=False)
    summary = Column(Text, nullable=True)
    key_points = Column(Text, nullable=True)          # مفصولة بأسطر
    content = Column(Text, nullable=True)             # النص المستخرج (مقتطع)
    url = Column(String(1000), nullable=True)
    site_name = Column(String(200), nullable=True)
    ieee_citation = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)


def init_sources_db() -> None:
    """ينشئ جدول sources إن لم يكن موجودًا."""
    Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------------
# حماية SSRF: لا نسمح بجلب عناوين داخلية
# ---------------------------------------------------------------------------
def _assert_safe_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(status_code=400, detail="الرابط يجب أن يبدأ بـ http:// أو https://")
    host = parsed.hostname or ""
    if not host:
        raise HTTPException(status_code=400, detail="رابط غير صالح: لا يوجد اسم مضيف.")
    if host.lower() in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        raise HTTPException(status_code=400, detail="لا يُسمح بجلب عناوين محلية.")
    try:
        for info in socket.getaddrinfo(host, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                raise HTTPException(status_code=400, detail="لا يُسمح بجلب عناوين داخلية أو خاصة.")
    except HTTPException:
        raise
    except (socket.gaierror, ValueError):
        raise HTTPException(status_code=400, detail=f"تعذّر تحويل اسم المضيف: {host}")


# ---------------------------------------------------------------------------
# جلب واستخراج النص
# ---------------------------------------------------------------------------
def _fetch_html(url: str) -> str:
    """يجلب الصفحة. عند رفض 403 يعيد المحاولة بترويسة بديلة قبل الاستسلام."""
    last_403: Optional[int] = None
    for ua in (_BROWSER_UA, _ALT_UA):
        headers = dict(_BASE_HEADERS)
        headers["User-Agent"] = ua
        try:
            with httpx.Client(follow_redirects=True, timeout=FETCH_TIMEOUT, headers=headers) as client:
                resp = client.get(url)
                if resp.status_code in (403, 429):
                    last_403 = resp.status_code
                    continue                      # جرّب الترويسة الأخرى
                resp.raise_for_status()
                ctype = resp.headers.get("content-type", "")
                if "html" not in ctype and "xml" not in ctype and "text" not in ctype:
                    raise HTTPException(status_code=415,
                                        detail=f"نوع المحتوى غير مدعوم ({ctype or 'غير معروف'}). الصفحات النصية فقط.")
                data = resp.content[:MAX_BYTES]
                return data.decode(resp.encoding or "utf-8", errors="ignore")
        except HTTPException:
            raise
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="انتهت مهلة جلب الصفحة.")
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=502,
                                detail=f"رفض الموقع الطلب (رمز {exc.response.status_code}).")
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"تعذّر جلب الرابط: {exc}")

    raise HTTPException(
        status_code=502,
        detail=(f"رفض الموقع الطلب (رمز {last_403}). بعض المواقع تحجب الجلب الآلي. "
                "جرّب رابطًا من موقع آخر (مقال إخباري، مدونة، arXiv، مستودع جامعي)، "
                "أو الصق نص الصفحة يدويًا كمصدر."))


# ---------------------------------------------------------------------------
# ويكيبيديا: واجهة رسمية بدل الجلب (لا تُحجب، ونصّها نظيف بلا قوائم)
# ---------------------------------------------------------------------------
def _is_wikipedia(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host.endswith("wikipedia.org")


def _wikipedia_extract(url: str) -> Optional[tuple]:
    """
    يستخدم واجهة MediaWiki الرسمية:
        /w/api.php?action=query&prop=extracts&explaintext=1
    ويعيد (العنوان, النص) أو None عند الفشل ليُجرَّب الجلب العادي.
    """
    parsed = urlparse(url)
    host = parsed.hostname or ""
    slug = [s for s in parsed.path.split("/") if s]
    if not slug:
        return None
    title = slug[-1]                     # .../wiki/Augmented_reality
    api = f"{parsed.scheme}://{host}/w/api.php"
    params = {
        "action": "query", "prop": "extracts", "explaintext": "1",
        "exsectionformat": "wiki", "redirects": "1", "format": "json",
        "titles": title.replace("_", " "),
    }
    headers = dict(_BASE_HEADERS)
    headers["Accept"] = "application/json"
    try:
        with httpx.Client(follow_redirects=True, timeout=FETCH_TIMEOUT, headers=headers) as client:
            resp = client.get(api, params=params)
            resp.raise_for_status()
            pages = (resp.json().get("query") or {}).get("pages") or {}
            for _, page in pages.items():
                text = (page.get("extract") or "").strip()
                if text and len(text) > 200:
                    return (page.get("title") or title.replace("_", " "), text)
    except Exception:
        return None
    return None


_TAG_DROP = re.compile(
    r"<(script|style|noscript|nav|header|footer|aside|form|svg)\b[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL)
_TAG_ANY = re.compile(r"<[^>]+>")


def _meta(html: str, *names: str) -> Optional[str]:
    for name in names:
        m = re.search(
            rf'<meta[^>]+(?:property|name)=["\']{re.escape(name)}["\'][^>]+content=["\']([^"\']+)["\']',
            html, re.IGNORECASE)
        if not m:
            m = re.search(
                rf'<meta[^>]+content=["\']([^"\']+)["\'][^>]+(?:property|name)=["\']{re.escape(name)}["\']',
                html, re.IGNORECASE)
        if m:
            return html_lib.unescape(m.group(1)).strip()
    return None


def _extract_title(html: str, url: str) -> str:
    title = _meta(html, "og:title", "twitter:title", "citation_title", "dc.title")
    if not title:
        m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if m:
            title = html_lib.unescape(_TAG_ANY.sub("", m.group(1))).strip()
    if not title:
        m = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.IGNORECASE | re.DOTALL)
        if m:
            title = html_lib.unescape(_TAG_ANY.sub("", m.group(1))).strip()
    if not title:
        seg = [s for s in urlparse(url).path.split("/") if s]
        title = (seg[-1].replace("-", " ").replace("_", " ").title() if seg
                 else urlparse(url).hostname or "Untitled page")
    return re.sub(r"\s+", " ", title)[:480]


def _extract_text_builtin(html: str) -> str:
    """مستخرج مدمج: يعتمد على وسوم <p> ثم يرجع لكل النص عند الحاجة."""
    cleaned = _TAG_DROP.sub(" ", html)
    paras = re.findall(r"<p\b[^>]*>(.*?)</p>", cleaned, re.IGNORECASE | re.DOTALL)
    chunks = []
    for p in paras:
        txt = html_lib.unescape(_TAG_ANY.sub(" ", p))
        txt = re.sub(r"\s+", " ", txt).strip()
        if len(txt) >= 60:                 # تجاهل الفقرات القصيرة (قوائم/أزرار)
            chunks.append(txt)
    if not chunks:                          # صفحة بلا <p> واضحة
        txt = html_lib.unescape(_TAG_ANY.sub(" ", cleaned))
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt
    return "\n\n".join(chunks)


def _extract_text(html: str, url: str) -> str:
    """يفضّل trafilatura إن توفّرت، وإلا المستخرج المدمج."""
    try:
        import trafilatura  # اختياري
        extracted = trafilatura.extract(html, url=url, include_comments=False,
                                        include_tables=False)
        if extracted and len(extracted.strip()) > 200:
            return extracted.strip()
    except ImportError:
        pass
    except Exception:
        pass
    return _extract_text_builtin(html)


def _site_name(html: str, url: str) -> str:
    site = _meta(html, "og:site_name", "citation_journal_title")
    if site:
        return site[:180]
    host = (urlparse(url).hostname or "").lower()
    return SITE_NAMES.get(host, host.replace("www.", "") or "Web")


def _year(html: str) -> str:
    raw = _meta(html, "article:published_time", "citation_publication_date",
                "citation_date", "dc.date", "datePublished")
    if raw:
        m = re.search(r"(19|20)\d{2}", raw)
        if m:
            return m.group(0)
    m = re.search(r'"datePublished"\s*:\s*"((?:19|20)\d{2})', html)
    if m:
        return m.group(1)
    return str(datetime.now(timezone.utc).year)


def _key_points(text: str, limit: int = KEY_POINTS) -> List[str]:
    """أهم الجُمل (ترجيح تكرار الكلمات) بترتيب ظهورها."""
    sents = [s for s in split_sentences(text) if len(s.split()) >= 6]
    if not sents:
        return []
    freqs = term_freq(tokenize(text))
    scored = []
    for i, s in enumerate(sents):
        toks = tokenize(s)
        if toks:
            scored.append((sum(freqs.get(t, 0) for t in toks) / len(toks), i, s))
    top = sorted(scored, key=lambda x: x[0], reverse=True)[:limit]
    return [s for _, _, s in sorted(top, key=lambda x: x[1])]


def _ieee(ref_number: int, site: str, title: str, year: str, url: str) -> str:
    accessed = datetime.now(timezone.utc).strftime("%d %b. %Y")
    return (f'[{ref_number}] {site}, "{title}," {year}. [Online]. '
            f'Available: {url}. [Accessed: {accessed}].')


# ---------------------------------------------------------------------------
# النماذج
# ---------------------------------------------------------------------------
class AddUrlRequest(BaseModel):
    url: str = Field(..., description="رابط الصفحة المراد إضافتها كمصدر")
    report_id: str = Field(default="default", description="معرّف التقرير")
    save: bool = Field(default=True, description="حفظ المصدر في قاعدة البيانات")


class SourceOut(BaseModel):
    id: Optional[int] = None
    report_id: str = "default"
    n: int = 1
    title: str
    summary: str = ""
    key_points: List[str] = Field(default_factory=list)
    content: str = ""
    url: str = ""
    site_name: str = ""
    ieee_citation: str = ""
    word_count: int = 0
    extractor: str = "builtin"
    created_at: Optional[datetime] = None


def _source_to_out(r: "Source") -> SourceOut:
    """تحويل صف قاعدة البيانات إلى نموذج الاستجابة (يستخدمه reports_api أيضًا)."""
    return SourceOut(
        id=r.id, report_id=r.report_id, n=r.ref_number or 1, title=r.title,
        summary=r.summary or "", key_points=(r.key_points or "").split("\n") if r.key_points else [],
        content=r.content or "", url=r.url or "", site_name=r.site_name or "",
        ieee_citation=r.ieee_citation or "",
        word_count=len((r.content or "").split()), created_at=r.created_at,
    )


def _next_ref(db: Session, report_id: str) -> int:
    rows = db.query(Source).filter(Source.report_id == report_id).all()
    return max([r.ref_number or 0 for r in rows] or [0]) + 1


# ---------------------------------------------------------------------------
# نقاط النهاية
# ---------------------------------------------------------------------------
@router.post("/add-url", response_model=SourceOut)
def add_url(payload: AddUrlRequest, db: Session = Depends(get_db)) -> SourceOut:
    """
    يجلب الرابط على الخادم (تجاوزًا لقيود CORS)، يستخرج نصه، يلخّصه،
    يستخرج نقاطًا رئيسية، يبني توثيق IEEE، ويحفظه كمصدر للتقرير.
    """
    url = payload.url.strip()
    _assert_safe_url(url)

    html = ""
    extractor = "builtin"
    title = ""
    text = ""

    # ويكيبيديا: الواجهة الرسمية أولًا (تتجاوز حجب 403 وتعطي نصًا نظيفًا)
    if _is_wikipedia(url):
        wiki = _wikipedia_extract(url)
        if wiki:
            title, text = wiki
            extractor = "wikipedia-api"

    if not text:
        html = _fetch_html(url)
        title = _extract_title(html, url)
        text = _extract_text(html, url)
        try:
            import trafilatura  # noqa: F401
            extractor = "trafilatura"
        except ImportError:
            extractor = "builtin"

    if not text or len(text) < 120:
        raise HTTPException(
            status_code=422,
            detail="تعذّر استخراج نص كافٍ من الصفحة. قد تكون الصفحة تعتمد على "
                   "JavaScript بالكامل أو محمية. جرّب رابطًا آخر أو الصق النص يدويًا.")

    summary = summarize_extractive(text, SUMMARY_SENTENCES)
    points = _key_points(text)
    site = _site_name(html, url)
    year = _year(html)

    # رقم التوثيق التالي داخل هذا التقرير
    existing = db.query(Source).filter(Source.report_id == payload.report_id).count()
    ref_number = existing + 1
    citation = _ieee(ref_number, site, title, year, url)

    row_id = None
    created = None
    if payload.save:
        try:
            row = Source(
                report_id=payload.report_id, ref_number=ref_number, title=title,
                summary=summary, key_points="\n".join(points), content=text[:20000],
                url=url, site_name=site, ieee_citation=citation,
            )
            db.add(row); db.commit(); db.refresh(row)
            row_id, created = row.id, row.created_at
        except Exception:
            db.rollback()   # الحفظ ليس حرجًا؛ نعيد النتيجة على أي حال

    return SourceOut(
        id=row_id, report_id=payload.report_id, n=ref_number, title=title,
        summary=summary, key_points=points, content=text[:20000], url=url,
        site_name=site, ieee_citation=citation,
        word_count=len(text.split()), extractor=extractor, created_at=created,
    )


@router.get("", response_model=List[SourceOut])
def list_sources(
    report_id: str = Query(default="default"),
    db: Session = Depends(get_db),
) -> List[SourceOut]:
    """يعيد مصادر تقرير معيّن بترتيب أرقام التوثيق."""
    rows = (db.query(Source)
              .filter(Source.report_id == report_id)
              .order_by(Source.ref_number.asc())
              .all())
    return [_source_to_out(r) for r in rows]


class SourceIn(BaseModel):
    """إنشاء مصدر يدويًا (كتاب/بحث ورقي) أو من ملف مرفوع."""
    report_id: str = "default"
    title: str = Field(..., min_length=1)
    summary: str = ""
    content: str = ""
    url: str = ""
    ieee_citation: str = ""
    ref_number: Optional[int] = None


class SourcePatch(BaseModel):
    title: Optional[str] = None
    summary: Optional[str] = None
    content: Optional[str] = None
    url: Optional[str] = None
    ieee_citation: Optional[str] = None
    ref_number: Optional[int] = None


@router.post("", response_model=SourceOut)
def create_source(payload: SourceIn, db: Session = Depends(get_db)) -> SourceOut:
    """يضيف مصدرًا بلا جلب من الويب (إدخال يدوي أو ملف)."""
    ref = payload.ref_number or _next_ref(db, payload.report_id)
    site = ""
    if payload.url:
        host = (urlparse(payload.url).hostname or "").lower()
        site = SITE_NAMES.get(host, host.replace("www.", ""))
    citation = payload.ieee_citation or f'[{ref}] {payload.title}, {datetime.now(timezone.utc).year}.'
    row = Source(
        report_id=payload.report_id, ref_number=ref, title=payload.title.strip(),
        summary=payload.summary, key_points="", content=(payload.content or "")[:20000],
        url=payload.url, site_name=site,
        ieee_citation=re.sub(r"^\s*\[\d+\]\s*", f"[{ref}] ", citation),
    )
    db.add(row); db.commit(); db.refresh(row)
    return _source_to_out(row)


@router.patch("/{source_id}", response_model=SourceOut)
def update_source(source_id: int, payload: SourcePatch, db: Session = Depends(get_db)) -> SourceOut:
    """
    يعدّل مصدرًا. يسمح بتغيير رقم التوثيق [n] يدويًا، بما في ذلك رقم مستخدم
    من مصدر آخر — الواجهة تُظهر تنبيهًا ولا تمنع الحفظ. الشرط الوحيد أن
    يكون الرقم 1 أو أكثر.
    """
    row = db.query(Source).filter(Source.id == source_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"لا يوجد مصدر بالرقم {source_id}")

    if payload.ref_number is not None and payload.ref_number != row.ref_number:
        if payload.ref_number < 1:
            raise HTTPException(status_code=400, detail="رقم التوثيق يجب أن يكون 1 أو أكثر.")
        # التكرار مسموح بقرار المستخدم: الواجهة تُنبّه فقط ولا تمنع الحفظ،
        # لأن الباحث قد يحتاج ترقيمًا مؤقتًا أثناء إعادة تنظيم مصادره.
        row.ref_number = payload.ref_number

    if payload.title is not None:
        row.title = payload.title.strip()
    if payload.summary is not None:
        row.summary = payload.summary
    if payload.content is not None:
        row.content = payload.content[:20000]
    if payload.url is not None:
        row.url = payload.url
    if payload.ieee_citation is not None:
        row.ieee_citation = payload.ieee_citation

    # اجعل رقم التوثيق داخل نص IEEE مطابقًا للرقم الفعلي
    if row.ieee_citation:
        row.ieee_citation = re.sub(r"^\s*\[\d+\]\s*", f"[{row.ref_number}] ", row.ieee_citation)

    db.commit(); db.refresh(row)
    return _source_to_out(row)


@router.delete("/{source_id}")
def delete_source(source_id: int, db: Session = Depends(get_db)) -> dict:
    """يحذف مصدرًا واحدًا."""
    row = db.query(Source).filter(Source.id == source_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"لا يوجد مصدر بالرقم {source_id}")
    db.delete(row); db.commit()
    return {"success": True, "deleted": source_id}
