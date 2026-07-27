"""
workspace_api.py
----------------
نقاط النهاية الخاصة بصفحة المسودة / مساحة العمل. هذا الملف هو "واجهة"
الوكيل القائد (Lead Synthesizer) نحو الواجهة الأمامية:

    POST   /api/workspace/enhance                  تحسين مسودة عبر الوكلاء الثلاثة
    GET    /api/workspace/drafts/{draft_id}/versions   سجل الإصدارات
    DELETE /api/workspace/versions/{version_id}    حذف إصدار
    GET    /api/workspace/wordbank/status          حالة بنك الكلمات

كل عملية تحسين تُحفظ تلقائيًا في جدول draft_versions ليظهر في
"سجل الإصدارات" (VERSION HISTORY) في الواجهة.

التفعيل في app/main.py:
    from app.workspace_api import router as workspace_router
    app.include_router(workspace_router)
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.orm import Session

from database import Base, engine, get_db
from workspace_agents import enhance_draft, load_word_bank
from writing_assistant import answer_writing_query, enhance_academic

router = APIRouter(prefix="/api/workspace", tags=["مساحة العمل"])


# ---------------------------------------------------------------------------
# جدول سجل الإصدارات
# ---------------------------------------------------------------------------
class DraftVersion(Base):
    """إصدار واحد من مسودة بعد تمريرها على الوكلاء."""

    __tablename__ = "draft_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    draft_id = Column(String(64), index=True, nullable=False, default="default")
    customer_id = Column(String(64), index=True, nullable=True)

    original_text = Column(Text, nullable=False)
    refined_text = Column(Text, nullable=False)
    changes_json = Column(Text, nullable=True)   # JSON string
    language = Column(String(8), default="ar")
    action = Column(String(32), nullable=True)   # الإجراء المقترح
    source = Column(String(32), default="rule_based")

    created_at = Column(DateTime, default=datetime.utcnow)


class Draft(Base):
    """المسودة الحالية لمساحة عمل واحدة (حفظ تلقائي أثناء الكتابة)."""

    __tablename__ = "drafts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    draft_id = Column(String(64), unique=True, index=True, nullable=False)
    text = Column(Text, nullable=False, default="")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def init_workspace_db() -> None:
    """ينشئ جدول draft_versions إن لم يكن موجودًا (يُستدعى عند الإقلاع)."""
    Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------------
# النماذج
# ---------------------------------------------------------------------------
class SourceIn(BaseModel):
    """مصدر من مصادر التقرير (تُرسلها الواجهة الأمامية)."""
    n: int = 1
    title: str = ""
    summary: str = ""
    content: str = ""
    url: str = ""
    ieee_citation: str = ""


class DraftEnhanceRequest(BaseModel):
    text: str = Field(..., min_length=1, description="نص المسودة الحالي")
    mode: str = Field(default="academic", description="academic (كتابة أكاديمية) أو support (رد دعم)")
    sources: List[SourceIn] = Field(default_factory=list, description="مصادر التقرير للتوثيق")
    language: Optional[str] = Field(default=None, description="ar أو en (يُكتشف تلقائيًا)")
    customer_id: Optional[str] = Field(default=None, description="معرّف العميل إن وُجد")
    draft_id: str = Field(default="default", description="معرّف المسودة لسجل الإصدارات")
    save_version: bool = Field(default=True, description="حفظ النتيجة في سجل الإصدارات")


class ContextItem(BaseModel):
    n: int
    title: str
    source: str = ""
    score: float = 0.0


class Suggestion(BaseModel):
    title: str = ""
    text: str = ""
    action: Optional[str] = None
    n: Optional[int] = None


class AgentReport(BaseModel):
    agent: str
    rationale: str = ""
    changes: List[dict] = Field(default_factory=list)
    articles: List[str] = Field(default_factory=list)


class DraftEnhanceResponse(BaseModel):
    refined_draft: str
    enhanced_text: str                     # توافق مع الواجهة الحالية
    changes: List[dict] = Field(default_factory=list)
    suggestions: List[Suggestion] = Field(default_factory=list)
    context: List[ContextItem] = Field(default_factory=list)
    cited: List[int] = Field(default_factory=list)
    decision: dict = Field(default_factory=dict)
    agents: List[AgentReport] = Field(default_factory=list)
    rationale: str = ""
    language: str = "ar"
    source: str = "rule_based"
    version_id: Optional[int] = None


class VersionOut(BaseModel):
    id: int
    draft_id: str
    original_text: str
    refined_text: str
    changes: List[dict] = Field(default_factory=list)
    language: str
    action: Optional[str] = None
    source: str
    created_at: datetime


# ---------------------------------------------------------------------------
# نقاط النهاية
# ---------------------------------------------------------------------------
@router.post("/enhance", response_model=DraftEnhanceResponse)
def enhance(payload: DraftEnhanceRequest, db: Session = Depends(get_db)) -> DraftEnhanceResponse:
    """
    يمرّر المسودة على الوكلاء الثلاثة (بحث -> لغة ونبرة -> دمج) ويعيد:
    المسودة المنقّحة، سجل التغييرات، الردود المقترحة، مقالات المعرفة
    المستخدمة، وتعليل كل وكيل. ويحفظ الإصدار في سجل الإصدارات.
    """
    import json

    try:
        if (payload.mode or "academic").lower() == "support":
            # مسار وكلاء الدعم (المشروع الأصلي): بحث + نبرة + دمج
            result = enhance_draft(
                draft=payload.text,
                language=payload.language,
                customer_id=payload.customer_id,
            )
        else:
            # المسار الافتراضي: تحرير أكاديمي بالاعتماد على مصادر التقرير
            result = enhance_academic(
                draft=payload.text,
                sources=[s.model_dump() for s in payload.sources],
                language=payload.language,
            )
    except FileNotFoundError as exc:
        # مثال: ملف قاعدة المعرفة غير موجود
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # حماية عامة حتى لا ينهار الخادم
        raise HTTPException(status_code=500, detail=f"تعذّر تحسين المسودة: {exc}") from exc

    version_id: Optional[int] = None
    if payload.save_version:
        try:
            row = DraftVersion(
                draft_id=payload.draft_id,
                customer_id=payload.customer_id,
                original_text=payload.text,
                refined_text=result["refined_draft"],
                changes_json=json.dumps(result["changes"], ensure_ascii=False),
                language=result["language"],
                action=(result.get("decision") or {}).get("action"),
                source=result["source"],
            )
            db.add(row)
            db.commit()
            db.refresh(row)
            version_id = row.id
        except Exception:
            db.rollback()  # الحفظ ليس حرجًا؛ لا نُفشل الطلب بسببه

    return DraftEnhanceResponse(version_id=version_id, **result)


@router.get("/drafts/{draft_id}/versions", response_model=List[VersionOut])
def list_versions(
    draft_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
) -> List[VersionOut]:
    """يعيد سجل إصدارات مسودة معيّنة، من الأحدث إلى الأقدم (مع سجل التغييرات)."""
    import json

    rows = (
        db.query(DraftVersion)
        .filter(DraftVersion.draft_id == draft_id)
        .order_by(DraftVersion.created_at.desc())
        .limit(limit)
        .all()
    )

    out: List[VersionOut] = []
    for r in rows:
        try:
            changes = json.loads(r.changes_json) if r.changes_json else []
            if not isinstance(changes, list):
                changes = []
        except (ValueError, TypeError):
            changes = []
        out.append(VersionOut(
            id=r.id, draft_id=r.draft_id, original_text=r.original_text,
            refined_text=r.refined_text, changes=changes, language=r.language or "ar",
            action=r.action, source=r.source or "rule_based",
            created_at=r.created_at or datetime.utcnow(),
        ))
    return out


@router.delete("/versions/{version_id}")
def delete_version(version_id: int, db: Session = Depends(get_db)) -> dict:
    """يحذف إصدارًا واحدًا من سجل الإصدارات."""
    row = db.query(DraftVersion).filter(DraftVersion.id == version_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"لا يوجد إصدار بالرقم {version_id}")
    db.delete(row)
    db.commit()
    return {"success": True, "deleted": version_id}


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="رسالة المستخدم")
    sources: List[SourceIn] = Field(default_factory=list, description="مصادر التقرير")
    draft: str = Field(default="", description="نص المسودة الحالي (لطلب المراجعة)")
    active_source_n: Optional[int] = Field(default=None, description="رقم المصدر المفتوح")
    language: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    insert: str = ""
    citations: List[int] = Field(default_factory=list)
    intent: str = ""
    source: str = "rule_based"


@router.post("/chat", response_model=ChatResponse)
def writing_chat(payload: ChatRequest) -> ChatResponse:
    """
    مساعد الكتابة الأكاديمية: يحلّل مصادر التقرير ويجيب عن أوامر مثل
    «لخّص»، «نقاط رئيسية»، «ماذا تقول المصادر عن ...»، «وثّق: ...»،
    «هيكل التقرير»، «راجع مسودتي». يعمل بدون مفتاح ذكاء اصطناعي.
    """
    try:
        result = answer_writing_query(
            message=payload.message,
            sources=[s.model_dump() for s in payload.sources],
            draft=payload.draft,
            active_source_n=payload.active_source_n,
            language=payload.language,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"تعذّر تنفيذ الطلب: {exc}") from exc
    return ChatResponse(**result)


class DraftContent(BaseModel):
    text: str = Field(default="", description="نص المسودة الحالي")


class DraftContentOut(BaseModel):
    draft_id: str
    text: str = ""
    updated_at: Optional[datetime] = None


@router.get("/drafts/{draft_id}/content", response_model=DraftContentOut)
def get_draft(draft_id: str, db: Session = Depends(get_db)) -> DraftContentOut:
    """يعيد المسودة المحفوظة تلقائيًا (نص فارغ إن لم تُحفظ بعد)."""
    row = db.query(Draft).filter(Draft.draft_id == draft_id).first()
    if row is None:
        return DraftContentOut(draft_id=draft_id, text="", updated_at=None)
    return DraftContentOut(draft_id=row.draft_id, text=row.text or "", updated_at=row.updated_at)


@router.put("/drafts/{draft_id}/content", response_model=DraftContentOut)
def put_draft(draft_id: str, payload: DraftContent, db: Session = Depends(get_db)) -> DraftContentOut:
    """
    حفظ تلقائي للمسودة (upsert). تُنادى من الواجهة بعد توقّف الكتابة،
    حتى لا يفقد المستخدم عمله عند التنقّل أو إعادة التحميل.
    """
    row = db.query(Draft).filter(Draft.draft_id == draft_id).first()
    if row is None:
        row = Draft(draft_id=draft_id, text=payload.text or "")
        db.add(row)
    else:
        row.text = payload.text or ""
        row.updated_at = datetime.utcnow()
    db.commit(); db.refresh(row)
    return DraftContentOut(draft_id=row.draft_id, text=row.text or "", updated_at=row.updated_at)


@router.get("/wordbank/status")
def wordbank_status() -> dict:
    """فحص سريع للتأكد من تحميل بنك الكلمات (words_alpha.txt)."""
    bank = load_word_bank()
    return {"loaded": bool(bank), "word_count": len(bank)}
