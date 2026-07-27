"""
reports_api.py
--------------
حفظ التقارير بشكل دائم:

    GET    /api/reports              كل التقارير ومعها مصادرها
    POST   /api/reports              إنشاء تقرير
    PATCH  /api/reports/{report_id}  تعديل (العنوان / اللون / تاريخ التسليم)
    DELETE /api/reports/{report_id}  حذف التقرير وكل مصادره
    POST   /api/reports/seed         إنشاء التقارير الافتراضية مرة واحدة فقط

ملاحظة مهمة عن `seed`: المصادر التي أضافها المستخدم سابقًا مرتبطة بمعرّفات
التقارير القديمة (r1, r2, r3). لذلك تُنشئ دالة seed التقارير **بالمعرّفات
نفسها** حتى تعود تلك المصادر للظهور بدل أن تبقى معلّقة بلا تقرير.

التفعيل في app/main.py:
    from app.reports_api import router as reports_router, init_reports_db
    app.include_router(reports_router)
    ...
    init_reports_db()
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.orm import Session

from database import Base, engine, get_db
from sources_api import Source, SourceOut, _source_to_out

router = APIRouter(prefix="/api/reports", tags=["التقارير"])

# التقارير الافتراضية — بالمعرّفات القديمة نفسها لإعادة ربط المصادر المحفوظة
SEED_REPORTS = [
    {"id": "r1", "title": "Autonomous Agents in Cybersecurity",
     "title_ar": "الوكلاء المستقلون في الأمن السيبراني", "accent": "#1746c4"},
    {"id": "r2", "title": "Gamification in Higher Education",
     "title_ar": "التلعيب في التعليم العالي", "accent": "#b3852f"},
    {"id": "r3", "title": "LLM Reasoning Systems",
     "title_ar": "أنظمة الاستدلال في النماذج اللغوية", "accent": "#0b2148"},
]


# ---------------------------------------------------------------------------
# الجدول
# ---------------------------------------------------------------------------
class Report(Base):
    """تقرير أكاديمي واحد."""

    __tablename__ = "reports"

    id = Column(String(64), primary_key=True)          # معرّف نصّي مثل r1
    title = Column(String(500), nullable=False)
    title_ar = Column(String(500), nullable=True)
    accent = Column(String(16), default="#1746c4")
    due_date = Column(String(20), nullable=True)        # YYYY-MM-DD
    position = Column(Integer, default=0)               # ترتيب العرض
    created_at = Column(DateTime, default=datetime.utcnow)


def init_reports_db() -> None:
    """ينشئ جدول reports إن لم يكن موجودًا."""
    Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------------
# النماذج
# ---------------------------------------------------------------------------
class ReportIn(BaseModel):
    id: Optional[str] = Field(default=None, description="اتركه فارغًا ليُولَّد تلقائيًا")
    title: str = Field(..., min_length=1)
    title_ar: Optional[str] = None
    accent: str = "#1746c4"
    due_date: Optional[str] = None


class ReportPatch(BaseModel):
    title: Optional[str] = None
    title_ar: Optional[str] = None
    accent: Optional[str] = None
    due_date: Optional[str] = None


class ReportOut(BaseModel):
    id: str
    title: str
    title_ar: Optional[str] = None
    accent: str = "#1746c4"
    due_date: Optional[str] = None
    created_at: Optional[datetime] = None
    sources: List[SourceOut] = Field(default_factory=list)


def _report_to_out(row: Report, sources: List[Source]) -> ReportOut:
    return ReportOut(
        id=row.id, title=row.title, title_ar=row.title_ar,
        accent=row.accent or "#1746c4", due_date=row.due_date,
        created_at=row.created_at,
        sources=[_source_to_out(s) for s in sources],
    )


# ---------------------------------------------------------------------------
# نقاط النهاية
# ---------------------------------------------------------------------------
@router.get("", response_model=List[ReportOut])
def list_reports(db: Session = Depends(get_db)) -> List[ReportOut]:
    """كل التقارير ومعها مصادرها (طلب واحد يكفي الواجهة)."""
    reports = db.query(Report).order_by(Report.position.asc(), Report.created_at.asc()).all()
    if not reports:
        return []
    by_report = {}
    for s in (db.query(Source)
                .filter(Source.report_id.in_([r.id for r in reports]))
                .order_by(Source.ref_number.asc())
                .all()):
        by_report.setdefault(s.report_id, []).append(s)
    return [_report_to_out(r, by_report.get(r.id, [])) for r in reports]


@router.post("", response_model=ReportOut)
def create_report(payload: ReportIn, db: Session = Depends(get_db)) -> ReportOut:
    rid = (payload.id or f"r{int(datetime.utcnow().timestamp() * 1000)}").strip()
    if db.query(Report).filter(Report.id == rid).first():
        raise HTTPException(status_code=409, detail=f"يوجد تقرير بالمعرّف {rid}")
    top = db.query(Report).order_by(Report.position.desc()).first()
    row = Report(
        id=rid, title=payload.title.strip(),
        title_ar=(payload.title_ar or payload.title).strip(),
        accent=payload.accent or "#1746c4", due_date=payload.due_date or None,
        position=((top.position if top else 0) - 1),      # الأحدث أولًا
    )
    db.add(row); db.commit(); db.refresh(row)
    return _report_to_out(row, [])


@router.patch("/{report_id}", response_model=ReportOut)
def update_report(report_id: str, payload: ReportPatch, db: Session = Depends(get_db)) -> ReportOut:
    row = db.query(Report).filter(Report.id == report_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"لا يوجد تقرير بالمعرّف {report_id}")
    if payload.title is not None:
        row.title = payload.title.strip()
        row.title_ar = (payload.title_ar or payload.title).strip()
    elif payload.title_ar is not None:
        row.title_ar = payload.title_ar.strip()
    if payload.accent is not None:
        row.accent = payload.accent
    if payload.due_date is not None:
        row.due_date = payload.due_date or None
    db.commit(); db.refresh(row)
    sources = (db.query(Source).filter(Source.report_id == report_id)
                 .order_by(Source.ref_number.asc()).all())
    return _report_to_out(row, sources)


@router.delete("/{report_id}")
def delete_report(report_id: str, db: Session = Depends(get_db)) -> dict:
    """يحذف التقرير وكل مصادره."""
    row = db.query(Report).filter(Report.id == report_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail=f"لا يوجد تقرير بالمعرّف {report_id}")
    removed = db.query(Source).filter(Source.report_id == report_id).delete()
    db.delete(row); db.commit()
    return {"success": True, "deleted": report_id, "sources_deleted": removed}


@router.post("/reset-demo")
def reset_demo(db: Session = Depends(get_db)) -> dict:
    """
    تنظيف لمرة واحدة: يحذف التقارير الوهمية (r1/r2/r3) وكل مصادرها من
    قاعدة البيانات المنشورة. نادِها مرة واحدة بعد النشر ثم يمكن تجاهلها.
    لا يمسّ أي تقرير أنشأه مستخدم حقيقي (معرّفاته مختلفة).
    """
    demo_ids = [r["id"] for r in SEED_REPORTS]
    sources_removed = (db.query(Source)
                         .filter(Source.report_id.in_(demo_ids))
                         .delete(synchronize_session=False))
    reports_removed = (db.query(Report)
                         .filter(Report.id.in_(demo_ids))
                         .delete(synchronize_session=False))
    db.commit()
    return {"success": True, "reports_deleted": reports_removed,
            "sources_deleted": sources_removed, "ids": demo_ids}


@router.post("/seed", response_model=List[ReportOut])
def seed_reports(db: Session = Depends(get_db)) -> List[ReportOut]:
    """
    لا يُنشئ أي بيانات وهمية. يبدأ كل زائر بمنصة فارغة ويضيف تقاريره بنفسه.
    (أُبقيت نقطة النهاية لتوافق الواجهة؛ تعيد ببساطة ما هو موجود فعلًا.)
    """
    return list_reports(db)
