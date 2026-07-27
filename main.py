"""
main.py
-------
نقطة انطلاق تطبيق FastAPI. يجمع هذا الملف كل شيء معًا:
قاعدة البيانات، النماذج، الوكيل الذكي، والأدوات، ويعرضها كـ API
موثّق تلقائيًا عبر Swagger على المسار /docs.
"""

from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from database import init_db, get_db, Conversation, Ticket
from models import (
    ChatRequest,
    ChatResponse,
    TicketCreate,
    TicketResponse,
    TicketStatusUpdate,
)
import agent as agent_module
import tools as tools_module
from workspace_api import router as workspace_router, init_workspace_db
from sources_api import router as sources_router, init_sources_db
from reports_api import router as reports_router, init_reports_db
from workspace_api import router as workspace_router, init_workspace_db

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# استيراد الـ Routers والـ Database Inits
# (تأكدي من صحة اسم الملفات والمجلدات لديكِ)

from routers.sources import sources_router
from routers.reports import reports_router
from database import (
    init_db,
    init_workspace_db,
    init_sources_db,
    init_reports_db,
)

# 1. إنشاء التطبيق
app = FastAPI(
    title="وكيل خدمة العملاء الذكي | Customer Service AI Agent",
    description=(
        "مشروع تعليمي يوضح كيف يبني وكيل ذكاء اصطناعي حقيقي يتخذ قرارات "
        "وينفذ إجراءات فعلية (أدوات/Tools) بدل الاكتفاء بالرد كمحادثة عادية."
    ),
    version="1.0.0",
)

# 2. إعدادات الـ CORS (مرة واحدة تكفي)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. تضمين الـ Routers (بدون تكرار)
app.include_router(workspace_router)
app.include_router(sources_router)
app.include_router(reports_router)


# 4. الأحداث عند بدء التشغيل
@app.on_event("startup")
def on_startup() -> None:
    """يُنشئ جداول قاعدة البيانات تلقائيًا عند أول تشغيل للتطبيق."""
    init_db()
    init_workspace_db()   # جدول draft_versions
    init_sources_db()     # جدول sources
    init_reports_db()     # جدول reports


# 5. الـ Endpoints الأساسية
@app.get("/", tags=["عام"])
def root() -> dict:
    return {
        "message": "وكيل خدمة العملاء الذكي يعمل الآن.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["عام"])
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, tags=["المحادثة"])
def chat(payload: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    """
    نقطة الدخول الرئيسية للمحادثة مع الوكيل الذكي.

    يستقبل رسالة العميل، يمررها للوكيل (LangGraph) الذي يقرر الإجراء
    المناسب وينفذه فعليًا، ثم يعيد ردًا منظمًا يوضح: الإجراء المتخذ،
    نص الرد، مصدر الإجابة إن وُجد، رقم التذكرة إن وُجدت، ودرجة الثقة.
    """
    try:
        result = agent_module.run_agent(
            db=db,
            customer_id=payload.customer_id,
            message=payload.message,
            language=payload.language,
        )
        return ChatResponse(**result)
    except FileNotFoundError as exc:
        # مثال: ملف قاعدة المعرفة غير موجود
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # حماية عامة حتى لا ينهار الخادم بالكامل
        raise HTTPException(status_code=500, detail=f"خطأ غير متوقع في معالجة الرسالة: {exc}") from exc


@app.post("/tickets", response_model=TicketResponse, tags=["التذاكر"])
def create_ticket_endpoint(payload: TicketCreate, db: Session = Depends(get_db)) -> TicketResponse:
    """ينشئ تذكرة دعم فني يدويًا (بدون المرور عبر الوكيل الذكي)."""
    result = tools_module.create_ticket_tool(
        db=db,
        customer_id=payload.customer_id,
        subject=payload.subject,
        description=payload.description,
        category=payload.category,
        priority=payload.priority,
    )
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "فشل إنشاء التذكرة"))

    ticket = db.query(Ticket).filter(Ticket.id == result["ticket_id"]).first()
    return TicketResponse.model_validate(ticket)


@app.get("/tickets/{ticket_id}", response_model=TicketResponse, tags=["التذاكر"])
def get_ticket_endpoint(ticket_id: int, db: Session = Depends(get_db)) -> TicketResponse:
    """يجلب تفاصيل تذكرة موجودة برقمها."""
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"لا توجد تذكرة بالرقم {ticket_id}")
    return TicketResponse.model_validate(ticket)


@app.patch("/tickets/{ticket_id}/status", response_model=TicketResponse, tags=["التذاكر"])
def update_ticket_status_endpoint(
    ticket_id: int, payload: TicketStatusUpdate, db: Session = Depends(get_db)
) -> TicketResponse:
    """يحدّث حالة تذكرة موجودة (مثال: إغلاقها بعد الحل)."""
    result = tools_module.update_ticket_status_tool(db=db, ticket_id=ticket_id, new_status=payload.status)
    if not result["success"]:
        raise HTTPException(status_code=404, detail=result.get("error", "تعذّر تحديث التذكرة"))

    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    return TicketResponse.model_validate(ticket)


@app.post("/tickets/{ticket_id}/escalate", response_model=TicketResponse, tags=["التذاكر"])
def escalate_ticket_endpoint(ticket_id: int, db: Session = Depends(get_db)) -> TicketResponse:
    """يصعّد تذكرة موجودة يدويًا (مثال: طلب صريح من موظف الدعم)."""
    ticket = db.query(Ticket).filter(Ticket.id == ticket_id).first()
    if ticket is None:
        raise HTTPException(status_code=404, detail=f"لا توجد تذكرة بالرقم {ticket_id}")

    ticket.escalated = True
    ticket.priority = "urgent"
    ticket.assigned_to = ticket.assigned_to or "فريق الدعم المتقدم"
    db.commit()
    db.refresh(ticket)
    return TicketResponse.model_validate(ticket)


@app.get("/customers/{customer_id}/history", tags=["العملاء"])
def get_customer_history_endpoint(
    customer_id: str, limit: int = Query(default=20, ge=1, le=200), db: Session = Depends(get_db)
) -> dict:
    """يعيد آخر محادثات عميل معيّن، مرتبة من الأحدث إلى الأقدم."""
    result = tools_module.get_customer_history_tool(db=db, customer_id=customer_id, limit=limit)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result.get("error", "فشل جلب السجل"))
    return result


@app.get("/conversations", tags=["العملاء"])
def list_conversations_endpoint(
    limit: int = Query(default=50, ge=1, le=500), db: Session = Depends(get_db)
) -> List[dict]:
    """يعيد آخر المحادثات المسجلة في النظام (لجميع العملاء)، لأغراض المراقبة والتدقيق."""
    rows = db.query(Conversation).order_by(Conversation.created_at.desc()).limit(limit).all()
    return [
        {
            "id": row.id,
            "customer_id": row.customer_id,
            "user_message": row.user_message,
            "agent_reply": row.agent_reply,
            "action": row.action,
            "source": row.source,
            "ticket_id": row.ticket_id,
            "language": row.language,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in rows
    ]
