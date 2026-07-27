"""
smoke_test.py
-------------
سكربت تجريبي سريع (Smoke Test) يشغّل عدة رسائل عبر الوكيل الذكي
مباشرة (دون الحاجة لتشغيل خادم FastAPI) ويطبع لكل رسالة:

    - رسالة العميل
    - قرار الوكيل (action)
    - التصنيف والأولوية
    - الأداة/المصدر المستخدم في اتخاذ القرار
    - الرد النهائي
    - مصدر الإجابة (إن وُجد)
    - رقم التذكرة (إن وُجدت)

طريقة التشغيل:
    python smoke_test.py

يستخدم هذا السكربت قاعدة بيانات منفصلة (smoke_test.db) حتى لا يتعارض
مع قاعدة بيانات التطبيق الفعلية عند التشغيل الحقيقي.
"""

import os

# نجبر المشروع على استخدام قاعدة بيانات خاصة بهذا الاختبار فقط،
# ويجب ضبط هذا المتغير قبل استيراد أي ملف من app حتى يُقرأ بشكل صحيح.
os.environ.setdefault("DATABASE_URL", "sqlite:///./smoke_test.db")

from app.database import init_db, SessionLocal  # noqa: E402
from app import agent as agent_module  # noqa: E402


SAMPLE_MESSAGES = [
    {
        "customer_id": "CUST-SMOKE-1",
        "message": "لم تصلني رسالة إعادة تعيين كلمة المرور، ماذا أفعل؟",
    },
    {
        "customer_id": "CUST-SMOKE-2",
        "message": "أريد ميزة جديدة لتصدير التقارير كملف Excel، هل هذا ممكن؟",
    },
    {
        "customer_id": "CUST-SMOKE-3",
        "message": "حسابي تم اختراقه ويوجد خصم مالي غير معروف، أريد التحدث مع مدير الآن!",
    },
    {
        "customer_id": "CUST-SMOKE-4",
        "message": "What are your support working hours?",
    },
]


def _print_result(index: int, message: str, result: dict) -> None:
    print("=" * 70)
    print(f"[{index}] رسالة العميل: {message}")
    print(f"    القرار (action)      : {result['action']}")
    print(f"    التصنيف (category)   : {result.get('category')}")
    print(f"    الأولوية (priority)  : {result.get('priority')}")
    print(f"    يحتاج تدخل بشري؟     : {result.get('requires_human')}")
    print(f"    الرد النهائي         : {result['reply']}")
    print(f"    المصدر (source)      : {result.get('source')}")
    print(f"    رقم التذكرة          : {result.get('ticket_id')}")
    print(f"    درجة الثقة (RAG)     : {result.get('confidence')}")


def main() -> None:
    print("تشغيل Smoke Test لمشروع وكيل خدمة العملاء الذكي...")
    init_db()
    db = SessionLocal()

    try:
        created_ticket_id = None

        for i, sample in enumerate(SAMPLE_MESSAGES, start=1):
            result = agent_module.run_agent(
                db=db,
                customer_id=sample["customer_id"],
                message=sample["message"],
            )
            _print_result(i, sample["message"], result)

            if result["action"] == "create_ticket" and created_ticket_id is None:
                created_ticket_id = result["ticket_id"]

        # اختبار إضافي: متابعة حالة تذكرة تم إنشاؤها فعليًا أعلاه
        if created_ticket_id is not None:
            follow_up_message = f"أريد معرفة حالة التذكرة TCK-{created_ticket_id}"
            result = agent_module.run_agent(
                db=db, customer_id="CUST-SMOKE-2", message=follow_up_message
            )
            _print_result(len(SAMPLE_MESSAGES) + 1, follow_up_message, result)

        # اختبار إضافي: متابعة رقم تذكرة غير موجود إطلاقًا
        not_found_message = "ما حالة التذكرة TCK-999999؟"
        result = agent_module.run_agent(
            db=db, customer_id="CUST-SMOKE-2", message=not_found_message
        )
        _print_result(len(SAMPLE_MESSAGES) + 2, not_found_message, result)

    finally:
        db.close()

    print("=" * 70)
    print("انتهى تشغيل Smoke Test بنجاح. راجع النتائج أعلاه.")


if __name__ == "__main__":
    main()
