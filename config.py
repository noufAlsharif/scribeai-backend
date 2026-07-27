"""
config.py
---------
ملف الإعدادات المركزي للمشروع.

يقرأ هذا الملف الإعدادات من متغيرات البيئة (ملف .env) باستخدام
pydantic-settings، بحيث يمكن التحكم في سلوك المشروع دون تعديل الكود.

أهم إعداد هنا هو USE_OPENAI:
- إذا كانت قيمته False (الافتراضي)، يعمل الوكيل في "وضع القواعد"
  (Rule-Based Mode) دون الحاجة لأي مفتاح OpenAI، وهذا يسمح للطالب
  بتجربة المشروع فورًا.
- إذا كانت قيمته True ووُجد مفتاح OpenAI صالح، يستخدم الوكيل نموذج
  اللغة لاتخاذ القرار وصياغة الردود.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    إعدادات التطبيق العامة.
    كل حقل هنا يقابل متغير بيئة بنفس الاسم (غير حساس لحالة الأحرف).
    """

    # هل نستخدم OpenAI فعليًا لاتخاذ القرار وتوليد الردود؟
    use_openai: bool = False

    # مفتاح OpenAI API (اختياري إذا كان use_openai=False)
    openai_api_key: str | None = None

    # اسم النموذج المستخدم في حال تفعيل OpenAI
    openai_model: str = "gpt-4.1-mini"

    # رابط الاتصال بقاعدة البيانات (SQLite افتراضيًا لسهولة التشغيل)
    database_url: str = "sqlite:///./customer_service.db"

    # الحد الأدنى لدرجة الثقة في نتيجة البحث داخل قاعدة المعرفة (RAG)
    # أي نتيجة أقل من هذه القيمة تُعامل على أنها "غير موثوقة"
    min_rag_score: float = 0.20

    # اسم ملف قاعدة المعرفة النصية
    knowledge_base_path: str = "data/knowledge_base.txt"

    # إعدادات قراءة ملف البيئة
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# نسخة واحدة مشتركة من الإعدادات تُستخدم في كل أنحاء المشروع
settings = Settings()
