"""
lang_utils.py — Détection de langue simple pour forcer la langue de réponse
"""

from langdetect import detect, LangDetectException


def detect_response_language(question: str) -> str:
    try:
        lang_code = detect(question)
    except LangDetectException:
        return "French"

    return "English" if lang_code == "en" else "French"


def language_instruction(question: str) -> str:
    lang = detect_response_language(question)
    return (
        f"\n\nIMPORTANT : tu dois répondre exclusivement en {lang}, "
        f"quelle que soit la langue du contexte fourni ci-dessus."
    )