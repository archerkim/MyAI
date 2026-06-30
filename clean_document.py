"""
clean_document.py — чистка документа на входе перед ингестом.

Реальный HTML (и выгрузки) тащат служебный текст — JavaScript-сообщения,
строки фидбэка, копирайты, навигацию. При ингесте они засоряют граф знаний
мусорными узлами (javascript_console, helping, errors_or_warnings). Эта программа
извлекает абзацы документа и отбрасывает boilerplate ДО подачи в экстрактор.

Эвристика отбрасывания абзаца (любое срабатывает):
  - короче min_chars или меньше min_words слов (UI-обрывки, заголовки);
  - содержит маркер boilerplate (javascript, copyright, «please report», URL, @…);
  - почти нет букв (таблицы чисел, навигация).

Выход — чистый текст: абзацы через пустую строку (формат, который понимает
ingest_pipeline._read_paragraphs для .txt). Пиши в файл или stdout.

Примеры:
    python3 clean_document.py Feynman_Lectures_vol1ch1.html -o feynman_clean.txt
    python3 clean_document.py page.html              # отчёт + текст в stdout
    python3 clean_document.py notes.txt --min-words 4 -o out.txt
"""

import argparse
import re
import sys
from pathlib import Path

from ingest_pipeline import _read_paragraphs  # переиспользуем HTML/txt-парсинг

# Подстроки-маркеры служебного текста (регистронезависимо).
_BOILERPLATE_MARKERS = (
    "javascript", "enable scripts", "browser", "operating system", "console",
    "cookie", "copyright", "©", "all rights reserved", "rights reserved",
    "privacy policy", "terms of use", "terms of service", "click here",
    "sign in", "log in", "subscribe", "newsletter", "caltech",
    "errors or warnings", "please report", "we appreciate", "helping",
    "thank you", "contact us", "all images", "permission is granted",
    "powered by", "mathjax", "this page", "for more information, see",
    # блок обратной связи / выходные данные (часто в прозе, ускользает от длины)
    "let us know", "sending us", "new millennium", "editor,", "regards",
    "dear reader", "version #",
)
# URL и email.
_URL_RE = re.compile(r"https?://|www\.|\S+@\S+\.\S+", re.IGNORECASE)


def is_boilerplate(p, min_chars=40, min_words=6):
    """Возвращает (bool, причина|None). True = абзац — служебный, выкинуть."""
    text = p.strip()
    if len(text) < min_chars:
        return True, "too_short"
    if len(text.split()) < min_words:
        return True, "too_few_words"
    low = text.lower()
    for m in _BOILERPLATE_MARKERS:
        if m in low:
            return True, f"marker:{m}"
    if _URL_RE.search(text):
        return True, "url_or_email"
    # доля буквенных символов: настоящая проза — преимущественно буквы
    letters = sum(c.isalpha() or c.isspace() for c in text)
    if letters / max(len(text), 1) < 0.6:
        return True, "low_letter_ratio"
    return False, None


def clean_document(path, min_chars=40, min_words=6, verbose=False):
    """Читает документ → список ЧИСТЫХ абзацев (boilerplate отброшен).

    verbose=True печатает каждый отброшенный абзац с причиной в stderr.
    Возвращает (kept, dropped) — два списка строк-абзацев.
    """
    paragraphs = _read_paragraphs(Path(path))
    kept, dropped = [], []
    for p in paragraphs:
        junk, reason = is_boilerplate(p, min_chars, min_words)
        if junk:
            dropped.append((p, reason))
            if verbose:
                snippet = p[:70].replace("\n", " ")
                print(f"  [drop:{reason}] {snippet}…", file=sys.stderr)
        else:
            kept.append(p)
    return kept, dropped


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Чистка документа (HTML/txt) от boilerplate перед ингестом.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("document", help="путь к HTML или txt документу")
    ap.add_argument("-o", "--out", default=None,
                    help="выходной .txt (по умолчанию stdout)")
    ap.add_argument("--min-chars", type=int, default=40,
                    help="минимальная длина абзаца в символах")
    ap.add_argument("--min-words", type=int, default=6,
                    help="минимальное число слов в абзаце")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="не печатать список отброшенных абзацев")
    args = ap.parse_args(argv)

    doc = Path(args.document)
    if not doc.is_file():
        print(f"[error] документ не найден: {doc}", file=sys.stderr)
        sys.exit(1)

    kept, dropped = clean_document(
        doc, args.min_chars, args.min_words, verbose=not args.quiet
    )
    total = len(kept) + len(dropped)
    print(f"[clean] абзацев: {total} → оставлено {len(kept)}, "
          f"отброшено {len(dropped)}", file=sys.stderr)

    text = "\n\n".join(kept) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"[clean] чистый текст → {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
