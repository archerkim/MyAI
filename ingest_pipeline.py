"""
ingest_pipeline.py — сквозной CLI-ингест документа в .brain-модуль.

Пайплайн (капстон нейро-символического ингеста):

    документ (HTML / txt)
        → чистый текст          (_html_to_text / чтение txt)
        → чанки                 (_chunk_text: абзацы, склейка до --max-chars)
        → семантические триплеты (StreamingExtractor с памятью между чанками;
                                  --no-memory → stateless extract_triples)
        → .brain-модуль          (build_module_from_triples)
        → [--mount] проверка     (монтаж модуля в свежий мозг + stats)

По умолчанию извлечение потоковое: StreamingExtractor протягивает канонические
имена концептов и местоимённый контекст через границы чанков, чтобы узлы
связного документа склеивались. Окупается на дефолт-модели qwen2.5:7b; на
llama3.1:8b память бесполезна — там используйте --no-memory.

Модуль сохраняется ОТДЕЛЬНЫМ файлом (своим module_id) и не вливается в главный
мозг: его можно подключать/отключать по требованию через mount_module /
unmount_module. Шаг --mount только проверяет, что артефакт самодостаточен.

Зависимости: ollama-демон с загруженной моделью (см. extractor.py), brain_core.so.
HTML парсится stdlib-парсером (html.parser) — bs4 не требуется.

Примеры:
    python3 ingest_pipeline.py Feynman_Lectures_vol1ch1.html
    python3 ingest_pipeline.py notes.txt -o physics.brain --module-id 7 --mount
    python3 ingest_pipeline.py doc.html --model qwen2.5:7b --max-chars 800
"""

import argparse
import sys
from html.parser import HTMLParser
from pathlib import Path

from brain_core_wrapper_local import (
    BrainConnectionLocal,
    INT_TO_LABEL,
    build_module_from_triples,
)
from extractor import DEFAULT_MODEL, StreamingExtractor, extract_triples

# Теги, чьё содержимое — не контент документа, а разметка/скрипты/навигация.
# Текст внутри них целиком отбрасывается.
_SKIP_TAGS = frozenset(
    {"script", "style", "head", "noscript", "nav", "header", "footer",
     "form", "button", "select", "option", "svg", "iframe"}
)
# Блочные теги: их границы разрывают абзацы (чтобы строки не слипались в кашу).
_BLOCK_TAGS = frozenset(
    {"p", "div", "br", "li", "tr", "section", "article", "blockquote",
     "h1", "h2", "h3", "h4", "h5", "h6"}
)

# Дефолтный потолок размера чанка в символах. ВАЖНО (замер 2026-06-30): на
# больших чанках модель РЕЗЮМИРУЕТ, а не извлекает исчерпывающе — на одном
# документе чанк 1200 дал 4 триплета, чанк 250 — 10 (×2.5) и без фактической
# ошибки. Поэтому дефолт ~абзац: больше вызовов LLM (медленнее), но втрое полнее
# граф. Подними --max-chars для скорости в ущерб полноте.
DEFAULT_MAX_CHARS = 400


class _TextExtractor(HTMLParser):
    """Достаёт читаемый текст из HTML: пропускает скрипты/стили/навигацию,
    разбивает контент на абзацы по границам блочных тегов."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._paragraphs = []     # накопленные абзацы
        self._buf = []            # текущий собираемый абзац
        self._skip_depth = 0      # >0 пока мы внутри _SKIP_TAGS

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in _BLOCK_TAGS:
            self._flush()

    def handle_data(self, data):
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._buf.append(text)

    def _flush(self):
        if self._buf:
            self._paragraphs.append(" ".join(self._buf))
            self._buf = []

    def paragraphs(self):
        self._flush()
        return self._paragraphs


def _html_to_text_paragraphs(html):
    """HTML-строка → список непустых абзацев."""
    parser = _TextExtractor()
    parser.feed(html)
    return parser.paragraphs()


def _read_paragraphs(path):
    """Читает документ и возвращает список абзацев.

    .html/.htm парсится как HTML; всё остальное считается plain text и
    бьётся на абзацы по пустым строкам.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in (".html", ".htm"):
        return _html_to_text_paragraphs(raw)
    # plain text: абзац = блок строк между пустыми строками
    paragraphs, buf = [], []
    for line in raw.splitlines():
        if line.strip():
            buf.append(line.strip())
        elif buf:
            paragraphs.append(" ".join(buf))
            buf = []
    if buf:
        paragraphs.append(" ".join(buf))
    return paragraphs


def _chunk_text(paragraphs, max_chars=DEFAULT_MAX_CHARS):
    """Склеивает абзацы в чанки не длиннее max_chars.

    Абзацы соединяются «\\n» пока влезают; абзац длиннее лимита идёт
    отдельным чанком как есть (не режем посреди предложения).
    """
    chunks, cur, cur_len = [], [], 0
    for p in paragraphs:
        plen = len(p)
        if cur and cur_len + plen + 1 > max_chars:
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(p)
        cur_len += plen + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def ingest_document(path, out_path, module_id=1, model=DEFAULT_MODEL,
                    max_chars=DEFAULT_MAX_CHARS, strict=True, memory=True):
    """Полный пайплайн: документ → чанки → триплеты → .brain-модуль.

    memory=True (деф.) использует StreamingExtractor: память протягивает
    канонические имена концептов и местоимённый контекст между чанками, чтобы
    узлы связного документа склеивались (energy_of_motion → kinetic_energy).
    Эффект реален на qwen2.5 (дефолт-модель); на llama3.1:8b память бесполезна —
    тогда есть смысл в memory=False (по-чанковый stateless extract_triples).

    Возвращает (triples, stats), где triples — все извлечённые триплеты
    (без dedup), stats — словарь get_stats() итогового модуля.
    """
    paragraphs = _read_paragraphs(path)
    if not paragraphs:
        raise ValueError(f"из документа не удалось извлечь текст: {path}")
    chunks = _chunk_text(paragraphs, max_chars)
    print(f"[1/4] текст: {len(paragraphs)} абзац(ев) → {len(chunks)} чанк(ов) "
          f"(≤{max_chars} симв.)")

    mode = "потоковая память" if memory else "по-чанково (без памяти)"
    print(f"[2/4] извлечение триплетов моделью '{model}' "
          f"(strict={strict}, режим: {mode})...")
    # При memory=True один StreamingExtractor живёт через все чанки и несёт
    # контекст; при memory=False каждый чанк извлекается независимо.
    streamer = (StreamingExtractor(model=model, strict=strict) if memory else None)
    all_triples = []
    for i, chunk in enumerate(chunks, 1):
        try:
            triples = (streamer.feed(chunk) if streamer
                       else extract_triples(chunk, model=model, strict=strict))
        except RuntimeError as e:
            # одна сломавшаяся LLM-выдача не должна ронять весь документ
            print(f"  - чанк {i}/{len(chunks)}: пропущен ({e})", file=sys.stderr)
            continue
        all_triples.extend(triples)
        print(f"  - чанк {i}/{len(chunks)}: +{len(triples)} триплет(ов) "
              f"(всего {len(all_triples)})")

    if not all_triples:
        raise ValueError("ни одного триплета не извлечено — модуль не собран")

    print(f"[3/4] сборка модуля: {len(all_triples)} триплет(ов) "
          f"→ {out_path} (module_id={module_id})")
    # без dedup: повторный (subj,rel,obj) усиливает проводимость ребра — это ок
    stats = build_module_from_triples(all_triples, out_path, module_id=module_id)
    print(f"      узлов: {stats['node_count']}, рёбер: {stats['total_edge_count']}")
    return all_triples, stats


def verify_mount(out_path, module_id):
    """Монтирует только что собранный модуль в СВЕЖИЙ in-memory мозг и печатает
    статистику — доказательство, что .brain самодостаточен и отделим."""
    print(f"[4/4] проверка: монтирую {out_path} в пустой мозг "
          f"(module_id={module_id})...")
    brain = BrainConnectionLocal()
    brain.connect()
    try:
        brain.mount_module(out_path, module_id)
        edges = brain.count_module_edges(module_id)
        stats = brain.get_stats()
        print(f"      смонтировано рёбер модуля: {edges}; "
              f"всего в мозге узлов: {stats['node_count']}, "
              f"рёбер: {stats['total_edge_count']}")
        _print_sample_edges(brain, limit=10)
    finally:
        brain.disconnect()


def _print_sample_edges(brain, limit=10):
    """Печатает несколько рёбер графа в человекочитаемом виде (concept --rel--> concept)."""
    node_ids = list(range(brain.get_stats()["node_count"]))
    edges = brain.get_subgraph_edges(node_ids)
    if not edges:
        return
    print(f"      примеры рёбер (до {limit}):")
    for e in edges[:limit]:
        subj = brain.get_lemma_by_id(e["from_id"])
        obj = brain.get_lemma_by_id(e["to_id"])
        rel = INT_TO_LABEL.get(e["dep_type"], str(e["dep_type"]))
        print(f"        ({subj}) --{rel}--> ({obj})")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Сквозной ингест документа (HTML/txt) в отдельный .brain-модуль.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("document", type=str, help="путь к HTML или txt документу")
    parser.add_argument("-o", "--out", type=str, default=None,
                        help="выходной .brain (по умолчанию <имя_документа>.brain)")
    parser.add_argument("--module-id", type=int, default=1,
                        help="id модуля, которым помечаются все рёбра")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="модель Ollama для извлечения триплетов")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                        help="максимальный размер чанка в символах")
    parser.add_argument("--no-strict", action="store_true",
                        help="отключить пост-фильтр триплетов экстрактора")
    parser.add_argument("--no-memory", action="store_true",
                        help="извлекать чанки независимо (без StreamingExtractor-памяти; "
                             "имеет смысл на llama3.1:8b, где память бесполезна)")
    parser.add_argument("--mount", action="store_true",
                        help="после сборки смонтировать модуль в свежий мозг и показать stats")
    args = parser.parse_args(argv)

    doc_path = Path(args.document).resolve()
    if not doc_path.is_file():
        print(f"[error] документ не найден: {doc_path}", file=sys.stderr)
        sys.exit(1)

    out_path = (Path(args.out).resolve() if args.out
                else doc_path.with_suffix(".brain"))

    try:
        ingest_document(
            doc_path, out_path,
            module_id=args.module_id,
            model=args.model,
            max_chars=args.max_chars,
            strict=not args.no_strict,
            memory=not args.no_memory,
        )
    except (ValueError, RuntimeError) as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

    if args.mount:
        verify_mount(out_path, args.module_id)

    print(f"\n[ok] модуль готов: {out_path}")


if __name__ == "__main__":
    main()
