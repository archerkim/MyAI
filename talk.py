"""
talk.py — «поговорить» с графом знаний через qwen.

Вопрос → находим якорные концепты → собираем их окрестность (двунаправленный BFS)
→ qwen объясняет СТРОГО по этим рёбрам (заземление; если данных нет — честно
говорит). Так можно проверить, что граф реально выучил.

Запуск:
    python3 talk.py mind/physics.brain                  # диалог (REPL)
    python3 talk.py mind/physics.brain -q "что такое сила?"
"""

import argparse
import re

from brain_core_wrapper_local import BrainConnectionLocal, INT_TO_LABEL
from synonyms import normalize_concept
from verbalizer import format_subgraph_for_prompt

_STOP = {"what", "is", "are", "the", "a", "an", "of", "and", "how", "do", "does",
         "to", "in", "they", "it", "why", "with", "что", "такое", "как", "и",
         "для", "это", "почему"}
_GROUND = (
    "Ты отвечаешь СТРОГО по фрагменту графа знаний ниже — только его факты, без "
    "внешних знаний. Если фрагмент пуст или не относится к вопросу — ответь "
    "'В графе нет данных для ответа.' Иначе объясни в 2–4 предложениях по связям.")


def _neighborhood(edges, anchors, depth=2):
    adj = {}
    for e in edges:
        adj.setdefault(e["from_id"], set()).add(e["to_id"])
        adj.setdefault(e["to_id"], set()).add(e["from_id"])
    seen, frontier = set(anchors), set(anchors)
    for _ in range(depth):
        nxt = set().union(*[adj.get(x, set()) for x in frontier]) if frontier else set()
        nxt -= seen; seen |= nxt; frontier = nxt
    return seen


def _anchors(brain, question):
    ids = []
    for w in re.findall(r"[a-zA-Zа-яА-Я]+", question.lower()):
        if w in _STOP:
            continue
        nid = brain.get_node_id(normalize_concept(w))
        if nid != -1 and nid not in ids:
            ids.append(nid)
    return ids


def answer(brain, all_edges, question, model="qwen2.5:7b", show_evidence=True):
    import ollama
    ids = _anchors(brain, question)
    if not ids:
        return "(в графе нет ни одного концепта из вопроса)"
    nodes = _neighborhood(all_edges, ids, depth=2)
    edges = sorted([e for e in all_edges if e["from_id"] in nodes and e["to_id"] in nodes],
                   key=lambda e: -e["conductance"])[:16]
    if show_evidence:
        print(f"  [база: {len(edges)} рёбер вокруг {[brain.get_lemma_by_id(i) for i in ids]}]")
        for e in edges[:8]:
            print(f"     ({brain.get_lemma_by_id(e['from_id'])}) "
                  f"--{INT_TO_LABEL.get(e['dep_type'],'?')}--> "
                  f"({brain.get_lemma_by_id(e['to_id'])})")
    sign = [{"id": x, "lemma": brain.get_lemma_by_id(x)} for x in nodes]
    ev = format_subgraph_for_prompt(sign, edges)
    r = ollama.Client().chat(model=model, options={"temperature": 0}, messages=[
        {"role": "system", "content": _GROUND},
        {"role": "user", "content": f"{ev}\n\nВопрос: {question}"}])
    return r.message.content.strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Поговорить с графом знаний.")
    ap.add_argument("module", help="путь к .brain")
    ap.add_argument("-q", "--question", default=None, help="один вопрос (иначе диалог)")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--module-id", type=int, default=1)
    a = ap.parse_args(argv)

    b = BrainConnectionLocal(); b.connect(); b.mount_module(a.module, a.module_id)
    all_edges = b.get_subgraph_edges(list(range(b.get_stats()["node_count"])))
    print(f"[граф: {b.get_stats()['node_count']} концептов, {len(all_edges)} рёбер]")
    try:
        if a.question:
            print("ОТВЕТ:", answer(b, all_edges, a.question, a.model))
        else:
            print("Диалог с графом. Пустая строка или 'exit' — выход.\n")
            while True:
                try:
                    q = input("вопрос> ").strip()
                except EOFError:
                    break
                if not q or q.lower() in ("exit", "quit", "выход"):
                    break
                print("ОТВЕТ:", answer(b, all_edges, q, a.model), "\n")
    finally:
        b.disconnect()


if __name__ == "__main__":
    main()
