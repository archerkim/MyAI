"""
autolearn.py — автономный цикл обучения «по нарастающей».

Связывает всё в самообучающегося ученика:
    источник материала (Acquirer) → извлечение фактов (StreamingExtractor)
    → TOPIC ROUTER кладёт в нужный модуль (Learner.auto_learn) → состояние растёт.

Acquirer — ПОДКЛЮЧАЕМЫЙ интерфейс:
  • LocalAcquirer — читает локальный корпус, упорядоченный по «классам» (префикс
    «1_», «2_», … в имени файла) — обучение от простого к сложному.
  • WebAcquirer — заготовка под интернет (WebSearch+WebFetch). НЕ активна без
    явного включения: выход в сеть — действие наружу, его согласовывает человек.

Главный модуль (Learner) ведёт словарь / эрудицию / понимание; модули создаются
и активируются сами (knowledge_manager).
"""

import os
import re

from extractor import DEFAULT_MODEL, StreamingExtractor
from knowledge_manager import Learner


def _sentences(text):
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x.strip()]


class LocalAcquirer:
    """Локальный корпус. Файлы вида '1_counting.txt', '2_addition.txt' —
    число-префикс задаёт «класс» (порядок сложности)."""

    def __init__(self, corpus_dir):
        self.dir = corpus_dir

    def _grade(self, name):
        m = re.match(r"(\d+)", name)
        return int(m.group(1)) if m else 999

    def __iter__(self):
        files = [f for f in os.listdir(self.dir) if f.endswith((".txt", ".html"))]
        for f in sorted(files, key=lambda n: (self._grade(n), n)):
            yield f, open(os.path.join(self.dir, f), encoding="utf-8",
                          errors="replace").read()


# КОНТРОЛИРУЕМЫЙ веб: только доверенные открытые источники. Wikipedia REST
# summary API отдаёт ЧИСТЫЙ текст-введение (начальный уровень, без HTML-мусора).
ALLOWED_HOSTS = {"simple.wikipedia.org", "en.wikipedia.org"}


class WebAcquirer:
    """Контролируемый отбор учебного материала из открытых источников.

    Человек в контуре: темы (titles) задаёт человек, источники — из ALLOWED_HOSTS,
    число материалов ограничено (max_items). Это НЕ открытый краулинг.
    lang='simple' = Simple English Wikipedia (текст начального уровня).
    """

    def __init__(self, titles, lang="simple", max_items=None):
        host = f"{lang}.wikipedia.org"
        if host not in ALLOWED_HOSTS:
            raise ValueError(f"источник {host} не в allowlist {ALLOWED_HOSTS}")
        self.titles = titles
        self.host = host
        self.max_items = max_items

    def _summary(self, title):
        import json
        import urllib.parse
        import urllib.request
        url = (f"https://{self.host}/api/rest_v1/page/summary/"
               f"{urllib.parse.quote(title.replace(' ', '_'))}")
        req = urllib.request.Request(
            url, headers={"User-Agent": "MyAI-learner/0.1 (educational; non-commercial)"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.load(r).get("extract", "")

    def __iter__(self):
        titles = self.titles[:self.max_items] if self.max_items else self.titles
        for t in titles:
            try:
                text = self._summary(t)
            except Exception as e:                       # сеть/404 — пропуск, не падаем
                print(f"  [web:{t}] ошибка загрузки: {e}")
                continue
            if text and len(text) > 40:
                yield f"web:{t}", text


def learn_loop(learner, acquirer, model=DEFAULT_MODEL, max_items=None, verbose=True):
    """Прогоняет материалы из acquirer через ученика. Возвращает финальное
    состояние. Память StreamingExtractor живёт в пределах одного материала."""
    learned = 0
    for name, text in acquirer:
        ex = StreamingExtractor(model=model)
        triples = []
        for sent in _sentences(text):
            triples += [(t["subject"], t["relation"], t["object"]) for t in ex.feed(sent)]
        if not triples:
            if verbose:
                print(f"  [{name}] фактов не извлечено — пропуск")
            continue
        topic, mid, stats = learner.auto_learn(text, triples, model)
        learned += 1
        if verbose:
            print(f"  [{name}] → тема '{topic}' (модуль {mid}): "
                  f"+{len(triples)} фактов, модуль теперь {stats['node_count']} концептов")
        if max_items and learned >= max_items:
            break
    return learner.state()


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Автономный цикл обучения по локальному корпусу.")
    ap.add_argument("corpus", help="каталог с материалами (файлы '1_*.txt', '2_*.txt', …)")
    ap.add_argument("--workdir", default="./knowledge")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-items", type=int, default=None)
    a = ap.parse_args()

    learner = Learner(a.workdir)
    state = learn_loop(learner, LocalAcquirer(a.corpus), a.model, a.max_items)
    print("\n=== СОСТОЯНИЕ УЧЕНИКА ===")
    print(json.dumps(state, ensure_ascii=False, indent=2))
