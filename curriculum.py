"""
curriculum.py — планировщик «что изучать дальше» (порядок ингеста документов).

Эмпирически (см. эксперимент с порядком) при ПОТОКОВОМ обучении порядок важен:
если основы скормить первыми, последующий материал «прицепляется» к ним через
память экстрактора, и граф выходит связнее. Как у человека: математику до физики,
а не наоборот.

Этот модуль выбирает порядок АВТОМАТИЧЕСКИ, по зависимостям между концептами:
  - документ ОПРЕДЕЛЯЕТ концепт, если тот стоит субъектом его триплетов
    (документ про него что-то утверждает);
  - документ ЗАВИСИТ от другого, если ИСПОЛЬЗУЕТ концепт, который ОПРЕДЕЛЯЕТ
    другой (ещё не изученный) документ.

Жадный выбор «следующего»: на каждом шаге берём документ с МИНИМУМОМ невыполненных
пререквизитов (его основы уже изучены), а среди таких — тот, что РАЗБЛОКИРУЕТ
больше будущего материала и сильнее связан с уже известным. Это даёт топологический
порядок «основы → надстройки» и максимизирует связность строящегося графа.
"""

import argparse

from extractor import DEFAULT_MODEL, extract_triples


def doc_profile(path, model=DEFAULT_MODEL):
    """Профиль документа: какие концепты он определяет и какие использует.

    Извлекаем ПО ПРЕДЛОЖЕНИЯМ: на целом документе (большой чанк) модель
    резюмирует и теряет концепты, а профиль зависимостей тогда пустой.
    """
    import re
    text = open(path, encoding="utf-8", errors="replace").read().strip()
    sents = [x.strip() for x in re.split(r"(?<=\.)\s+", text) if x.strip()]
    defines, objects, n = set(), set(), 0
    for s in sents:
        for t in extract_triples(s, model=model):
            defines.add(t["subject"]); objects.add(t["object"]); n += 1
    return {"defines": defines, "uses": defines | objects, "triples": n}


def schedule(profiles):
    """profiles: {имя: {'defines','uses'}} → список имён в порядке изучения.

    Возвращает (order, log), где log[i] поясняет выбор i-го документа.
    """
    definer = {}                                           # концепт → кто его определяет
    for name, p in profiles.items():
        for c in p["defines"]:
            definer.setdefault(c, set()).add(name)

    order, log, known = [], [], set()
    remaining = dict(profiles)
    while remaining:
        best, best_score, best_info = None, None, None
        for name, p in remaining.items():
            # невыполненные пререквизиты: использует концепт, который определяет
            # ДРУГОЙ ещё не изученный документ
            unmet = {c for c in p["uses"]
                     if c not in p["defines"]
                     and any(o != name and o in remaining for o in definer.get(c, ()))}
            # разблокировка: сколько оставшихся документов используют то, что этот определяет
            unlock = sum(1 for o, q in remaining.items()
                         if o != name and (p["defines"] & q["uses"]))
            grounding = len(p["uses"] & known)             # связь с уже изученным
            score = (-len(unmet), unlock, grounding)       # приоритет: меньше долгов → больше разблокировки → связь
            if best_score is None or score > best_score:
                best_score, best, best_info = score, name, (len(unmet), unlock, grounding)
        order.append(best)
        log.append(f"{best}: невыполн.пререкв={best_info[0]}, разблокирует={best_info[1]}, "
                   f"связь_с_известным={best_info[2]}")
        known |= remaining[best]["defines"]
        del remaining[best]
    return order, log


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Спланировать порядок изучения документов.")
    ap.add_argument("documents", nargs="+", help="пути к документам (txt/html)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()

    print(f"[*] анализ {len(a.documents)} документов моделью {a.model}...")
    profiles = {}
    for path in a.documents:
        prof = doc_profile(path, a.model)
        profiles[path] = prof
        name = path.rsplit("/", 1)[-1]
        print(f"    {name}: определяет {len(prof['defines'])} концептов "
              f"({prof['triples']} триплетов)")

    order, log = schedule(profiles)
    print("\n=== РЕКОМЕНДУЕМЫЙ ПОРЯДОК ИЗУЧЕНИЯ ===")
    for i, (path, why) in enumerate(zip(order, log), 1):
        print(f"  {i}. {path.rsplit('/', 1)[-1]}   [{why}]")
