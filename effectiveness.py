"""
effectiveness.py — метрика ЭФФЕКТИВНОСТИ всей нейро-символической системы.

graph_quality.py измеряет КАЧЕСТВО СУБСТРАТА (насколько разумен сам граф фактов).
Но система — это не только граф: это граф ПЛЮС способность рассуждать над ним
(reasoner), предвосхищать (predictor) и размышлять на стыке источников (reflection).
Эффективность = сколько НАДЁЖНОГО ПОНИМАНИЯ система производит на имеющихся фактах.

Скоркарта из шести осей (каждая 0..1, эталон не нужен — всё из графа):

  substrate      — композит graph_quality: соундность самих поглощённых фактов.
  amplification  — доля знания, ПОРОЖДЁННОГО выводом: derived/(derived+asserted).
                   Сколько граф «достраивает» сверх прямо сказанного.
  soundness      — доля выводов, НЕ вносящих логических нарушений (циклы/
                   противоречия). Усиление бесполезно, если плодит мусор.
  groundedness   — доля выводов с ПОЛНЫМ доказательством до фактов-листьев графа.
                   Проверяет, что каждый вывод трассируем (заземлён), а не висит.
  emergence      — доля выводов, рождённых на СТЫКЕ ≥2 источников и неразложимых
                   по отдельности (reflection): мера «1+1 > 2». Нужны ≥2 модуля.
  corroboration  — доля транзитивных заключений с ≥2 независимыми свидетелями:
                   устойчивость к ошибке одного источника.

Композит — взвешенная сумма (_WEIGHTS). Это ИНДИКАТОР для сравнения версий и ловли
регрессий, не абсолютная истина. Запускать на ≥2 смонтированных модулях, иначе
emergence/corroboration структурно нулевые (это честно, а не баг).
"""

import argparse
from collections import defaultdict

import graph_quality
from brain_core_wrapper_local import ALLOWED_RELATIONS
from reasoner import forward_chain, validate, find_contradictions
from reflection import (synergistic_deductions, convergent_conclusions,
                        _leaves, GraphReflector)

_WEIGHTS = {  # сумма = 1.0
    "substrate": 0.30,
    "amplification": 0.15,
    "soundness": 0.20,
    "groundedness": 0.15,
    "emergence": 0.12,
    "corroboration": 0.08,
}


def _violating(facts):
    """Множество рёбер, попавших в логические нарушения (повторяет graph_quality)."""
    bad = set()
    for kind, who in validate(facts):
        if kind.startswith("self_reference"):
            bad.add(tuple(who[0]))
        elif kind.startswith("cycle"):
            rel = kind[kind.index("[") + 1:-1]
            for a, b in zip(who, who[1:]):
                bad.add((a, rel, b))
    for kind, who in find_contradictions(facts):
        for t in who:
            if isinstance(t, tuple) and len(t) == 3:
                bad.add(t)
    return bad


def evaluate(facts, fact_sources):
    """facts — множество троек; fact_sources — тройка → set(module_id).
    Возвращает скоркарту с осями, композитом и диагностиками."""
    facts = set(facts)
    asserted = len(facts)
    if asserted == 0:
        return {"composite": 0.0, "error": "пустой граф"}

    # --- субстрат ---
    substrate = graph_quality.evaluate(facts)["composite"]

    # --- вывод ---
    derived, proofs = forward_chain(facts)
    nd = len(derived)

    # amplification: какая доля итогового знания получена выводом
    amplification = nd / (nd + asserted) if (nd + asserted) else 0.0

    # soundness: доля выводов, не участвующих в нарушениях полного графа
    if nd:
        bad = _violating(facts | set(derived))
        bad_derived = sum(1 for d in derived if d in bad)
        soundness = 1.0 - bad_derived / nd
    else:
        soundness = 1.0   # нечего портить

    # groundedness: доля выводов, чьи листья доказательства — реальные факты графа
    if nd:
        grounded = 0
        for d in derived:
            lv = _leaves(proofs, d)
            if lv and lv <= facts:        # все листья присутствуют в графе
                grounded += 1
        groundedness = grounded / nd
    else:
        groundedness = 1.0

    # emergence: доля выводов, эмерджентных на стыке источников
    insights, _ = synergistic_deductions(facts, fact_sources)
    emergence = (len(insights) / nd) if nd else 0.0

    # corroboration: доля знания с ≥2 НЕЗАВИСИМЫМИ опорами. Две формы опоры:
    #   (a) СОГЛАСИЕ источников — один факт утверждён ≥2 модулями (виден только в
    #       сыром провенансе до слияния: C-граф хранит один module_id на ребро);
    #   (b) КОНВЕРГЕНЦИЯ путей — транзитивное заключение с ≥2 промежуточными.
    agreed = [t for t, ms in fact_sources.items() if len(ms) >= 2]
    conv = convergent_conclusions(facts, fact_sources)
    transitive_derived = [d for d in derived if d[1] in ("is_a", "part_of", "causes")]
    supported = len(agreed) + len(conv)
    claims = asserted + len(transitive_derived)
    corroboration = min(supported / claims, 1.0) if claims else 0.0

    subs = {
        "substrate": substrate,
        "amplification": amplification,
        "soundness": soundness,
        "groundedness": groundedness,
        "emergence": emergence,
        "corroboration": corroboration,
    }
    composite = sum(_WEIGHTS[k] * subs[k] for k in _WEIGHTS)

    n_sources = len({m for ms in fact_sources.values() for m in ms})
    return {
        "composite": round(composite, 3),
        "subscores": {k: round(v, 3) for k, v in subs.items()},
        "diagnostics": {
            "asserted_facts": asserted,
            "derived_facts": nd,
            "emergent_insights": len(insights),
            "convergent_conclusions": len(conv),
            "multi_source_agreed": len(agreed),
            "sources": n_sources,
            "substrate_composite": substrate,
        },
    }


def evaluate_brain(brain):
    facts, fact_sources = GraphReflector(brain).load()
    return evaluate(facts, fact_sources)


def facts_from_raw(raw_triples, normalize=True):
    """Строит (facts, fact_sources) из СЫРЫХ триплетов [s, r, o, module] ДО слияния
    в граф. Только так видно СОГЛАСИЕ источников: C-граф хранит один module_id на
    ребро, поэтому один факт от N источников там схлопывается. normalize=True
    прогоняет концепты через языковой слой синонимов (варианты → канон), чтобы
    «atoms in motion» и «atomic movement» считались одним фактом и корроборировали."""
    from collections import defaultdict
    facts, fact_sources = set(), defaultdict(set)
    norm = None
    if normalize:
        from synonyms import normalize_triple
        norm = normalize_triple
    for s, r, o, m in raw_triples:
        tri = norm((s, r, o)) if norm else (s, r, o)
        if r not in ALLOWED_RELATIONS or tri[0] == tri[2]:
            continue
        facts.add(tri)
        fact_sources[tri].add(m)
    return facts, fact_sources


def evaluate_raw(raw_triples, normalize=True):
    facts, fact_sources = facts_from_raw(raw_triples, normalize)
    return evaluate(facts, fact_sources)


def print_report(rep):
    if "error" in rep:
        print(f"[effectiveness] {rep['error']}")
        return
    print(f"=== ЭФФЕКТИВНОСТЬ СИСТЕМЫ: {rep['composite']:.3f} / 1.000 ===")
    print("оси (вес):")
    for k, w in _WEIGHTS.items():
        bar = "█" * round(rep["subscores"][k] * 20)
        print(f"  {k:14} {rep['subscores'][k]:.3f}  {bar:<20} (вес {w})")
    print("диагностики:")
    for k, v in rep["diagnostics"].items():
        print(f"  {k:24} {v}")
    if rep["diagnostics"]["sources"] < 2:
        print("\n  ⓘ источников <2 → emergence/corroboration структурно ≈0 "
              "(нужны ≥2 модуля с общими концептами).")


# --- встроенный реальный корпус для --live: ДВА источника, общие концепты ---
_LIVE_CORPUS = [
    # источник 1 — про энергию и теплоту
    ("Energy is the capacity to do work. Heat is a form of energy. "
     "Temperature measures the average kinetic energy of particles. "
     "Kinetic energy is the energy of motion."),
    # источник 2 — про атомы и газ (общие концепты: energy, particle, motion)
    ("An atom is the smallest unit of matter. A molecule is made of atoms. "
     "A gas consists of particles in constant motion. "
     "Pressure is caused by particles hitting the walls."),
]


def build_live_brain(model=None):
    """Реальный сквозной прогон: два текста → extract_triples → два модуля C-ядра."""
    from brain_core_wrapper_local import BrainConnectionLocal
    from extractor import extract_triples, DEFAULT_MODEL
    model = model or DEFAULT_MODEL
    brain = BrainConnectionLocal(); brain.connect()
    total = 0
    for i, text in enumerate(_LIVE_CORPUS, start=1):
        triples = extract_triples(text, model=model)
        brain.set_module_id(i)
        for t in triples:               # extract_triples → list[dict subject/relation/object]
            brain.add_triple(t["subject"], t["relation"], t["object"])
        total += len(triples)
        print(f"  источник {i}: {len(triples)} триплетов извлечено qwen")
    brain.set_module_id(0)
    print(f"  всего фактов в графе: {total}")
    return brain


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Метрика эффективности нейро-символической системы.")
    ap.add_argument("modules", nargs="*", help="пути к .brain (каждый — отдельный источник)")
    ap.add_argument("--live", action="store_true",
                    help="реальный прогон: извлечь встроенный 2-источниковый корпус через qwen")
    ap.add_argument("--raw", help="JSON-список сырых [s,r,o,module] (провенанс ДО слияния)")
    ap.add_argument("--no-normalize", action="store_true",
                    help="не применять слой синонимов к --raw")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()

    from brain_core_wrapper_local import BrainConnectionLocal

    if a.raw:
        import json
        raw = json.load(open(a.raw))
        print(f"[raw] сырых триплетов: {len(raw)}  (synonyms={'off' if a.no_normalize else 'on'})")
        print_report(evaluate_raw(raw, normalize=not a.no_normalize))
    elif a.live:
        print("[live] извлечение реального корпуса через Ollama…")
        brain = build_live_brain(a.model)
        print_report(evaluate_brain(brain))
        brain.disconnect()
    elif a.modules:
        brain = BrainConnectionLocal(); brain.connect()
        for i, path in enumerate(a.modules, start=1):
            brain.mount_module(path, i)   # каждый модуль = свой источник
        print_report(evaluate_brain(brain))
        brain.disconnect()
    else:
        ap.error("укажи .brain-модули или --live")
