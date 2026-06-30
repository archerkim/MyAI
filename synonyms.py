"""
synonyms.py — «языковой слой»: нормализация синонимичных концептов к канону.

Идея: понимание начинается с ЯЗЫКА. Прежде чем складывать знания, модель должна
знать, что «atoms in motion», «atoms jiggling» и «atomic movement» — ОДНО И ТО ЖЕ.
Без этого три источника об одном факте дают три разных ребра и не корроборируют.

Этот слой — фундамент: словарь групп-синонимов, каждая группа → один канон.
Нормализация применяется к концептам ПОСЛЕ извлечения (и после ед./мн.-дедупа),
так что эквивалентные формулировки схлопываются в один узел графа.

Словарь сейчас задан вручную («скормлен») — это первый шаг. Дальше его можно
выучивать: эмбеддингами (nomic-embed-text) или LLM-генерацией групп синонимов.
"""

from extractor import _canonicalize_concept

# Канон → множество вариантов (всё в snake_case, ед.число — как после extractor).
SYNONYMS = {
    "motion": {"jiggling", "jiggle", "jiggling_motion", "jiggling_around",
               "movement", "moving", "atomic_motion", "motion_of_atom",
               "atom_motion", "atomic_movement", "perpetual_motion",
               "constant_motion", "molecular_motion"},
    "matter": {"everything", "substance", "material", "stuff", "thing"},
    "attraction": {"pulling", "attract", "attracting", "atomic_pulling",
                   "attraction_between_atom"},
    "particle": {"corpuscle", "tiny_particle", "little_particle"},
    "heat": {"thermal_energy", "warmth"},
}

# вариант → канон (плоская карта для быстрого поиска)
_VARIANT_TO_CANON = {}
for _canon, _vars in SYNONYMS.items():
    for _v in _vars:
        _VARIANT_TO_CANON[_canonicalize_concept(_v)] = _canon
    _VARIANT_TO_CANON[_canonicalize_concept(_canon)] = _canon  # канон → сам себя


def normalize_concept(concept):
    """Концепт → канон: сначала ед./мн.-дедуп, затем синонимы."""
    base = _canonicalize_concept(concept)
    return _VARIANT_TO_CANON.get(base, base)


def normalize_triple(triple):
    """Применяет языковой слой к субъекту и объекту триплета (dict или кортеж)."""
    if isinstance(triple, dict):
        return {"subject": normalize_concept(triple["subject"]),
                "relation": triple["relation"],
                "object": normalize_concept(triple["object"])}
    s, r, o = triple
    return (normalize_concept(s), r, normalize_concept(o))


if __name__ == "__main__":
    for c in ["jiggling", "atoms", "atomic_motion", "perpetual_motion",
              "everything", "molecule", "atomic_pulling"]:
        print(f"  {c:18} -> {normalize_concept(c)}")
