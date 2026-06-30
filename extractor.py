"""
extractor.py — нейро-символический ингест (локальный LLM через Ollama).

Превращает сырой текст в структурированные семантические триплеты
(subject, relation, object) с каноническими концептами и контролируемым
словарём отношений, используя локальную модель Ollama (structured outputs
через JSON-schema, генерируемую из Pydantic).

Это решает две проблемы старого spaCy-пайплайна:
  1. Полисемия: LLM выдаёт КАНОНИЧЕСКИЙ концепт ("mass_physics" vs "mass_biology"),
     а не поверхностную лемму — модули складываются без ложных мостов.
  2. Смысл: рёбра — это семантические отношения (causes, is_a, ...), а не
     синтаксис, поэтому граф годится для рассуждений и объяснимости.

Зависимости (установить перед использованием):
    pip install ollama                 # тянет за собой и pydantic
    ollama serve                       # демон Ollama должен быть запущен
    ollama pull llama3.1               # или qwen2.5:7b для русского текста

Никаких ключей и интернета не требуется — всё считается локально.
"""

from enum import Enum
from typing import List

from pydantic import BaseModel, Field

from brain_core_wrapper_local import ALLOWED_RELATIONS

# Модель по умолчанию. qwen2.5:7b выбрана по результатам бенчмарка (24 предложения):
# recall 96% против 88% у llama3.1:8b, и — ключевое — НАТИВНО решает пассивный
# залог и defined_as, а также реально использует контекст StreamingExtractor
# (склейка имён концептов между предложениями). llama3.1:8b всё это игнорировала.
# Запасной вариант (быстрее, ниже качество): model="llama3.1".
DEFAULT_MODEL = "qwen2.5:7b"

# Пост-фильтр триплетов: детерминированная страховка от типичных артефактов
# слабых моделей (фразы вместо концептов, протёкшие отрицания). Работает
# независимо от модели и промпта.
MAX_CONCEPT_WORDS = 3  # концепт = 1–3 слова; длиннее — это описание/фраза
_NEGATION_TOKENS = frozenset(
    {"no", "not", "without", "cannot", "never", "none", "neither", "nor"}
)


def _is_valid_concept(concept):
    """True, если строка похожа на атомарный канонический концепт, а не на
    фразу/формулу/артефакт отрицания."""
    if not concept or not concept.strip():
        return False
    words = concept.split("_")
    if len(words) > MAX_CONCEPT_WORDS:
        return False  # «average_kinetic_energy_of_particles» — это фраза
    if any(w in _NEGATION_TOKENS for w in words):
        return False  # «matter_transfer_no», «cannot_be_created» — протёкшее отрицание
    return True

# --- модель данных для structured outputs ---
# Динамически строим Enum из единого источника правды (RELATION_TYPE_MAP),
# чтобы LLM была ограничена ровно теми отношениями, что знает C-ядро.
# JSON-schema этого Enum уходит в Ollama как грамматика → модель физически
# не может выдать отношение вне словаря.
RelationEnum = Enum("RelationEnum", {r: r for r in ALLOWED_RELATIONS}, type=str)


class Triple(BaseModel):
    subject: str = Field(
        description="Каноническое имя концепта-субъекта: lowercase snake_case, "
        "ед. число, без артиклей. При полисемии уточняй домен суффиксом "
        "(напр. 'mass_physics', 'mass_biology'). Один и тот же концепт всегда "
        "пиши одинаково."
    )
    relation: RelationEnum = Field(description="Тип семантической связи из словаря.")
    object: str = Field(description="Каноническое имя концепта-объекта (те же правила, что и subject).")


class Extraction(BaseModel):
    triples: List[Triple] = Field(
        description="Список фактов, явно утверждаемых текстом. Не выдумывай "
        "связи, которых нет в тексте."
    )


_SYSTEM_PROMPT = (
    "Ты — экстрактор графа знаний. Извлекай из текста только ЯВНО утверждаемые "
    "факты как триплеты (субъект, отношение, объект).\n\n"
    "Правила канонизации концептов (subject и object):\n"
    "- lowercase snake_case, единственное число, без артиклей/стоп-слов;\n"
    "- ОДИН АТОМАРНЫЙ КОНЦЕПТ из 1–3 слов (kinetic_energy, joule, hot_object). "
    "НИКОГДА не помещай в концепт целую фразу, описание или формулу "
    "(НЕ 'measure_of_average_kinetic_energy', НЕ 'product_of_frequency_and_wavelength');\n"
    "- один концепт — всегда одинаковое имя;\n"
    "- при многозначности уточняй домен суффиксом (mass_physics vs mass_biology);\n"
    "- предпочитай устоявшиеся термины, а не дословные формулировки.\n\n"
    "Выбор отношения (РАЗЛИЧАЙ внимательно):\n"
    "- единицы измерения → measured_in (frequency measured_in hertz), НЕ is_a;\n"
    "- «X is a/type of Y» (Y — категория) → is_a; «X is defined as / is the ...» "
    "(Y — расшифровка смысла) → defined_as;\n"
    "- «X is part of / belongs to / is contained in Y», X — компонент Y → "
    "part_of, а НЕ is_a (nucleus part_of atom, НЕ nucleus is_a atom);\n"
    "- «X is the opposite of Y», «X versus Y» → opposite_of как ОТДЕЛЬНОЕ ребро "
    "(heat opposite_of cold). НИКОГДА не вшивай слово opposite в имя концепта;\n"
    "- «X is an example/instance of Y» → example_of (water example_of liquid);\n"
    "- enables = X делает Y возможным, но Y может и без X (catalyst enables "
    "reaction); causes = X прямо порождает Y. Катализатор/условие → enables;\n"
    "- свойство → has_property.\n"
    "ПАССИВНЫЙ ЗАЛОГ: «Y is caused by X» означает X causes Y (X — причина!). "
    "«Y is used by X» → X used_for ... Всегда ставь субъектом настоящего деятеля, "
    "а не грамматическое подлежащее пассива.\n\n"
    "Запреты:\n"
    "- НЕ выдумывай таксономию и категории, которых нет в тексте "
    "(если не сказано «X is a vector» — не пиши этого);\n"
    "- учитывай ОТРИЦАНИЯ: если текст говорит, что связи НЕТ "
    "(«wave transfers energy WITHOUT matter»), НЕ создавай такой триплет;\n"
    "- математические равенства/формулы (F = m·a) пропускай: для них нет "
    "подходящего отношения в словаре.\n\n"
    "Используй ТОЛЬКО отношения из заданного словаря. Если связь реальна, но не "
    "подходит ни под одно конкретное — используй related_to. Верни строго JSON по схеме.\n\n"
    "Примеры (обрати внимание на АТОМАРНЫЕ концепты и отброшенные отрицания):\n"
    "Текст: 'Gravity is a force. Energy is measured in joules.'\n"
    "→ (gravity, is_a, force), (energy, measured_in, joule)\n"
    "Текст: 'Temperature is a measure of the average kinetic energy of particles.'\n"
    "→ (temperature, related_to, kinetic_energy)   "
    "[НЕ (temperature, defined_as, average_kinetic_energy_of_particles)]\n"
    "Текст: 'A wave transfers energy without transferring matter.'\n"
    "→ (wave, causes, energy_transfer)   "
    "[про matter ничего: связь отрицается]\n"
    "Текст: 'Heat flows from a hot object to a cold object.'\n"
    "→ (heat, has_property, flow)   "
    "[НЕ flow_from_hot_object_to_cold_object — это фраза, а не концепт]\n"
    "Текст: 'The nucleus is part of an atom.'\n"
    "→ (nucleus, part_of, atom)   [part_of, НЕ is_a]\n"
    "Текст: 'Heat is the opposite of cold.'\n"
    "→ (heat, opposite_of, cold)   "
    "[opposite_of как ребро; НЕ (heat, is_a, opposite_of_cold)]\n"
    "Текст: 'Acceleration is caused by force.'\n"
    "→ (force, causes, acceleration)   "
    "[пассив: деятель force — субъект; направление НЕ перевёрнуто]\n"
    "Текст: 'A catalyst enables a chemical reaction.'\n"
    "→ (catalyst, enables, chemical_reaction)   [enables, НЕ causes]\n"
    "Текст: 'Copper is an example of a conductor.'\n"
    "→ (copper, example_of, conductor)"
)


def extract_triples(text, model=DEFAULT_MODEL, host=None, strict=True):
    """Извлекает семантические триплеты из текста локальной моделью Ollama.

    Параметры:
        text   — исходный текст для извлечения.
        model  — имя модели Ollama (по умолчанию llama3.1 = 8B).
        host   — адрес демона Ollama (по умолчанию http://localhost:11434).
        strict — отбрасывать триплеты с фразами-концептами и протёкшими
                 отрицаниями (см. _is_valid_concept). По умолчанию True.

    Возвращает список dict вида {"subject", "relation", "object"},
    готовый к подаче в BrainConnectionLocal.add_triple(**triple).

    Примечание: переформулировка/нормализация текста ПЕРЕД извлечением
    (rewrite-проход) проверена и ОТКЛОНЕНА — llama3.1:8b неуправляемо
    перефразирует (переворачивает активные предложения в пассив, дополняет
    выдуманными фактами, меняет язык), что роняет recall с 88% до 29%.
    Пассивный залог во входе — известный предел 8B; для надёжного разбора нужна
    более крупная модель (qwen2.5).

    Бросает RuntimeError с понятным текстом, если демон Ollama недоступен или
    модель не загружена (ollama pull <model>).
    """
    import ollama  # ленивый импорт: модуль грузится и без установленного SDK

    client = ollama.Client(host=host) if host else ollama.Client()
    return _run_extraction(client, model, text, strict)


def _run_extraction(client, model, user_content, strict):
    """Один вызов экстракции: chat → парсинг → пост-фильтр. Общий для
    одиночного extract_triples и потокового StreamingExtractor."""
    import ollama

    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            # JSON-schema из Pydantic-модели — Ollama превращает её в грамматику
            # и гарантирует валидный JSON ровно нужной формы (включая enum отношений).
            format=Extraction.model_json_schema(),
            # детерминизм: извлечение фактов — не творческая задача.
            options={"temperature": 0},
        )
    except ConnectionError as e:
        raise RuntimeError(
            "Не удалось подключиться к Ollama. Запущен ли демон? `ollama serve`"
        ) from e
    except ollama.ResponseError as e:
        # типичный случай: модель не скачана.
        raise RuntimeError(
            f"Ollama вернула ошибку для модели '{model}': {e}. "
            f"Возможно, нужно `ollama pull {model}`."
        ) from e

    extraction = Extraction.model_validate_json(response.message.content)
    triples = [
        {"subject": t.subject, "relation": t.relation.value, "object": t.object}
        for t in extraction.triples
    ]

    if strict:
        kept, dropped = [], 0
        for tr in triples:
            if _is_valid_concept(tr["subject"]) and _is_valid_concept(tr["object"]):
                kept.append(tr)
            else:
                dropped += 1
        if dropped:
            print(f"[extractor] отброшено {dropped} триплет(ов) пост-фильтром "
                  f"(фразы-концепты / отрицания)")
        return kept

    return triples


class StreamingExtractor:
    """Потоковый экстрактор с КРАТКОСРОЧНОЙ ПАМЯТЬЮ.

    Текст обычно связный: предложения перетекают друг в друга местоимениями
    («it», «these particles») и повторно упоминают одни концепты разными
    словами. При независимом извлечении чанков это теряется. Память несёт через
    границы чанков две вещи:

      1. recent_text — предыдущий фрагмент, чтобы модель разрешала местоимения
         в реальные концепты (it → atom), а не плодила субъекты вроде "it".
      2. known_concepts — канонические имена уже добавленных концептов. Модель
         просят ПЕРЕИСПОЛЬЗОВАТЬ те же имена при повторной встрече → узлы
         склеиваются между чанками (energy_of_motion не разойдётся с
         kinetic_energy). Это прямо усиливает связность графа.

    Использование:
        ex = StreamingExtractor(model="llama3.1")
        for chunk in chunks:
            triples = ex.feed(chunk)   # учитывает контекст предыдущих чанков

    ВАЖНО (эмпирически, 2026-06-30): эффект зависит от модели.
    - llama3.1:8b — память бесполезна (8B игнорирует мета-контекст, как и CoT/
      rewrite); иногда хуже.
    - qwen2.5:7b — память РАБОТАЕТ для склейки имён: каноническое имя из ранних
      предложений протягивается дальше (S2 'energy of motion' → kinetic_energy
      вместо разрозненного energy_gain). Кореференцию qwen и без памяти решает
      хорошо, так что там выигрыш мал.
    Это ОТДЕЛЬНЫЙ опциональный класс; дефолтный путь ингеста — одиночный
    extract_triples. Память полезна при ингесте СВЯЗНОГО документа на qwen2.5.
    """

    def __init__(self, model=DEFAULT_MODEL, host=None, strict=True,
                 memory_size=40, keep_context_chars=400):
        import ollama
        self.client = ollama.Client(host=host) if host else ollama.Client()
        self.model = model
        self.strict = strict
        self.memory_size = memory_size          # сколько последних концептов помнить
        self.keep_context_chars = keep_context_chars  # сколько символов прошлого фрагмента нести
        self.known_concepts = []                # упорядоченный dedup (последние — в конце)
        self.recent_text = ""

    def _remember(self, triples):
        for tr in triples:
            for c in (tr["subject"], tr["object"]):
                if c in self.known_concepts:
                    self.known_concepts.remove(c)   # подвинуть в конец (LRU)
                self.known_concepts.append(c)
        if len(self.known_concepts) > self.memory_size:
            self.known_concepts = self.known_concepts[-self.memory_size:]

    def _build_user_content(self, text):
        parts = []
        if self.known_concepts:
            parts.append(
                "[Концепты, уже добавленные в граф — если в новом фрагменте "
                "встречается ТОТ ЖЕ концепт, используй ТОЧНО это имя (для склейки), "
                "но НЕ выдумывай связи к ним без основания в тексте]:\n"
                + ", ".join(self.known_concepts)
            )
        if self.recent_text:
            parts.append(
                "[Предыдущий фрагмент — только для разрешения местоимений "
                "(it/this/they/the …) в реальные концепты; факты из него НЕ "
                "извлекай повторно]:\n" + self.recent_text
            )
        parts.append("[Извлеки триплеты ТОЛЬКО из этого нового фрагмента]:\n" + text)
        return "\n\n".join(parts)

    def feed(self, text):
        """Извлекает триплеты из очередного фрагмента с учётом памяти и
        обновляет память. Возвращает список triple-dict."""
        triples = _run_extraction(
            self.client, self.model, self._build_user_content(text), self.strict
        )
        self._remember(triples)
        self.recent_text = text[-self.keep_context_chars:]
        return triples


if __name__ == "__main__":
    import sys

    sample = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Force causes acceleration. Acceleration is a change in motion, "
        "measured in meters per second squared."
    )
    print(f"[*] model: {DEFAULT_MODEL}")
    print(f"[*] allowed relations: {ALLOWED_RELATIONS}\n")
    print(f"[*] extracting from: {sample!r}\n")
    for tr in extract_triples(sample):
        print(f"  ({tr['subject']}) --{tr['relation']}--> ({tr['object']})")
