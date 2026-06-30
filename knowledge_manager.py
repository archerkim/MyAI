"""
knowledge_manager.py — самоорганизующиеся модули знаний + выборочная активация.

Реализует ядро автономного ученика:
  • TOPIC ROUTER — классифицирует материал по теме и САМ создаёт отдельный модуль
    для новой темы (математика → math.brain), складывая туда факты.
  • MODULE MANAGER — держит активными только НУЖНЫЕ модули: задача по химии →
    core(язык)+chemistry(+пререквизиты), music ВЫКЛЮЧЕН. Экономит «мышление».
  • СОСТОЯНИЕ — главный контроллер меряет словарь (vocabulary), эрудицию
    (breadth тем) и понимание (graph_quality активного графа).

Каждая тема — отдельный .brain-файл со своим module_id (провенанс), монтируется
по требованию. Язык (синонимы/база) — это core-модуль, активен всегда.
"""

import json
import os

from brain_core_wrapper_local import BrainConnectionLocal

CORE_TOPIC = "language"      # базовый языковой/ядровой модуль — всегда активен


class ModuleRegistry:
    """Реестр тема → {module_id, файл, размер}. Персистится в JSON."""

    def __init__(self, path):
        self.path = path
        self.topics = {}
        self._next_id = 1
        if os.path.exists(path):
            self.load()

    def ensure(self, topic):
        if topic not in self.topics:
            self.topics[topic] = {"module_id": self._next_id, "concepts": 0}
            self._next_id += 1
            self.save()
        return self.topics[topic]["module_id"]

    def set_size(self, topic, concepts):
        self.topics[topic]["concepts"] = concepts
        self.save()

    def known(self):
        return list(self.topics)

    def save(self):
        json.dump({"topics": self.topics, "next_id": self._next_id},
                  open(self.path, "w"), ensure_ascii=False, indent=2)

    def load(self):
        d = json.load(open(self.path))
        self.topics = d["topics"]; self._next_id = d["next_id"]


def classify_topic(text, known_topics, model="qwen2.5:7b"):
    """LLM определяет ПРЕДМЕТ материала. Может вернуть существующую тему из
    known_topics или предложить НОВУЮ (тогда менеджер создаст модуль)."""
    import ollama
    prompt = (
        "Определи ПРЕДМЕТ/тему этого учебного текста одним словом-меткой "
        "(snake_case, англ.): mathematics, physics, chemistry, biology, music, "
        "history, ... Если подходит одна из уже известных тем — верни ровно её.\n"
        f"Известные темы: {known_topics or '—'}\n\nТекст: {text[:800]}"
    )
    schema = {"type": "object", "properties": {"topic": {"type": "string"}},
              "required": ["topic"]}
    r = ollama.Client().chat(model=model, options={"temperature": 0}, format=schema,
                             messages=[{"role": "user", "content": prompt}])
    return json.loads(r.message.content)["topic"].strip().lower()


def relevant_topics(task, known_topics, model="qwen2.5:7b"):
    """LLM выбирает, какие темы НУЖНЫ для задачи (остальные модули не монтируем)."""
    import ollama
    prompt = (
        "Какие из известных тем НУЖНЫ, чтобы работать над этой задачей? Верни "
        "только релевантные (включая пререквизиты, напр. химии нужна физика). "
        "Музыку для химии НЕ включай.\n"
        f"Известные темы: {known_topics}\n\nЗадача: {task}"
    )
    schema = {"type": "object", "properties": {"topics": {"type": "array",
              "items": {"type": "string"}}}, "required": ["topics"]}
    r = ollama.Client().chat(model=model, options={"temperature": 0}, format=schema,
                             messages=[{"role": "user", "content": prompt}])
    out = [t.strip().lower() for t in json.loads(r.message.content)["topics"]]
    return [t for t in out if t in known_topics]


class Learner:
    """Главный контроллер: маршрутизация знаний по модулям, выборочная активация,
    измерение состояния."""

    def __init__(self, workdir):
        os.makedirs(workdir, exist_ok=True)
        self.workdir = workdir
        self.registry = ModuleRegistry(os.path.join(workdir, "registry.json"))
        self._active = None
        self._active_topics = []

    def _file(self, topic):
        return os.path.join(self.workdir, f"{topic}.brain")

    # --- TOPIC ROUTER: факты → модуль темы (создаёт при необходимости) ---
    def learn_facts(self, topic, triples):
        """Складывает триплеты в .brain-модуль темы (создаёт его, если новый)."""
        mid = self.registry.ensure(topic)
        fpath = self._file(topic)
        b = BrainConnectionLocal(); b.connect()
        try:
            if os.path.exists(fpath):
                b.load(fpath)
            b.set_module_id(mid)
            for (s, r, o) in triples:
                b.add_triple(s, r, o)
            b.save(fpath)
            stats = b.get_stats()
        finally:
            b.disconnect()
        self.registry.set_size(topic, stats["node_count"])
        return topic, mid, stats

    def auto_learn(self, text, triples, model="qwen2.5:7b"):
        """Сам классифицирует тему и маршрутизирует факты в её модуль."""
        topic = classify_topic(text, self.registry.known(), model)
        return self.learn_facts(topic, triples)

    # --- MODULE MANAGER: монтируем только нужные темы (+ язык) ---
    def activate(self, topics):
        """Поднимает свежий активный мозг: core(язык) + перечисленные темы.
        Остальные модули НЕ монтируются (экономия). Возвращает список активных."""
        if self._active:
            self._active.disconnect()
        self._active = BrainConnectionLocal(); self._active.connect()
        active = []
        for t in [CORE_TOPIC] + [x for x in topics if x != CORE_TOPIC]:
            fpath = self._file(t)
            if t in self.registry.topics and os.path.exists(fpath):
                self._active.mount_module(fpath, self.registry.topics[t]["module_id"])
                active.append(t)
        self._active_topics = active
        return active

    def activate_for(self, task, model="qwen2.5:7b"):
        """Сам выбирает нужные темы для задачи и активирует их."""
        topics = relevant_topics(task, self.registry.known(), model)
        return self.activate(topics)

    def active_brain(self):
        return self._active

    # --- СОСТОЯНИЕ: словарь / эрудиция / понимание ---
    def state(self):
        from predictor import facts_from_brain
        from graph_quality import evaluate
        vocab = set()
        all_facts = set()
        per_module = {}
        for topic in self.registry.topics:
            fpath = self._file(topic)
            if not os.path.exists(fpath):
                continue
            b = BrainConnectionLocal(); b.connect(); b.load(fpath)
            facts = facts_from_brain(b)
            all_facts |= facts
            concepts = {x for (s, _, o) in facts for x in (s, o)}
            vocab |= concepts
            per_module[topic] = len(concepts)
            b.disconnect()
        # понимание = качество ВСЕГО накопленного знания (graph_quality)
        understanding = round(evaluate(list(all_facts))["composite"], 3) if all_facts else None
        # понимание активного среза (что сейчас в «рабочей памяти»)
        active_understanding = None
        if self._active:
            af = facts_from_brain(self._active)
            if af:
                active_understanding = round(evaluate(list(af))["composite"], 3)
        return {
            "vocabulary": len(vocab),                 # всего изученных концептов
            "erudition": len(per_module),             # широта: число тем-модулей
            "understanding": understanding,           # качество всего знания
            "active_understanding": active_understanding,
            "modules": per_module,
            "active": self._active_topics,
        }
