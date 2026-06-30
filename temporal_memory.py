"""
temporal_memory.py — [research] би-темпоральная память фактов + update-семантика.

Закрывает главный пробел «памяти»: знание МЕНЯЕТСЯ во времени. Каждый факт несёт
два времени (как в Zep/Graphiti):
  • valid_from / valid_to — когда факт ИСТИНЕН в мире (время действительности);
  • recorded_at         — когда он БЫЛ УЗНАН (время транзакции).
Факт не удаляется, а «закрывается» (valid_to=now) — это soft-delete, сохраняющий
историю. Так система помнит, что было верно РАНЬШЕ, и что верно СЕЙЧАС.

Update-семантика:
  • assert_fact — добавить/подтвердить (повтор того же факта копит confidence —
    корроборация);
  • для ФУНКЦИОНАЛЬНЫХ отношений (одно значение: defined_as, located_in, …) новый
    объект ВЫТЕСНЯЕТ старый (старый закрывается) — паттерн «переехал в другой город»;
  • invalidate — «забыть» (закрыть факт);
  • update — заменить значение (закрыть текущее, открыть новое).

current(at) даёт срез знания на момент времени → его можно подать в граф.
Хранилище — SQLite (stdlib), запросы по времени.
"""

import sqlite3
import time

# Отношения с ЕДИНСТВЕННЫМ актуальным значением: новый объект вытесняет старый.
FUNCTIONAL_DEFAULT = {"defined_as", "measured_in", "located_in", "has_value",
                      "born_in", "capital_of"}


class TemporalStore:
    def __init__(self, path=":memory:", functional=None):
        self.db = sqlite3.connect(path)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS facts(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT, relation TEXT, object TEXT,
                valid_from REAL, valid_to REAL,      -- время действительности
                recorded_at REAL, confidence REAL)""")
        self.db.commit()
        self.functional = set(FUNCTIONAL_DEFAULT if functional is None else functional)

    def _now(self):
        return time.time()

    def assert_fact(self, s, r, o, valid_from=None, confidence=1.0):
        """Добавляет факт. Повтор того же актуального факта копит confidence
        (корроборация). Для функциональных отношений новый объект вытесняет старый."""
        now = self._now()
        vf = now if valid_from is None else valid_from
        cur = self.db.execute(
            "SELECT id, confidence FROM facts WHERE subject=? AND relation=? AND "
            "object=? AND valid_to IS NULL", (s, r, o)).fetchone()
        if cur:                                            # уже актуален → корроборация
            self.db.execute("UPDATE facts SET confidence=? WHERE id=?",
                            (cur[1] + confidence, cur[0]))
            self.db.commit()
            return cur[0]
        if r in self.functional:                           # вытеснить старое значение
            # старый факт действителен ДО начала нового (vf), а не до now:
            # «жил в Париже [t1..t2], переехал в Лондон [t2..]»
            self.db.execute(
                "UPDATE facts SET valid_to=? WHERE subject=? AND relation=? AND "
                "valid_to IS NULL AND object!=?", (vf, s, r, o))
        self.db.execute(
            "INSERT INTO facts(subject,relation,object,valid_from,valid_to,"
            "recorded_at,confidence) VALUES(?,?,?,?,?,?,?)",
            (s, r, o, vf, None, now, confidence))
        self.db.commit()
        return self.db.execute("SELECT last_insert_rowid()").fetchone()[0]

    def invalidate(self, s, r, o=None, at=None):
        """«Забыть»: закрыть факт(ы) (valid_to=at). o=None — все (s,r,*)."""
        at = self._now() if at is None else at
        if o is None:
            self.db.execute("UPDATE facts SET valid_to=? WHERE subject=? AND "
                            "relation=? AND valid_to IS NULL", (at, s, r))
        else:
            self.db.execute("UPDATE facts SET valid_to=? WHERE subject=? AND "
                            "relation=? AND object=? AND valid_to IS NULL", (at, s, r, o))
        self.db.commit()

    def update(self, s, r, o_new, valid_from=None, confidence=1.0):
        """Заменить значение: закрыть все текущие (s,r,*), открыть новое."""
        self.invalidate(s, r)
        return self.assert_fact(s, r, o_new, valid_from, confidence)

    def current(self, at=None):
        """Срез актуальных фактов на момент at (по умолчанию — сейчас)."""
        at = self._now() if at is None else at
        rows = self.db.execute(
            "SELECT subject,relation,object FROM facts WHERE valid_from<=? AND "
            "(valid_to IS NULL OR valid_to>?)", (at, at)).fetchall()
        return [tuple(r) for r in rows]

    def history(self, subject=None, relation=None):
        """Полная история (вкл. закрытые факты), упорядоченная по времени узнавания."""
        q = ("SELECT subject,relation,object,valid_from,valid_to,recorded_at,"
             "confidence FROM facts")
        cond, args = [], []
        if subject:
            cond.append("subject=?"); args.append(subject)
        if relation:
            cond.append("relation=?"); args.append(relation)
        if cond:
            q += " WHERE " + " AND ".join(cond)
        return self.db.execute(q + " ORDER BY recorded_at, id", args).fetchall()

    def close(self):
        self.db.close()
