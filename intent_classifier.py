INTENT_TRIGGERS = {
    "causal": ["почему", "причина", "следствие"],
    "procedural": ["как", "способ", "инструкция"],
    "definitional": ["что такое", "кто такой", "определение"],
    "factual": ["кто", "что", "где", "когда", "какой"]
}

def classify_intent(query_text):
    """
    простой классификатор на основе ключевых слов-триггеров.
    """
    query_text_lower = query_text.lower()

    for intent, triggers in INTENT_TRIGGERS.items():
        for trigger in triggers:
            if query_text_lower.startswith(trigger):
                return intent

    # если ничего не подошло, возвращаем "общий" тип
    return "definitional"