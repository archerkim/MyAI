from brain_core_wrapper_local import DEP_TYPE_MAP

_INT_TO_DEP_LABEL = {v: k for k, v in DEP_TYPE_MAP.items()}

def dep_type_to_string(dep_type_int):
    """конвертирует целочисленный enum DependencyType в его строковое представление."""
    return _INT_TO_DEP_LABEL.get(dep_type_int, "UNKNOWN")
def format_subgraph_for_prompt(nodes, edges):
    """превращает подграф в текстовое представление для llm."""
    prompt_part = "key concepts and their relations:\n"

    node_map = {node['id']: node['lemma'] for node in nodes}

    if not edges:
        return "key concepts: " + ", ".join(node_map.values())

    for edge in edges:
        from_lemma = node_map.get(edge['from_id'], 'unknown')
        to_lemma = node_map.get(edge['to_id'], 'unknown')
        # dep_type_to_string - нужна функция-конвертер
        dep_str = dep_type_to_string(edge['dep_type'])

        prompt_part += f"- '{from_lemma}' --({dep_str}, strength:{edge['conductance']:.2f})--> '{to_lemma}'\n"

    return prompt_part

def verbalize_answer(query, nodes, edges, intent):
    if not nodes:
        return "i could not find relevant information."

    context_str = format_subgraph_for_prompt(nodes, edges)

    prompt = f"based on the following structured knowledge, answer the user's query.\n" \
             f"do not use any external knowledge.\n\n" \
             f"=== knowledge graph context ===\n" \
             f"{context_str}\n" \
             f"=== user query ===\n" \
             f"'{query}'\n\n" \
             f"answer:"

    return prompt