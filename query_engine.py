import spacy
from intent_classifier import classify_intent
from verbalizer import verbalize_answer
from multiprocessing.managers import BaseManager

class QueryEngine:
    def __init__(self, brain_connection, spacy_model_name="en_core_web_trf"):
        """
        инициализирует движок с подключением к мозгу и моделью spaCy.
        """
        if not brain_connection or not brain_connection.is_connected():
            raise ValueError("queryengine requires an active brain connection.")
        self.brain = brain_connection
        try:
            self.nlp = spacy.load(spacy_model_name)
        except OSError:
            print(f"could not load spacy model '{spacy_model_name}'. downloading...")
            spacy.cli.download(spacy_model_name)
            self.nlp = spacy.load(spacy_model_name)

        print("[query_engine] initialized.")

    def _get_anchor_nodes(self, query_text):
        """парсит запрос и находит id узлов-якорей в графе."""
        doc = self.nlp(query_text)
        anchor_ids = []
        for token in doc:
            if not token.is_stop and not token.is_punct and token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ']:
                node_id = self.brain.get_node_id(token.lemma_)
                if node_id != -1:
                    anchor_ids.append(node_id)
        return list(set(anchor_ids))

    def run_query(self, query_text, spread_steps=3):
        """
        основной метод для выполнения полного цикла "мысли".
        """
        print(f"\n[*] received query: '{query_text}'")

        # --- шаг 1: определяем интент ---
        intent = classify_intent(query_text)
        print(f"[*] detected intent: {intent}")

        # --- шаг 2: находим узлы-якоря ---
        anchor_ids = self._get_anchor_nodes(query_text)
        if not anchor_ids:
            print("[!] no anchor nodes found in graph for this query.")
            return verbalize_answer(query_text, [], [], intent) # вызываем вербализатор с пустым контекстом

        print(f"[*] found {len(anchor_ids)} anchor nodes: {anchor_ids}")

        # --- шаг 3: выполняем "мышление" через wrapper ---

        # 3a. сбрасываем предыдущие "мысли"
        self.brain.reset_activations()

        # 3b. устанавливаем начальную активацию
        for node_id in anchor_ids:
            self.brain.set_activation(node_id, 1.0)

        # 3c. запускаем распространение активации
        print(f"[*] spreading activation for {spread_steps} steps...")
        for step in range(spread_steps):
            self.brain.spread_activation_step(anchor_ids)
            print(f"  - step {step+1} complete.")

        # 3d. получаем "значимые" узлы
        significant_nodes = self.brain.get_significant_nodes()
        if not significant_nodes:
            print("[!] spread activation did not yield any significant nodes.")
            return verbalize_answer(query_text, [], [], intent)

        print(f"[*] found {len(significant_nodes)} significant nodes.")

        # 3e. получаем связи между ними, чтобы построить подграф

        significant_ids = [node['id'] for node in significant_nodes]
        subgraph_edges = self.brain.get_subgraph_edges(significant_ids)

        print(f"[*] reconstructed subgraph with {len(subgraph_edges)} edges.")

        # --- шаг 4: вербализация ответа ---
        print("[*] sending context to verbalizer...")
        final_answer = verbalize_answer(query_text, significant_nodes, subgraph_edges, intent)

        return final_answer

def get_brain():
    """Выполняет 3 шага для подключения и возвращает прокси-объект мозга."""
    try:
        class BrainManager(BaseManager):
            pass
        BrainManager.register('get_brain')
        ADDRESS = ('localhost', 5000)
        AUTH_KEY = b'1234'

        manager = BrainManager(address=ADDRESS, authkey=AUTH_KEY)

        manager.connect()
        print(f"[Клиент query_engine.py] Успешно подключен к серверу.")

        brain_proxy = manager.get_brain()
        return brain_proxy
    except ConnectionRefusedError:
        print("\n[Клиент] ОШИБКА: Не удалось подключиться. Сервер 'мозга' не запущен или адрес неверен.")
        return None
# --- пример использования ---
if __name__ == '__main__':
    try:
        # используем `with`, чтобы гарантировать `connect` и `disconnect`
        brain = get_brain()

        # --- этап наполнения мозга (для теста) ---
        print("\n--- ingesting sample data ---")
        sample_sentence_1 = [
            {'lemma': 'мальчик', 'dep': 'nsubj', 'head_offset': 1},
            {'lemma': 'спать', 'dep': 'ROOT', 'head_offset': 0}
        ]
        sample_sentence_2 = [
            {'lemma': 'уставший', 'dep': 'amod', 'head_offset': 1},
            {'lemma': 'мальчик', 'dep': 'ROOT', 'head_offset': 0}
        ]
        brain.ingest_sentence(sample_sentence_1)
        brain.ingest_sentence(sample_sentence_2)
        stats = brain.get_stats()
        print(f"brain now has {stats['node_count']} nodes and {stats['total_edge_count']} edges.")
        print("-----------------------------\n")

        # --- этап запроса ---
        engine = QueryEngine(brain, spacy_model_name='ru_core_news_sm')

        # выполняем тестовый запрос
        # для этого запроса якорем будет "мальчик"
        # активация должна "перетечь" на "уставший" и "спать"
        answer = engine.run_query("что делал уставший мальчик?")

        print("\n--- final answer from verbalizer ---")
        print(answer)

    except (RuntimeError, FileNotFoundError, ConnectionError, ValueError) as e:
        print(f"\n[!!!] an error occurred during query engine test: {e}")