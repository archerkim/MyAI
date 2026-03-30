from multiprocessing.managers import BaseManager
import atexit
from brain_core_wrapper_local import BrainConnectionLocal # Импортируем наш класс

# --- Настройка Сервера ---
# 1. Создаем экземпляр "мозга", который будет общим
global_brain_instance = BrainConnectionLocal()

# 2. Создаем менеджер, который будет делиться этим экземпляром
class BrainManager(BaseManager):
    pass

# 3. Регистрируем наш класс "мозга" в менеджере под именем "get_brain"
BrainManager.register('get_brain',
                      callable=lambda: global_brain_instance,
                      exposed=('ingest_sentence', 'disconnect', 'connect',
                               'get_stats', 'get_node_id', 'set_activation',
                               'get_significant_nodes', 'get_lemma_by_id',
                               'spread_activation_step', 'get_subgraph_edges',
                               'reset_activations', 'is_connected'))

# 4. Настраиваем адрес и пароль для подключения
# (localhost означает "только на этом компьютере")
address = ('localhost', 5000)
authkey = b'1234' # Пароль для подключения

manager = BrainManager(address=address, authkey=authkey)
# --- Конец Настройки ---

@atexit.register
def cleanup_server():
    """
    Эта функция будет вызвана, когда сервер (этот скрипт)
    будет остановлен (включая Ctrl+C, если он не перехвачен).
    """
    print("\n[Сервер] Завершение работы...")
    if global_brain_instance and global_brain_instance.graph_ptr:
        global_brain_instance.disconnect()
    print("[Сервер] Мозг безопасно отключен.")

def run_server():
    """
    Основной цикл сервера.
    """
    try:
        # Подключаем "мозг" (загружаем ресурсы)
        global_brain_instance.connect()

        print(f"[Сервер] Мозг запущен и ожидает подключений по адресу {address}")

        # Запускаем сервер
        server = manager.get_server()
        server.serve_forever()

    except KeyboardInterrupt:
        # (Этот блок не обязателен, если вы используете atexit,
        # но он делает выход по Ctrl+C более чистым)
        print("\n[Сервер] Получен сигнал Ctrl+C...")
        # (atexit все равно сработает после этого)

if __name__ == "__main__":
    run_server()