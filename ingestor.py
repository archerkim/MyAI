from celery_app import app
from multiprocessing.managers import BaseManager

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
        print(f"[Клиент ingestor.py] Успешно подключен к серверу.")

        brain_proxy = manager.get_brain()
        return brain_proxy
    except ConnectionRefusedError:
        print("\n[Клиент] ОШИБКА: Не удалось подключиться. Сервер 'мозга' не запущен или адрес неверен.")
        return None
@app.task
def ingest_data_task(processed_data):
    """
    принимает структурированные данные и сохраняет их в ОБЩИЙ граф.
    """
    try:
        # получаем доступ к единственному экземпляру мозга
        brain = get_brain()

        for element in processed_data:
            if element['type'] == 'text':
                sentence_content = element['content']
                brain.ingest_sentence(sentence_content['tokens'])
            # ... остальная логика ...

        # мы больше не вызываем connect/disconnect здесь!
        # мозг живет сам по себе.

        return {"status": "success", "elements_ingested": len(processed_data)}

    except Exception as e:
        print(f"[!] error during ingestion: {e}")
        # celery сам обработает retry
        # raise self.retry(exc=e, countdown=60) # раскомментировать, если нужно