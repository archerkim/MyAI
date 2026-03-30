import argparse
import sys
from pathlib import Path
import shutil
import time

from config import DROPBOX_DIR
from query_engine import QueryEngine
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
        print(f"[Клиент cli.py] Успешно подключен к серверу.")

        brain_proxy = manager.get_brain()
        return brain_proxy
    except ConnectionRefusedError:
        print("\n[Клиент] ОШИБКА: Не удалось подключиться. Сервер 'мозга' не запущен или адрес неверен.")
        return None
def handle_ingest(args):
    source_path = Path(args.path).resolve()
    if not source_path.exists():
        print(f"[error] source path does not exist: {source_path}", file=sys.stderr)
        sys.exit(1)
    DROPBOX_DIR.mkdir(parents=True, exist_ok=True)
    files_to_copy = [source_path] if source_path.is_file() else [f for f in source_path.iterdir() if f.is_file()]
    if not files_to_copy:
        print(f"[info] no files to ingest at: {source_path}")
        return
    print(f"[*] queuing {len(files_to_copy)} file(s) for ingestion...")
    for file in files_to_copy:
        try:
            shutil.copy(file, DROPBOX_DIR)
            print(f"  - '{file.name}' -> queued.")
        except Exception as e:
            print(f"  - failed to queue '{file.name}': {e}", file=sys.stderr)
    print("\n[*] command finished. ensure system services (watcher, workers) are running.")

def handle_query(args):
    """
    обработчик для команды 'query'.
    создает соединение с мозгом, передает его в QueryEngine и выполняет запрос.
    """
    query_text = args.text
    print(f"[*] establishing connection to brain...")

    try:
        brain = get_brain()

        engine = QueryEngine(brain)

        start_time = time.time()
        answer = engine.run_query(query_text)
        end_time = time.time()

        print("\n" + "="*20 + " final answer " + "="*20)
        print(answer)
        print("="*54)
        print(f"(query processed in {end_time - start_time:.2f} seconds)")

    except Exception as e:
        print(f"\n[error] failed to execute query: {e}", file=sys.stderr)
        sys.exit(1)

def handle_stats(args):
    """
    обработчик для команды 'stats'.
    использует `with` для безопасного подключения и получения статистики.
    """
    print("[*] establishing connection to brain for stats...")
    try:
        brain = get_brain()

        stats = brain.get_stats()
        print("\n--- brain statistics ---")
        print(f"  - total nodes (concepts): {stats['node_count']}")
        print(f"  - total edges (relations): {stats['total_edge_count']}")
        print("------------------------")
    except Exception as e:
        print(f"\n[error] failed to get stats: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="gsr brain command-line interface.")
    # ... (код парсера без изменений)
    subparsers = parser.add_subparsers(dest="command", required=True)
    # ingest
    parser_ingest = subparsers.add_parser("ingest", help="queue a file/directory for ingestion.")
    parser_ingest.add_argument("path", type=str)
    parser_ingest.set_defaults(func=handle_ingest)
    # query
    parser_query = subparsers.add_parser("query", help="ask a question to the brain.")
    parser_query.add_argument("text", type=str)
    parser_query.set_defaults(func=handle_query)
    # stats
    parser_stats = subparsers.add_parser("stats", help="get statistics about the brain.")
    parser_stats.set_defaults(func=handle_stats)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()