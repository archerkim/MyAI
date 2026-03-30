import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from tasks import parse_document_task
from config import DROPBOX_DIR

class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            print(f"[*] new file detected: {event.src_path}")
            parse_document_task.delay(event.src_path)

def start_watching():
    observer = Observer()
    event_handler = NewFileHandler()
    observer.schedule(event_handler, path=DROPBOX_DIR, recursive=False)

    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()

if __name__ == "__main__":
    start_watching()