from celery import Celery
from config import REDIS_URL

# 1. define the main celery application instance
app = Celery(
    'gsr_brain_tasks', # give your app a name
    broker=REDIS_URL,
    # 2. tell celery which modules contain your tasks
    include=['tasks', 'ingestor']
)

# optional: configure other celery settings here if needed
app.conf.update(
    task_serializer='json',
    accept_content=['json'],  # ensure tasks use json
    result_serializer='json',
    timezone='utc',
    enable_utc=True,
)

if __name__ == '__main__':
    app.start()