web: gunicorn app:server --timeout 500
celery_main_worker: celery -A tasks worker --loglevel=info