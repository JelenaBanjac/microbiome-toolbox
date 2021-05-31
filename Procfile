web: gunicorn app:server --timeout 500
celeryd: celery -A tasks worker -E -B --loglevel=INFO