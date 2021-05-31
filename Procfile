web: gunicorn app:server --timeout 500
celery_main_worker: celery -A tasks worker --beat -Q uw -l info --without-gossip --without-mingle --without-heartbeat