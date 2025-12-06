web: cd backend && pip install -r requirements-production.txt && gunicorn app:app --workers 1 --threads 2 --timeout 120 --max-requests 1000 --max-requests-jitter 50
