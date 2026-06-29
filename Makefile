
dev:
	uvicorn main:app --reload

run:
	fastapi dev main.py

init:
	pip-compile requirements.in && pip-compile requirements-dev.in && pip-sync requirements-dev.txt
