
dev:
	uvicorn main:app --reload

run:
	fastapi dev main.py

init:
	pip freeze > requirements.txt