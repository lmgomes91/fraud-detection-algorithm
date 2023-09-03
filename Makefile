clean:
	@find . -name "__pycache__" -type d | xargs rm -rf
	@rm -f coverage
	@rm -rf build
	@rm -rf dist
	@rm -rf .pytest_cache

run: clean
	python main.py
	@make clean