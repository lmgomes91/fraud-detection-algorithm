clean:
	@find . -name "__pycache__" -type d | xargs rm -rf
	@rm -f coverage
	@rm -rf build
	@rm -rf dist
	@rm -rf .pytest_cache

run: clean
	python main.py
	@make clean

run-kohonen: clean
	python main.py --method kohonen
	@make clean

run-lstm: clean
	python main.py --method lstm
	@make clean

run-mlp: clean
	python main.py --method mlp
	@make clean

run-rbf: clean
	python main.py --method rbf
	@make clean

run-rnn: clean
	python main.py --method rnn
	@make clean