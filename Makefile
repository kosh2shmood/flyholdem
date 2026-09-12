.PHONY: setup test demo replay record check-ui browser-check
setup:
	uv sync --frozen
	npm ci --prefix ui
check-ui:
	node --check ui/src/app.js
	node --check ui/src/fly-avatar.js
test: check-ui
	uv run --frozen pytest -q
demo: setup
	uv run --frozen flyholdem serve --port 8766
replay: setup
	uv run --frozen flyholdem serve --replay examples/fixture-demo.jsonl --port 8766
record:
	uv run --frozen flyholdem record --hands 6
browser-check:
	uv run --frozen python scripts/browser_check.py
fetch-malecns:
	uv run --frozen python -m flyholdem.connectome.registry
build-kernel:
	uv run --frozen python -m flyholdem.neural.kernel.build
test-oracle:
	PYTHONPATH=src uv run --frozen --project oracle pytest tests/numerical/test_brian_oracle.py -q
