PYTHON ?= python

.PHONY: install install-embed fetch eval search test lint

install:
	$(PYTHON) -m pip install -q -e ".[dev]"

install-embed:
	$(PYTHON) -m pip install -q -e ".[dev,embed]"

# re-download the three acts from fedlex and rewrite data/articles.jsonl (needs network)
fetch:
	$(PYTHON) -m fedlex_rag.cli fetch

# no key, no network, no model: the committed vector cache covers the corpus and the questions
eval:
	$(PYTHON) -m fedlex_rag.cli eval

# a free question needs the model: make install-embed first
search:
	$(PYTHON) -m fedlex_rag.cli search "$(Q)"

test:
	$(PYTHON) -m pytest --cov --cov-report=term-missing:skip-covered

lint:
	$(PYTHON) -m ruff check .
