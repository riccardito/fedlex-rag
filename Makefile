PYTHON ?= python

.PHONY: install install-embed install-llm fetch eval eval-search record ask search test lint

install:
	$(PYTHON) -m pip install -q -e ".[dev]"

install-embed:
	$(PYTHON) -m pip install -q -e ".[dev,embed]"

# for --live: the Anthropic client; the key comes from ANTHROPIC_API_KEY or the OS keyring
install-llm:
	$(PYTHON) -m pip install -q -e ".[dev,embed,llm]"

# re-download the three acts from fedlex and rewrite data/articles.jsonl (needs network)
fetch:
	$(PYTHON) -m fedlex_rag.cli fetch

# no key, no network, no model: the committed vectors and the recorded Claude answers replay
eval:
	$(PYTHON) -m fedlex_rag.cli eval

eval-search:
	$(PYTHON) -m fedlex_rag.cli eval --search-only

# record the Claude answers that are missing (needs a key: make install-llm)
record:
	$(PYTHON) -m fedlex_rag.cli eval --live

# one question end to end: make ask Q="..." (a new question needs --live)
ask:
	$(PYTHON) -m fedlex_rag.cli ask "$(Q)" $(LIVE)

# a free question needs the model: make install-embed first
search:
	$(PYTHON) -m fedlex_rag.cli search "$(Q)"

test:
	$(PYTHON) -m pytest --cov --cov-report=term-missing:skip-covered

lint:
	$(PYTHON) -m ruff check .
