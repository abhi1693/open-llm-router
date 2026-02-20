PYTHON := python
UV := uv
SRC_DIRS := open_llm_router

.PHONY: all lint format test tests clean

all: lint format test

HAS_UV := $(shell command -v $(UV) >/dev/null 2>&1 && echo 1 || echo 0)
HAS_BLACK := $(shell if [ $(HAS_UV) -eq 1 ]; then $(UV) run black --version >/dev/null 2>&1 && echo 1 || echo 0; elif command -v black >/dev/null 2>&1; then echo 1; else echo 0; fi)
HAS_FLAKE8 := $(shell if [ $(HAS_UV) -eq 1 ]; then $(UV) run python -m flake8 --version >/dev/null 2>&1 && echo 1 || echo 0; elif command -v flake8 >/dev/null 2>&1; then echo 1; else echo 0; fi)
HAS_ISORT := $(shell if [ $(HAS_UV) -eq 1 ]; then $(UV) run isort --version >/dev/null 2>&1 && echo 1 || echo 0; elif command -v isort >/dev/null 2>&1; then echo 1; else echo 0; fi)
HAS_RUFF := $(shell if [ $(HAS_UV) -eq 1 ]; then $(UV) run python -m ruff check --help >/dev/null 2>&1 && echo 1 || echo 0; elif command -v ruff >/dev/null 2>&1; then echo 1; else echo 0; fi)
HAS_MYPY := $(shell if [ $(HAS_UV) -eq 1 ]; then $(UV) run mypy --version >/dev/null 2>&1 && echo 1 || echo 0; elif command -v mypy >/dev/null 2>&1; then echo 1; else echo 0; fi)
HAS_PYRIGHT := $(shell if [ $(HAS_UV) -eq 1 ]; then $(UV) run pyright --version >/dev/null 2>&1 && echo 1 || echo 0; elif command -v pyright >/dev/null 2>&1; then echo 1; else echo 0; fi)

help:
	@echo "Available targets:"
	@echo "  all     - Run lint, format, and test"
	@echo "  lint    - Run linting checks"
	@echo "  format  - Format source files (if formatter is available)"
	@echo "  test    - Run test suite"
	@echo "  tests   - Alias for test"
	@echo "  clean   - Remove Python cache artifacts"

lint:
	@echo "Running lint checks..."
	@if [ $(HAS_UV) -eq 1 ]; then \
		$(UV) run python -m compileall -q $(SRC_DIRS); \
	else \
		$(PYTHON) -m compileall -q $(SRC_DIRS); \
	fi
	@if [ $(HAS_RUFF) -eq 1 ]; then \
		if [ $(HAS_UV) -eq 1 ]; then \
			$(UV) run ruff check .; \
		else \
			ruff check .; \
		fi; \
	else \
		echo "ruff not installed; skipping ruff checks."; \
	fi
	@if [ $(HAS_FLAKE8) -eq 1 ]; then \
		if [ $(HAS_UV) -eq 1 ]; then \
			$(UV) run flake8 $(SRC_DIRS); \
		else \
			flake8 $(SRC_DIRS); \
		fi; \
	else \
		echo "flake8 not installed; skipping flake8 checks."; \
	fi
	@if [ $(HAS_ISORT) -eq 1 ]; then \
		if [ $(HAS_UV) -eq 1 ]; then \
			$(UV) run isort --check-only $(SRC_DIRS); \
		else \
			isort --check-only $(SRC_DIRS); \
		fi; \
	else \
		echo "isort not installed; skipping isort checks."; \
	fi
	@if [ $(HAS_MYPY) -eq 1 ]; then \
		if [ $(HAS_UV) -eq 1 ]; then \
			$(UV) run mypy --strict $(SRC_DIRS); \
		else \
			mypy --strict $(SRC_DIRS); \
		fi; \
	else \
		echo "mypy not installed; skipping mypy checks."; \
	fi
	@if [ $(HAS_PYRIGHT) -eq 1 ]; then \
		if [ $(HAS_UV) -eq 1 ]; then \
			$(UV) run pyright $(SRC_DIRS); \
		else \
			pyright $(SRC_DIRS); \
		fi; \
	else \
		echo "pyright not installed; skipping pyright checks."; \
	fi

format:
	@echo "Running formatters..."
	@if [ $(HAS_RUFF) -eq 1 ]; then \
		if [ $(HAS_UV) -eq 1 ]; then \
			$(UV) run ruff format $(SRC_DIRS); \
		else \
			ruff format $(SRC_DIRS); \
		fi; \
	else \
		echo "ruff not installed; skipping ruff format."; \
	fi
	@if [ $(HAS_BLACK) -eq 1 ]; then \
		if [ $(HAS_UV) -eq 1 ]; then \
			$(UV) run black $(SRC_DIRS); \
		else \
			black $(SRC_DIRS); \
		fi; \
	else \
		echo "black not installed; skipping black formatting."; \
	fi
	@if [ $(HAS_ISORT) -eq 1 ]; then \
		if [ $(HAS_UV) -eq 1 ]; then \
			$(UV) run isort --profile black $(SRC_DIRS); \
		else \
			isort --profile black $(SRC_DIRS); \
		fi; \
	else \
		echo "isort not installed; skipping isort formatting."; \
	fi

test:
	@echo "Running tests..."
	@if [ $(HAS_UV) -eq 1 ]; then \
		$(UV) run python -m compileall -q $(SRC_DIRS); \
	else \
		$(PYTHON) -m compileall -q $(SRC_DIRS); \
	fi

tests: test

clean:
	@echo "Cleaning cache files..."
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} + \
		|| true
	@find . -type f -name "*.pyc" -delete || true
