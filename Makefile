UV := uv
SRC_DIRS := open_llm_router
TEST_DIRS := tests
LINT_DIRS := $(SRC_DIRS) $(TEST_DIRS)

.PHONY: all lint format test tests clean

all: lint format test

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
	@$(UV) run python -m compileall -q $(LINT_DIRS)
	@$(UV) run ruff check .
	@$(UV) run flake8 $(LINT_DIRS)
	@$(UV) run isort --check-only $(LINT_DIRS)
	@$(UV) run mypy --strict $(SRC_DIRS)
	@$(UV) run pyright $(LINT_DIRS)

format:
	@echo "Running formatters..."
	@$(UV) run ruff format $(LINT_DIRS)
	@$(UV) run black $(LINT_DIRS)
	@$(UV) run isort --profile black $(LINT_DIRS)

test:
	@echo "Running tests..."
	@$(UV) run python -m compileall -q $(SRC_DIRS)
	@$(UV) run pytest -q $(TEST_DIRS)

tests: test

clean:
	@echo "Cleaning cache files..."
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} + \
		|| true
	@find . -type f -name "*.pyc" -delete || true
