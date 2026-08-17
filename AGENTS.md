# AGENTS.md

## Core Philosophy: YAGNI, Minimalism & Self-Explanatory Code
- **YAGNI Principle:** "You Aren't Gonna Need It." Never write speculative code, placeholder classes, or features for future use cases. Implement only what is required.
- **Readability counts:** A good one liner will be readable. If it's not readable, it's not a good one liner.
- **Self-explanatory code:** Prefer expressive names and straightforward code over comments and documentation.
- **Comments are exceptional:** Add a comment only when the reason behind the code cannot be made clear through the code itself.

## Coding Guidelines

### Project Tech Stack
- **Programming Language:** Python >=3.11, <3.12
- **Dependency Management:** UV
- **Task Runner:** Makefile - use project `make` targets for Python execution, formatting, linting, training, evaluation, and other defined workflows.

### Dependencies & Reproducibility
- Treat `pyproject.toml` as the source of truth for project dependencies and supported Python versions.
- Keep `uv.lock` synchronized with `pyproject.toml` and commit it to Git.
- Add, remove, and update dependencies using `uv`; never edit `uv.lock` manually.
- Do not add a dependency when the standard library or an existing dependency already solves the problem adequately.
- Pin or constrain dependencies when version differences can affect benchmark results.
- Do not upgrade unrelated dependencies while implementing a task.

### Always Do
- Run `make format` and `make lint` before marking any coding task as complete.
- Execute all Python tasks via the Makefile to ensure commands run under the correct `uv run` abstraction.
- Run deterministic data validations (e.g., shape check, null checks) before passing data to models.
- Add each new module to `config/logging.yaml` with its intended logging level. Before committing, set every module's configured level to `info`.
- Require each operational production module to accept a configured component logger and emit meaningful lifecycle messages. Package initializers, tests, immutable data types, and abstract interfaces are exempt.
- When a log entry and raised exception report the same failure, construct the message once and reuse it for both.
- Add concise docstrings to public classes and functions when their behavior or contract is not self-evident. Do not document obvious helpers or restate type annotations.

### Never Do
- Do not use global `pip` or standard `python` execution lines; always use the Makefile wrappers.
- Do not commit heavy data files such as `.csv`, `.parquet`, `.pkl`, or `.pt` model weights to Git.
- Do not use in-place mutations on dataframes; favor method chaining to prevent silent data contamination.
- Do not rewrite working code merely to match personal style preferences.
- Do not add comments that simply describe the following line of code.
- Do not add large docstrings to obvious functions.
- Do not create README sections, architecture documents, examples, or tutorials unless explicitly requested.
- Do not add tests, abstractions, logging, configuration, error handling, or validation for hypothetical scenarios unless required by the task.
- Do not modify unrelated files.

## Git Commit Guidelines
- Each commit should represent a single, self-contained change.
- Do not mix unrelated changes in a single commit.
- Follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification for commit messages.
  - feat: A new feature
  - fix: A bug fix
  - docs: Documentation only changes
  - style: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
  - refactor: A code change that neither fixes a bug nor adds a feature
  - test: Adding missing tests or correcting existing tests
- Never add a parantheses prefix to the commit type (e.g., "feat: ..." instead of "feat(): ...").
- Use the imperative mood in commit messages (e.g., "Add feature" instead of "Added feature").
- Keep commit messages concise and descriptive, ideally under 50 characters for the subject line.

## Documentation
- README.md should be concise, focusing on the purpose of the project, installation instructions, and usage examples.
- README.md should contain the current implemented project architecture.
- README.md should contain updated badges in the header section.
