# Contributing to Redis OpenAI Agents

Thank you for your interest in contributing to Redis OpenAI Agents! This document provides guidelines and instructions for contributing.

## Getting Started

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Redis 8 (for running tests)

### Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/redis/redis-openai-agents.git
   cd redis-openai-agents
   ```

2. Install dependencies:
   ```bash
   uv sync --all-extras --group dev
   ```

3. Start Redis:
   ```bash
   docker run -d --name redis -p 6379:6379 redis:8
   ```

4. Run tests to verify setup:
   ```bash
   uv run pytest --run-api-tests
   ```

## Development Workflow

### Code Style

We use [ruff](https://github.com/astral-sh/ruff) for formatting and linting, and [mypy](https://mypy-lang.org/) for type checking.

```bash
# Format code
make format

# Run linter
make lint

# Run type checker
make mypy

# Run all checks
make check
```

### Testing

All code changes must include tests. We use pytest with the following conventions:

- Unit tests go in `tests/unit/`
- Integration tests go in `tests/integration/`
- Always run tests with `--run-api-tests` flag

```bash
# Run all tests
uv run pytest --run-api-tests

# Run specific test file
uv run pytest tests/unit/test_session.py --run-api-tests

# Run with coverage
make test-cov
```

### Documentation

Documentation is built with Sphinx. To build and preview:

```bash
make docs
make docs-serve
```

## GitHub Actions Secrets

The CI/CD workflows require several repository secrets. Configure them at **Settings > Secrets and variables > Actions** in your GitHub repository.

| Secret | Required By | How to Obtain |
|--------|-------------|---------------|
| `OPENAI_API_KEY` | `test.yml` | Create an API key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys). Used to run tests marked with `@pytest.mark.requires_api_keys`. |
| `PYPI` | `release.yml` | Generate an API token at [pypi.org/manage/account/token](https://pypi.org/manage/account/token). Used to publish releases to PyPI on tagged pushes. |
| `GIST_TOKEN` | `coverage-gist.yml` | Create a GitHub Personal Access Token with the **`gist`** scope at [github.com/settings/tokens](https://github.com/settings/tokens). Used to update the coverage badge. |
| `GIST_ID` | `coverage-gist.yml` | Create a new **public** Gist at [gist.github.com](https://gist.github.com) with any placeholder file, then copy the Gist ID from the URL (e.g. `https://gist.github.com/<user>/<gist-id>`). The workflow writes `redis-openai-agents-coverage.json` into this Gist. |

> **Note:** `GITHUB_TOKEN` is provided automatically by GitHub Actions — no manual setup needed.

Once `GIST_TOKEN` and `GIST_ID` are configured, you can add a coverage badge to the README:

```markdown
![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/<USER>/<GIST_ID>/raw/redis-openai-agents-coverage.json)
```

## Pull Request Process

1. **Fork & Branch**: Fork the repository and create a feature branch from `main`.

2. **Make Changes**: Implement your changes following our code style guidelines.

3. **Test**: Ensure all tests pass and add new tests for new functionality.

4. **Document**: Update documentation if needed.

5. **Commit**: Use conventional commit messages:
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `test:` for test changes
   - `refactor:` for code refactoring
   - `chore:` for maintenance tasks

6. **Submit PR**: Open a pull request with a clear description of changes.

## Reporting Issues

When reporting issues, please include:

- Python version
- Redis version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs or error messages

## Code of Conduct

Please be respectful and constructive in all interactions. We're building this together.

## Questions?

- Open a [GitHub Issue](https://github.com/redis/redis-openai-agents/issues)
- Email: applied.ai@redis.com

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
