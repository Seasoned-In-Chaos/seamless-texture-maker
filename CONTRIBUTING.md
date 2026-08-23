# Contributing to SEAMS

Thanks for your interest in improving SEAMS. This project is actively developed by a small team, so please open an issue to discuss any non-trivial change before investing time in a pull request.

## Getting Set Up

```bash
git clone https://github.com/Seasoned-In-Chaos/seamless-texture-maker.git
cd seamless-texture-maker

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements-dev.txt

python main.py
```

See [Getting Started](docs/pages/getting-started.mdx) for full system requirements and installer-build instructions.

## Running Tests

```bash
python -m pytest -q
```

Please add or update tests for any behavior change under `tests/`. Look at the existing files for the project's conventions — most tests exercise `SeamlessProcessor` and the `app.core` modules directly rather than mocking internals.

## Before Submitting a Pull Request

1. Run the full test suite and confirm it passes.
2. If you changed a processing algorithm (seamless tiling, delighting, PBR map generation), verify the output visually — a 2x2 tiled render on a real photo catches seam artifacts that synthetic test images don't. Unit tests alone are not sufficient for this class of change.
3. Keep changes focused. A bug fix shouldn't bundle unrelated refactoring — it makes review harder and obscures what actually changed.
4. Match the existing code style (no enforced linter yet — read nearby code and follow its conventions).
5. Update `CHANGELOG.md` under `[Unreleased]` for any user-facing change.

## Reporting Bugs

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md). Include your SEAMS version (Help → About, or the title bar), Windows version, and whether you're running the installed build or from source — many issues are specific to one or the other.

## Reporting Security Issues

Please do not open a public issue for a security vulnerability. Use GitHub's private vulnerability reporting (Security tab → Report a vulnerability) instead.

## License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
