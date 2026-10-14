# Contributing to SEAMS

Thanks for your interest in improving SEAMS. This project is actively developed by a small team, so please open an issue to discuss any non-trivial change before investing time in a pull request.

## Getting Set Up

Install [Rust](https://rustup.rs) 1.90 or newer. With the GNU toolchain you also need MinGW-w64 (`winget install BrechtSanders.WinLibs.POSIX.UCRT`); with the MSVC toolchain, the Visual Studio C++ build tools.

```bash
git clone https://github.com/Seasoned-In-Chaos/seamless-texture-maker.git
cd seamless-texture-maker

cargo run -p seams
```

See the [README](README.md#architecture) for how the workspace is laid out.

## Running Tests

```bash
cargo test --workspace
```

Add or update tests for any behaviour change. Tests live next to the code they cover, in a `tests` module at the bottom of each file. Most exercise the processing functions and the engine directly; none need a window.

## Before Submitting a Pull Request

1. Run the checks CI runs, and confirm all three pass:

   ```bash
   cargo fmt --all -- --check
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --workspace
   ```

2. If you changed a processing algorithm (seamless tiling, delighting, PBR map generation), verify the output visually. The tiled view on a real photo catches seam artifacts that synthetic test images don't. Unit tests alone are not sufficient for this class of change.
3. If you changed the interface, run the app and look at it. Colours, radii and type sizes come from `apps/seams/src/theme.rs`; don't hard-code them in widgets.
4. Non-test code must not panic: no `unwrap`, `expect`, `panic!` or unchecked indexing on values that come from files or settings. Return the crate's error type. `unsafe` is forbidden.
5. Respect the layering: `seams-core` has no file or UI code, and nothing below `apps/seams` may depend on egui.
6. Keep changes focused. A bug fix shouldn't bundle unrelated refactoring.
7. Update `CHANGELOG.md` under `[Unreleased]` for any user-facing change.

## Reporting Bugs

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md). Include your SEAMS version (Help, then About SEAMS), your Windows version, and the log file at `%APPDATA%\SeamlessTextureMaker\logs\seams.log` if the problem is a crash or a failed operation.

## Reporting Security Issues

Please do not open a public issue for a security vulnerability. Use GitHub's private vulnerability reporting (Security tab, then Report a vulnerability) instead.

## License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).
