# ***contributing***

Contributions are welcome — whether that's a bug fix, a new feature, or an improvement to the docs.

## *reporting bugs*

Open a [GitHub issue](https://github.com/itsluketwist/thinkpack/issues) with a clear description of the problem, the steps to reproduce it, and the expected vs actual behaviour.

## *suggesting features*

Open a [GitHub issue](https://github.com/itsluketwist/thinkpack/issues) describing what you'd like and why it would be useful.

## *submitting a pull request*

1. Fork the repository and create a branch from `main`.
2. Set up your development environment — see [DEVELOPMENT.md](DEVELOPMENT.md).
3. Make your changes.
4. Run `make lint` and `make test` — both must pass before submitting.
5. Open a pull request against `main` with a clear description of what changed and why.

## *code style*

- Use Python type hints throughout.
- Comments should be short and all lower-case — explain *why*, not *what*.
- Each non-empty file should have a one-line docstring at the top.
- Always include a trailing comma in multi-line function arguments and calls.
- Keep code readable — this project is read by non-expert reviewers.
