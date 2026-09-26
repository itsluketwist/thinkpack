# ***development***

## *setup*

Clone the repository:

```shell
git clone https://github.com/itsluketwist/thinkpack.git
```

We use [`uv`](https://docs.astral.sh/uv/) for project management.
Once cloned, create a virtual environment with the dev dependencies, and activate it:

```shell
pip install uv  # if not already installed

uv sync

. .venv/bin/activate
```

Install the pre-commit hooks (run once after cloning):

```shell
pre-commit install
```

## *commands*

| Command | Description |
|---|---|
| `make lint` | Run pre-commit on all files |
| `make test` | Run the test suite |
| `make check` | Run both lint and tests |
| `make coverage` | Run tests with a coverage report |
| `make bundle` | Copy `llms.txt` into the package data (run after editing `llms.txt`) |

Most tests download real tokenizers from the HuggingFace Hub (cached after the first run).
To skip them, run `pytest tests --no-slow`.
