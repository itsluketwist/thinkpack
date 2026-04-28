# ***development***

## *setup*

Clone the repository:

```shell
git clone https://github.com/itsluketwist/thinkpack.git
```

We use [`uv`](https://astral.sh/blog/uv) for project management.
Once cloned, create a virtual environment and install with dev dependencies:

```shell
python -m venv .venv

. .venv/bin/activate

pip install uv

uv sync
```

Install pre-commit hooks (run once after cloning):

```shell
make newlint
```

## *commands*

| Command | Description |
|---|---|
| `make lint` | Run pre-commit on all files |
| `make test` | Run the test suite |
| `make coverage` | Run tests with coverage report |
| `make bundle` | Copy `llms.txt` into package data (run after editing `llms.txt`) |
