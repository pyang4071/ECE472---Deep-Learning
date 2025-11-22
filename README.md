# ECE472 Starter Repository

This repository provides a starter template for homework assignments in ECE472 at The Cooper Union. It is designed to help you get
up and running quickly with a modern Python development environment.

## 🚀 Getting Started

Follow these steps to set up your environment and start your first assignment.

### 1. Installation

You'll need four tools: `uv` for Python environment management, `copier` for templating, `just` as a command runner, and `pre-commit` for code quality.

-   **Install `uv`**: `uv` is an extremely fast Python package installer and resolver. Follow the [official installation
instructions](https://docs.astral.sh/uv/getting-started/installation/).

-   **Install `copier`**: `copier` creates projects from templates. Install it with `uv`:
    ```bash
    uv tool install copier
    ```

-   **Install `just`**: `just` is a handy command runner. See the [installation
guide](https://just.systems/man/en/pre-built-binaries.html) for your platform.

-   **Install `pre-commit`**: `pre-commit` runs checks on your code before you commit. Install it with `uv`:
    ```bash
    uv tool install pre-commit
    ```

### 2. Set Up Pre-Commit Hooks

This repository uses `pre-commit` to automatically format and lint your code before you make a commit. This helps maintain a
consistent code style.

After installing `pre-commit`, run the following command in the root of this repository to set up the hooks:

```bash
pre-commit install
```

Now, `ruff` will automatically check and format your files every time you commit your changes.

### 3. Create Your First Assignment

Once the tools are installed, you can create a new homework project. For example, to create `hw01`:

```bash
just new hw01
```

This command will:
1.  Use `copier` to create a new directory `hw01/` from the `hw-template/`.
2.  Ask you a few questions to set up your project's `pyproject.toml` file.
3.  Install the project's dependencies into a new virtual environment using `uv`.
4.  Run the new project to confirm it works.

### 4. Run Your Assignment

To run your homework code at any time:

```bash
just run hw01
```

## Backing Up Your Work on GitHub

After you have initialized your local repository, you should link it to a new repository on GitHub to back up your work and track your changes.

1.  Go to [GitHub](https://github.com) and create a new, empty repository. Do **not** initialize it with a `README.md` or `.gitignore` file, as your project already has these.

2.  Copy the repository URL that GitHub provides (e.g., `https://github.com/your-username/your-repo-name.git`).

3.  In your local repository's terminal, link it to the remote repository on GitHub. Replace the placeholder URL with your own:
    ```bash
    git remote add origin <YOUR_REPOSITORY_URL>
    ```

4.  Push your initial commit to GitHub to upload your files:
    ```bash
    git push -u origin main
    ```

## 🛠️ Development Workflow

Here are some more details about how to work with your assignment projects.

### Project Structure and Dependencies

Each homework assignment is a self-contained Python project located in its own directory (e.g., `hw01/`). Inside, you'll find a
`pyproject.toml` file. This file defines your project's metadata and, most importantly, its dependencies.

#### Adding Dependencies

To add a new library (e.g., `scikit-learn`) to your project, you can navigate into the project's directory and use `uv add`:

```bash
cd hw01
uv add scikit-learn
```

Alternatively, you can add dependencies from other locations (like the root of this repository) by using the `--project` flag. This is especially useful for automation:

```bash
uv add scikit-learn --project hw01
```

Both commands will add `scikit-learn` to the `dependencies` list in your `hw01/pyproject.toml` and install it into the project's
virtual environment.

For more details, see the [`uv` documentation on managing dependencies](https://docs.astral.sh/uv/concepts/projects/dependencies/).

> **⚠️ Important:** Always use `uv add` to manage your project's dependencies. Avoid using `uv pip install`, as this will not update
your `pyproject.toml` file, leading to reproducibility issues.

### 🪵 Effective Logging

This template comes with a powerful logging setup using `structlog`. Instead of using `print()` statements for debugging, which you
have to add and remove constantly, you can use the logger.

#### How to Use the Logger

The entry point of your application (`src/<project_name>/__init__.py`) already configures and creates a logger instance for you.

```python
import structlog

log = structlog.get_logger()
log.info("This is an informational message.")
log.debug("This is a debug message.", data="some value")
```

#### Log Levels

Log messages have different levels of severity. By default, you will only see `INFO` level messages and above. The common levels
are:

-   `DEBUG`: Detailed information, typically of interest only when diagnosing problems.
-   `INFO`: Confirmation that things are working as expected.
-   `WARNING`: An indication that something unexpected happened, or indicative of some problem in the near future (e.g., ‘disk
space low’). The software is still working as expected.
-   `ERROR`: Due to a more serious problem, the software has not been able to perform some function.
-   `CRITICAL`: A serious error, indicating that the program itself may be unable to continue running.

To see `DEBUG` messages, you can set the `LOG_LEVEL` environment variable when running your code:

```bash
LOG_LEVEL=DEBUG just run hw01
```

This is incredibly useful for "print debugging". You can leave `log.debug(...)` statements in your code and only enable them when
you need them, without having to comment or uncomment lines. This keeps your code clean and your debugging output organized.
