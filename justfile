# Creates a new homework assignment from the template if it doesn't exist,
# then runs it.
#
# Usage:
#   just new hw02
new hw_name:
    @if [ ! -d "{{hw_name}}" ]; then \
        echo "==> Creating new homework '{{hw_name}}' from template..."; \
        copier copy hw-template {{hw_name}}; \
        just run {{hw_name}}; \
    else \
        echo "==> Homework '{{hw_name}}' already exists. Skipping creation."; \
    fi

# Runs a specific homework assignment.
#
# Usage:
#   just run hw02
run hw_name:
    @echo "==> Running homework '{{hw_name}}'..."
    @uv run --project {{hw_name}} {{hw_name}}

# Creates a clean copy of the repository in a specified directory
# with a fresh git history.
#
# Usage: just export ../path/to/new-repo
export path:
    @if [ -d "{{path}}" ] && [ "$(ls -A '{{path}}')" ]; then \
        echo "Error: Directory '{{path}}' exists and is not empty." >&2; \
        exit 1; \
    fi
    @mkdir -p "{{path}}"
    @echo "--> Exporting a clean version of the repository to '{{path}}'..."
    @git archive HEAD | tar -x -C "{{path}}"
    @(cd "{{path}}" && \
      git init --initial-branch=main > /dev/null && \
      git add . && \
      git commit -m "Initial commit" > /dev/null)
    @echo "--> Successfully created a clean repository in '{{path}}'."

# Generates a PDF of all source and config files for a homework.
#
# Usage:
#   just pdf hw02
pdf hw_name:
    @echo "==> Generating PDF for '{{hw_name}}'..."
    @find {{hw_name}} \( -path '*/.venv' -o -path '*/__pycache__' -o -name '*~' \) -prune -o \( -name "*.py" -o -name "*.toml" \) | xargs a2ps -2 --media=letter -o {{hw_name}}.ps
    @gs -sDEVICE=pdfwrite -sPAPERSIZE=letter -dPDFFitPage -sOutputFile={{hw_name}}.pdf -dNOPAUSE -dBATCH {{hw_name}}.ps `find {{hw_name}} \\( -path '*/.venv' -o -path '*/__pycache__' -o -name '*~' \\) -prune -o -name "*.pdf" -print` > /dev/null 2>&1
    @rm {{hw_name}}.ps
    @echo "==> PDF generated at '{{hw_name}}.pdf'"

# run test
# just test hw_name test_script
test hw_name test_script:
    @echo "==> Running test script on '{{hw_name}}'"
    @uv run --project {{hw_name}} {{test_script}}

