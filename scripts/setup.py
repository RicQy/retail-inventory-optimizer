#!/usr/bin/env python3
"""Setup script for development environment."""

import subprocess
import sys


def run_command(command: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"Running: {command}")
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        check=check,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result


def main() -> None:
    """Main setup function."""
    print("Setting up development environment...")

    # Check if Poetry is available
    poetry_result = run_command("poetry --version", check=False)

    if poetry_result.returncode == 0:
        print("Using Poetry for dependency management...")
        run_command("poetry install")
        run_command("poetry run pre-commit install")
    else:
        print("Poetry not found. Using pip...")
        run_command("pip install -r requirements.txt")
        run_command("pip install -r requirements-dev.txt")
        run_command("pre-commit install")

    print("Development environment setup complete!")
    print("\nNext steps:")
    print("1. Activate your virtual environment")
    print("2. Run tests: pytest")
    print("3. Check code quality: black . && isort . && flake8 . && mypy .")


if __name__ == "__main__":
    main()
