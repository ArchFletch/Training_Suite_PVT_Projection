"""Render a local `pysidedeploy.spec` from the tracked template."""

from __future__ import annotations

import argparse
from pathlib import Path


def _path_text(value: str) -> str:
    return Path(value).resolve().as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a local pyside6-deploy spec file.")
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--exec-directory", required=True)
    parser.add_argument("--python-path", required=True)
    parser.add_argument("--icon", default="")
    args = parser.parse_args()

    template_text = Path(args.template).read_text(encoding="utf-8")
    rendered = (
        template_text.replace("__TITLE__", args.title)
        .replace("__PROJECT_DIR__", _path_text(args.project_dir))
        .replace("__INPUT_FILE__", _path_text(args.input_file))
        .replace("__EXEC_DIRECTORY__", _path_text(args.exec_directory))
        .replace("__PYTHON_PATH__", _path_text(args.python_path))
        .replace("__ICON__", _path_text(args.icon) if args.icon else "")
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
