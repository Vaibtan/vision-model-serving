"""Django administrative entry point."""

from __future__ import annotations

import os
import sys

from django.core.management import execute_from_command_line


def main() -> None:
    os.environ.setdefault(
        "DJANGO_SETTINGS_MODULE",
        "vision_model_serving.web.settings",
    )
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
