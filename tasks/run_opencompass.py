"""Register this repository's TTT model classes, then run OpenCompass CLI."""

import inference_model  # noqa: F401
from opencompass.cli.main import main


if __name__ == "__main__":
    main()
