import os
from pkgutil import extend_path

# Discover extension packages installed from separate source trees.
__path__ = extend_path(__path__, __name__)

# Set TOKENIZERS_PARALLELISM to avoid fork warning
# This must be set before any tokenizers are imported/used
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
