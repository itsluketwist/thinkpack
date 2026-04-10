"""Shared regex patterns for reasoning block tags."""

import re


# matches any opening reasoning tag, e.g. <think>, <thinking>, <reasoning>, <thought>
OPEN_TAG = re.compile(
    r"<(think|thinking|reasoning|thought)>",
    re.IGNORECASE,
)

# matches any closing reasoning tag, e.g. </think>, </thinking>, etc.
CLOSE_TAG = re.compile(
    r"</(think|thinking|reasoning|thought)>",
    re.IGNORECASE,
)
