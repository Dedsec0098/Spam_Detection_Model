#!/usr/bin/env bash
set -e

# This script runs before the Streamlit app starts on Hugging Face Spaces.
# It ensures required NLTK data is available.
python - <<'PY'
import ssl
import nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download only the data we need
nltk.download('punkt')
nltk.download('stopwords')

print('NLTK data downloaded')
PY
