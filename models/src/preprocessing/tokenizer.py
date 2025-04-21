"""
Utility functions for loading and using Hugging Face tokenizers.
"""

from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
import os
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Cache directory for tokenizers (optional, but good practice)
CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "huggingface", "tokenizers_custom"
)
os.makedirs(CACHE_DIR, exist_ok=True)


def load_tokenizer(
    tokenizer_name: str = "gpt2", cache_dir: str | None = CACHE_DIR
) -> Tokenizer:
    """
    Loads a pre-trained tokenizer from the Hugging Face Hub.

    Args:
        tokenizer_name: The name of the tokenizer on the Hugging Face Hub
                        (e.g., "gpt2", "bert-base-uncased").
        cache_dir: Directory to cache downloaded tokenizer files. Defaults to
                   ~/.cache/huggingface/tokenizers_custom. Set to None to use
                   the default cache location of the huggingface_hub library.

    Returns:
        An instance of the Hugging Face Tokenizer.

    Raises:
        Exception: If the tokenizer cannot be loaded.
    """
    logging.info(f"Attempting to load tokenizer: {tokenizer_name}")
    try:
        try:
            config_path = hf_hub_download(
                repo_id=tokenizer_name,
                filename="tokenizer.json",
                cache_dir=cache_dir,
                library_name="nlp-linkedin-offers",
                library_version="0.1.0",
            )
            tokenizer = Tokenizer.from_file(config_path)
            logging.info(
                f"Successfully loaded tokenizer '{tokenizer_name}' from tokenizer.json."
            )
            return tokenizer
        except Exception as e:
            logging.warning(
                f"Could not load '{tokenizer_name}' directly from tokenizer.json: {e}. "
                "Attempting legacy loading (might be slower or require transformers)."
            )
            raise FileNotFoundError(
                f"Could not find or load tokenizer.json for '{tokenizer_name}'. "
                "Consider installing the 'transformers' library and using AutoTokenizer for broader compatibility."
            )

    except Exception as e:
        logging.error(f"Failed to load tokenizer '{tokenizer_name}': {e}")
        raise
