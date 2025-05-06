import sys
import os

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.preprocessing.normalizer import TextNormalizer

if __name__ == "__main__":
    normalizer = TextNormalizer()
    test_text = "Hello, world! This is a test. The chicken is 4 years old."

    normalized_text = normalizer.normalize_text_for_topic_modeling(test_text)
    print(f"Normalized Text: {normalized_text}")

    # Check if the normalization process works as expected
    assert "hello world test chicken four year old" in normalized_text, (
        f"Normalized text '{normalized_text}' does not match expected output. \n this was {test_text}"
    )
    print("\nText normalization test passed!")
