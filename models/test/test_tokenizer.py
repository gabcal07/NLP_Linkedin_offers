from models.src.preprocessing.tokenizer import load_tokenizer

if __name__ == "__main__":
    tokenizer = load_tokenizer("gpt2")
    test_text = "Hello, world! This is a test."

    encoded = tokenizer.encode(test_text)
    print(f"Encoded IDs: {encoded.ids}")
    print(f"Encoded Tokens: {encoded.tokens}")

    decoded = tokenizer.decode(encoded.ids)
    print(f"Decoded Text: {decoded}")

    assert (
        "Hello, world" in decoded
    ), f"Decoded text '{decoded}' does not match original '{test_text}' closely enough."

    print("\nTokenizer encode/decode test passed!")
