import pytest
from src.training.data_processing import clean_tweet

def test_clean_tweet_removes_handles():
    """Ensure @user handles are stripped out."""
    input_text = "Hello @elonmusk, how is Mars?"
    expected = "hello how is mars"
    assert clean_tweet(input_text) == expected

def test_clean_tweet_removes_urls():
    """Ensure links are removed to reduce noise in the model."""
    input_text = "Check this out https://google.com for info"
    expected = "check this out for info"
    assert clean_tweet(input_text) == expected

def test_clean_tweet_lowercases():
    """Ensure text is normalized to lowercase."""
    input_text = "I AM SCREAMING"
    expected = "i am screaming"
    assert clean_tweet(input_text) == expected

def test_clean_tweet_handles_special_characters():
    """Ensure punctuation and symbols are removed."""
    input_text = "Victory!!! 🎉 #win"
    # Assuming your regex removes # and non-alpha
    assert "victory" in clean_tweet(input_text)
    assert "#" not in clean_tweet(input_text)

def test_clean_tweet_empty_input():
    """Ensure the function doesn't crash on empty or null input."""
    assert clean_tweet("") == ""
