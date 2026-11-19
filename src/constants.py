"""Shared constants. Bit order of the availability mask is (text, audio, video) = (4, 2, 1)."""
EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
SENTIMENTS = ["negative", "neutral", "positive"]
TEXT, AUDIO, VIDEO = 4, 2, 1
FULL = 7
PATTERNS = [7, 6, 5, 3, 4, 2, 1]          # 111 110 101 011 100 010 001
MODALITIES = ["text", "audio", "video"]
DIMS = {"text": 768, "audio": 88, "video": 512}
D_IN = sum(DIMS.values()) + 3             # 1371
# Official MELD counts: (dialogues, utterances); check against your downloaded CSVs.
EXPECTED_COUNTS = {"train": (1038, 9989), "dev": (114, 1109), "test": (280, 2610)}
PROFILES = {
    "uniform": {p: 1 / 7 for p in PATTERNS},
    # ILLUSTRATIVE profile only (not an empirical deployment statistic)
    "skewed": {7: 0.55, 6: 0.15, 5: 0.10, 3: 0.05, 4: 0.05, 2: 0.05, 1: 0.05},
}


def pattern_name(m: int) -> str:
    return f"{(m >> 2) & 1}{(m >> 1) & 1}{m & 1}"


def cardinality(m: int) -> int:
    return bin(int(m)).count("1")
