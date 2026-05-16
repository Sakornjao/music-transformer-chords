# music-transformer-chords

## Run

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Analyze the default file:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache python3 main.py
```

Analyze a specific file:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache python3 main.py input/accompaniment.wav --limit 16
```

Create a browser GUI report:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache python3 main.py input/accompaniment.wav --report chord_report.html
```

This creates three web files:

```text
chord_report.html  # page structure and chord data
chord_report.css   # visual styles
chord_report.js    # audio/chord interaction logic
```

Use a trained chord model checkpoint:

```bash
NUMBA_CACHE_DIR=/tmp/numba_cache python3 main.py input/accompaniment.wav --checkpoint path/to/model.pt
```

Run tests:

```bash
python3 -m unittest discover -s tests
```
