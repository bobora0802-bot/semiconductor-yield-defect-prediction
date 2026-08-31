# Data provenance

This project uses the **SECOM** dataset from the UCI Machine Learning
Repository:

- Michael McCann and Adrian Johnston (2008), *SECOM*.
- DOI: https://doi.org/10.24432/C54305
- Dataset page: https://archive.ics.uci.edu/dataset/179/secom
- License: Creative Commons Attribution 4.0 International (CC BY 4.0).

The raw archive contains 1,567 manufacturing examples, 590 anonymised process
measurements, pass/fail labels, timestamps, and missing values. The raw files
are preserved unchanged under `data/raw/`.

Run `python scripts/download_data.py` to download the official archive.

