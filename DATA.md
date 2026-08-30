# Data provenance and licenses

## WDBC

The vendored file `data/wdbc_breast_cancer.csv` is the 569-row,
30-feature **Breast Cancer Wisconsin (Diagnostic)** dataset from the UCI
Machine Learning Repository.

- UCI dataset identifier: 17
- DOI: <https://doi.org/10.24432/C5DW2B>
- Creators: William Wolberg, Olvi Mangasarian, Nick Street, and W. Street
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Repository source: <https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic>

The sealed method record commits the exact CSV bytes used in the study. The
public manifest provides a second repository-level hash.

## Handwritten digits

The digits experiment calls `sklearn.datasets.load_digits` from
scikit-learn 1.9.0. The loader contains 1,797 8x8 images with integer-valued
pixels from 0 to 16. Scikit-learn identifies it as a copy of the test split of
the UCI **Optical Recognition of Handwritten Digits** dataset.

- UCI dataset identifier: 80
- DOI: <https://doi.org/10.24432/C50P49>
- Creators: E. Alpaydin and C. Kaynak
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- UCI source: <https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits>
- Loader documentation: <https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html>

The data are loaded from the pinned scikit-learn package; the experiment does
not download data at runtime.

## Modular arithmetic

The modular-addition datasets are generated exhaustively by tracked Python
code. There is no external dataset. Protocol files record each modulus,
train/evaluation split, seed, and ordering rule.

## Evaluation-set scope

All certified events are deterministic events on the fixed evaluation sets
defined by the protocols. Dataset provenance does not turn these statements
into population-generalization guarantees.

