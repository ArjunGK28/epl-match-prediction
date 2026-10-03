# EPL Match Prediction with Meta Statistics

UE24CS352A Machine Learning mini-project (Sem 5, Section K, Team 12, problem statement 69).

Team: PES1UG24CS922, PES1UG24CS643

## Problem statement

Reproduce the Stanford CS229 (Autumn 2019) project
[Gaining a Statistical Edge in Soccer Prediction using Machine Learning: Role of Meta Statistics in Match Prediction](https://cs229.stanford.edu/proj2019aut/data/assignment_308832_raw/26414102.pdf)
by Varun Harbola and Kyuho Lee.

The task is three-class classification of English Premier League matches
(home win / draw / home loss) from squad ratings known before kick-off.

## What the paper does

- **Data:** 3,800 EPL matches, seasons 2009/10 to 2018/19, scraped from whoscored.com.
- **Feature sets:**
  - F1: home and away average roster rating
  - F2: F1 plus average rating by position (attack, midfield, defence, goalkeeper)
  - F3: per-player rating, pass success %, passes, shots, key passes, blocks and interceptions per game
- **Models:** GDA, SVM (linear, degree-5 polynomial, RBF), softmax regression (linear, quadratic features), one-hidden-layer neural network.
- **Split:** random 80/20.
- **Headline result:** 58.89% test accuracy (degree-5 polynomial SVM on F1), against 46.19% for always predicting a home win.

## Setup and run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/reproduce.py
```

The script trains every model on F1, F2 and F3 and writes `results/results.csv` and
`results/accuracy.png`. A full run takes about 15 minutes; most of that is the linear and
polynomial SVMs on unscaled inputs.

## Repository layout

| Path | Contents |
|---|---|
| `src/reproduce.py` | Loads the data, trains all models, writes the results table and plot |
| `src/gda.py` | Gaussian discriminant analysis written by hand |
| `data/match_vectors_shuffled.csv` | F1/F2 features and labels in the authors' shuffled order |
| `data/match_vectors.csv` | The same matches, unshuffled, with a header row |
| `data/match_vectors_extended.csv` | F3 features and labels |
| `data/raw/` | Per-season rosters, lineups and match results as scraped by the authors |
| `results/` | Output of the latest run |
| `per.pdf` | The paper |

## Data

All data comes from the authors' repository,
[varunharbola/EPL_match_prediction](https://github.com/varunharbola/EPL_match_prediction),
which they scraped from whoscored.com. It has 3,791 matches (the paper states 3,800):
46.19% home wins, 24.80% draws, 29.02% home losses. Labels are 1, 0 and -1.

## Results

Test accuracy on the 20% hold-out (759 matches); training accuracy is in `results/results.csv`.

| Model | F1 | F2 | F3 |
|---|---|---|---|
| GDA | 58.50% | 56.52% | failed (singular covariance) |
| SVM (linear) | 57.97% | 57.84% | 54.41% |
| SVM (degree-5 polynomial) | **58.89%** | 55.73% | 47.04% |
| SVM (RBF) | 58.50% | 57.84% | 46.77% |
| SoftMax (linear) | 58.63% | 58.10% | 54.02% |
| SoftMax (quadratic) | 58.63% | 57.58% | not run |
| Neural network | 58.63% | 57.58% | 54.41% |

![Training and test accuracy](results/accuracy.png)

- The best result, 58.89% from the degree-5 polynomial SVM on F1, matches the paper.
- GDA and the SVMs reproduce the numbers saved in the authors' notebooks exactly, because
  they use the authors' shuffled split and scikit-learn settings.
- Every model beats the always-home-win baseline of 46.19% on F1 and F2.
- On F1 almost no model predicts a draw (draw recall is 0 to 1%).
- On F3 the polynomial and RBF SVMs reach 100% training accuracy and fall to about 47% on test.
  The RBF SVM predicts a home win for every test match.

### Differences from the authors' code

- The number of hidden nodes in the neural network is chosen by 5-fold cross-validation on the
  training set. The authors chose it by test accuracy.
- SoftMax regression uses scikit-learn's `LogisticRegression` on standardised inputs, with
  degree-2 polynomial features for the quadratic version.
- SoftMax with quadratic features on F3 is not run; the paper also stopped it.
