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
.venv\Scriptsctivate
pip install -r requirements.txt
python src/build_features.py
python src/run_experiments.py
```

`build_features.py` turns the raw files into `data/processed/matches.csv`. `run_experiments.py`
trains every model on F1, F2 and F3 and writes `results/results.csv` and `results/accuracy.png`.
A full run takes about a minute.

## Repository layout

| Path | Contents |
|---|---|
| `src/build_features.py` | Builds F1, F2 and F3 from the raw rosters, lineups and results |
| `src/run_experiments.py` | Splits the data, trains all models, writes the results table and plot |
| `src/gda.py` | Gaussian discriminant analysis written by hand |
| `data/raw/` | Per-season rosters, lineups and match results |
| `data/processed/matches.csv` | One row per match: season, teams, score, label and all features |
| `results/` | Output of the latest run |
| `per.pdf` | The paper |

## Data

The raw files in `data/raw/` come from the authors' repository,
[varunharbola/EPL_match_prediction](https://github.com/varunharbola/EPL_match_prediction),
which they scraped from whoscored.com. We did not redo the scraping. Everything after that
(features, split, models) is our own code.

There are 3,791 matches with both a lineup and a result (the paper states 3,800):
46.19% home wins, 24.80% draws, 29.02% home losses. Labels are 1, 0 and -1.

## How we build the features

- Only the **starting eleven** of each team is used. Substitutes and minutes played are in the
  raw lineups but are not known before kick-off, so they are left out.
- Each player's statistics are that player's season averages from the roster file of the same season.
- **F1** (2 features): mean rating of the home eleven and of the away eleven.
- **F2** (10 features): F1 plus the mean rating of the goalkeeper, defenders, midfielders and
  forwards of each team.
- **F3** (154 features): rating, pass success %, passes, shots, key passes, blocks and
  interceptions per game for each of the 22 starters, in lineup order.
- A starter missing from the roster file gets the team's season average.

## Method

- One random 80/20 split, stratified by outcome, with a fixed seed (3,032 train, 759 test).
- All models except GDA get standardised inputs, with the scaler fitted on the training set.
- SVMs and SoftMax use scikit-learn defaults. The neural network has one hidden layer, and the
  number of hidden nodes (4, 8 or 16) is chosen by 5-fold cross-validation on the training set.
- No hyperparameters have been tuned yet.

## Results

Test accuracy on the 759 held-out matches; training accuracy and per-class recall are in
`results/results.csv`.

| Model | F1 | F2 | F3 |
|---|---|---|---|
| GDA | 57.05% | 53.62% | failed (singular covariance) |
| SVM (linear) | 56.92% | 57.31% | 53.23% |
| SVM (degree-5 polynomial) | 52.04% | 52.04% | 49.01% |
| SVM (RBF) | 56.65% | 55.73% | 55.60% |
| SoftMax (linear) | 56.79% | 56.92% | 53.49% |
| SoftMax (quadratic) | **57.44%** | 54.28% | not run |
| Neural network | **57.44%** | 57.05% | 53.75% |

![Training and test accuracy](results/accuracy.png)

- Best test accuracy is 57.44% on F1, against the paper's 58.89% and the always-home-win
  baseline of 46.19%.
- On F1 no model predicts draws: draw recall is 0% for every model.
- On F3 training accuracy rises (up to 78% for the RBF SVM) while test accuracy falls, the same
  overfitting pattern the paper reports.
- The degree-5 polynomial SVM is the weakest model here, unlike in the paper. It runs with
  scikit-learn's default `coef0=0` on standardised inputs and has not been tuned.
- GDA fails on F3 because the class covariance matrices are singular, as in the paper.
- SoftMax with quadratic features on F3 is not run; the paper also stopped it.

## Known limitations

- Player ratings are season averages, so they include the match being predicted.
- The split is random across seasons rather than chronological.
