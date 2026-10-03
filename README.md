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

## Status

Repository scaffolded. Data and code are not added yet.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
