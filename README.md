# Comparing Classifiers (early ML practice, 2024)

One of my first machine-learning notebooks: four scikit-learn classifiers trained on the same tiny dataset, then asked to predict the same new input, to see how differently they behave.

## What it does

`Comparing_classifiers.ipynb`:

1. Builds a small hand-made dataset of `[height, weight, shoe size]` labelled `male` / `female`.
2. Trains a **Decision Tree**, **Random Forest**, **Support Vector Classifier** and **K-Nearest Neighbours** on it.
3. Takes a new person's measurements as input and prints each model's prediction.

On the saved run (170 cm, 60 kg, shoe size 43), three models predicted `female` and the SVC predicted `male`, a direct look at how models with different decision boundaries can disagree on the same point.

## What I'd do differently now

This was a first exercise in the scikit-learn API, not a real evaluation:

- The dataset is a few hand-typed rows, so there is no train/test split and no accuracy to compare.
- SVC and KNN depend on feature scale, and the features aren't standardised.
- A proper comparison would use a real dataset, cross-validation and metrics like accuracy and F1.

My later lab notebooks in [ML-Lab](https://github.com/eshitakundu/ML-Lab) use held-out splits and proper metrics.

## Run it

```bash
pip install scikit-learn jupyter
jupyter notebook Comparing_classifiers.ipynb
```
