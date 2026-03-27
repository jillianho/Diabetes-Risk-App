# Diabetes Risk App

A Streamlit app that estimates diabetes risk probability using a trained model plus transparent proxy logic when optional lab values are missing.

## Model Documentation

- Model artifact: `model.pkl`
- Training script: `train_model.py`
- Model type: `sklearn.pipeline.Pipeline`
  - `SimpleImputer(strategy='median')`
  - `LogisticRegression(max_iter=1000)`
- Dataset file used in training: `diabetes.csv`
- Target: `Outcome`
- Input features (Pima-format): pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age

## Validation Metrics

The training script reports:
- Accuracy
- Classification report (precision, recall, f1-score, support)

To regenerate metrics from the current code:

```bash
python train_model.py
```

## Important Limitations

- This app is not medical advice and not a diagnosis.
- Some values may be estimated by heuristics when user lab values are not provided.
- Proxy logic is approximation-only and should be treated as directional.
- Probabilities may not be perfectly calibrated for all populations.

## Calibration Note

Before using percentages for decision support, consider probability calibration:
- Platt scaling (`CalibratedClassifierCV(..., method='sigmoid')`)
- Isotonic regression (`CalibratedClassifierCV(..., method='isotonic')`)

Evaluate with calibration curves and Brier score on a held-out set.

## Browser and Mobile Notes

- The dial uses CSS `conic-gradient` with a fallback style.
- Validate rendering on Safari and Firefox.
- Validate narrow-screen behavior on real phones due to custom CSS.

## Sharing Results

The app provides a downloadable plain-text summary on the results page.
