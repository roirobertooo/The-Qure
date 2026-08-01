# Experimental feature-fusion pipeline

This module combines three inputs in a small demonstration network:

- a 64-dimensional MRI-derived vector;
- one Entanglement Entropy Score;
- a four-value one-hot encoding of the supplied impairment category.

The resulting 69-value input passes through a `69 -> 32 -> 1` multilayer perceptron. The training targets are fixed values assigned to the four impairment categories in the code. They are not longitudinal patient outcomes.

For that reason, the output is best understood as a demonstration of feature fusion and interface plumbing. It is not a measured probability of conversion to Alzheimer's disease.

## Train the demonstration model

```bash
python train_risk_model.py --max_samples 50 --epochs 20
```

The command expects class-organized images under `data/train` and writes `alzheimer_risk_model.pth` unless another output path is supplied.

## Inspect the pipeline

```bash
python test_risk_pipeline.py
```

The interactive script can list local images, process a selected image, or compare the demonstration outputs across supplied categories.

## Programmatic use

```python
from alzheimer_risk_pipeline import AlzheimerRiskPipeline

pipeline = AlzheimerRiskPipeline("alzheimer_risk_model.pth")
result = pipeline.predict_risk(
    image_path="path/to/mri.jpg",
    category_name="Mild Impairment",
    uncertainty_samples=100,
)
```

`predict_risk` also reports a variation band produced by adding small perturbations to the input features. That band is not a calibrated clinical confidence interval.

## Interpretation limits

- The category supplied to the model is also the basis for the category-level training target.
- The target values are literature-inspired constants embedded in the code, not observed outcomes for the included images.
- The model has no preserved prospective, external, or held-out clinical validation.
- The perturbation-based variation band is capped by implementation choice.
- The EES path uses statevector simulation and has no demonstrated quantum performance advantage.

Do not use the generated percentages for diagnosis, prognosis, screening, or treatment decisions.
