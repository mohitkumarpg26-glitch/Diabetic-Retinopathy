# Validation performed on this bundle

- Python syntax compilation passed for both application files.
- Data pipeline tested with synthetic readable images, one consistent-label duplicate pair and one conflicting-label pair.
  It retained one copy of the first pair, excluded both conflicting-label images, produced disjoint stratified splits,
  and wrote RGB 224x224 images.
- Read-only verification passed against the actual existing project: 2,450 train / 523 validation / 531 test images.
- Streamlit 1.67.0 ran on a temporary localhost server and returned `status: ready` with no trained model.
- Streamlit AppTest passed for populated evaluation tables, audit counts, learning-curve data,
  the missing-image prediction guard and the empty test-evaluation state. Synthetic fixtures were kept outside this bundle.
- Undergrading metric denominators were checked against a hand-computed example.

Not yet verified here: DenseNet training, DirectML/CUDA operator support, checkpoint resume across hardware,
or prediction with trained weights. Run both supplied `smoke` commands before training. No trained weights,
example accuracy claims, synthetic evaluation outputs, or dataset images are included in the deliverable.

The tested Streamlit runtime had newer pandas/Pillow/sklearn versions than the user-facing training requirements;
the requirements intentionally constrain NumPy and related packages for Torch 2.4.1 compatibility.
Run the smoke checks and first real training epoch on your installed environment before treating the pipeline as validated.
