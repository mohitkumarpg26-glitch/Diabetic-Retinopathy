# Retina project: complete PowerShell command guide

Mohit Kumar · 2026PAD2529 · mohit.kumar.pg26@nsut.ac.in

This bundle adds code and UI to your existing `C:\Users\SAHMO\Downloads\projecto` folder.
The bundle contains no dataset or trained weights. Do not repeat data preparation if you already completed
the cleaning and resizing commands in the chat. Use the same cleaned splits for every experiment.

## 1. Copy this bundle into the project

Run from PowerShell. Use the actual downloaded ZIP path if it differs:

```powershell
$project = 'C:\Users\SAHMO\Downloads\projecto'
New-Item -ItemType Directory -Force -Path $project | Out-Null
Expand-Archive -LiteralPath 'C:\Users\SAHMO\Documents\Codex\2026-11-13\macos-dmg-3-18-1-alpha\outputs\Retina_Project.zip' -DestinationPath $project -Force
Set-Location $project
```

This overwrites only files shipped in the bundle, not your raw data, existing src scripts, or reports.
All subsequent commands run from this project folder.

## 2. Create the environment: choose ONE hardware route

For RX 6600 / DirectML on the current desktop, let the download already running finish first.
Do not start two pip installations into the same environment at once. Python 3.12 x64 is required for these pinned Windows wheels.

```powershell
python -m venv .venv-directml
.\.venv-directml\Scripts\python.exe -I -m pip install --index-url https://pypi.org/simple --upgrade pip
.\.venv-directml\Scripts\python.exe -I -m pip install --index-url https://pypi.org/simple 'torch-directml==0.2.5.dev240914'
.\.venv-directml\Scripts\python.exe -I -m pip install --index-url https://pypi.org/simple -r requirements.txt
$py = "$PWD\.venv-directml\Scripts\python.exe"
$device = 'directml'
```

Alternatively, GTX 1650 Max-Q laptop (NVIDIA driver must support CUDA 12.1):

```powershell
python -m venv .venv-cuda
.\.venv-cuda\Scripts\python.exe -I -m pip install --index-url https://pypi.org/simple --upgrade pip
.\.venv-cuda\Scripts\python.exe -I -m pip install 'torch==2.4.1' 'torchvision==0.19.1' --index-url https://download.pytorch.org/whl/cu121
.\.venv-cuda\Scripts\python.exe -I -m pip install --index-url https://pypi.org/simple -r requirements.txt
$py = "$PWD\.venv-cuda\Scripts\python.exe"
$device = 'cuda'
& $py -I -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

CPU fallback (separate environment):

```powershell
python -m venv .venv-cpu
.\.venv-cpu\Scripts\python.exe -I -m pip install 'torch==2.4.1' 'torchvision==0.19.1' --index-url https://download.pytorch.org/whl/cpu
.\.venv-cpu\Scripts\python.exe -I -m pip install --index-url https://pypi.org/simple -r requirements.txt
$py = "$PWD\.venv-cpu\Scripts\python.exe"
$device = 'cpu'
```

The `$py` and `$device` variables last only in the current PowerShell window. Set them again if you reopen it.
Pinned Torch versions keep the project compatible with the DirectML preview. NumPy 1.26 avoids older Torch/NumPy 2 incompatibilities.
DirectML remains a hardware/operator compatibility test; passing the smoke test does not guarantee CUDA-like speed or complete operator coverage.
Record warnings about CPU fallbacks, and measure the first epoch before estimating total time.

## 3. Data download and extraction (SKIP: you have already done this)

Sign into Kaggle, join APTOS 2019, and accept its competition rules using your own account.
Download All at https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
Wait until the ZIP download finishes. Then extract:

```powershell
New-Item -ItemType Directory -Force .\data\raw | Out-Null
Expand-Archive -LiteralPath 'C:\Users\SAHMO\Downloads\aptos2019-blindness-detection.zip' -DestinationPath .\data\raw
Get-ChildItem .\data\raw
```

Expected structure: `data/raw/train.csv`, `data/raw/train_images/*.png`, plus the competition test files.
The competition `test.csv` has no public diagnosis labels, so our held-out evaluation uses a split of labeled `train.csv`.
Never include Kaggle credentials, virtual environments, or the dataset in public GitHub commits.

## 4. Audit, clean, split and resize (SKIP if completed in chat)

From a fresh extraction:

```powershell
& $py -I .\retina_project.py prepare
```

Or reuse the existing audit ONLY if the source data have not changed:

```powershell
& $py -I .\retina_project.py prepare --reuse-audit
```

This audits readability and exact file duplicates, excludes all conflicting-label groups pending expert review,
keeps one copy per consistently labeled duplicate group, creates seed-42 stratified 70/15/15 splits,
and creates RGB 224x224 copies using padding. Source images remain untouched.
Do not run this after starting experiments. Freeze the splits before model selection.

Your current results: 3,662 originals; 123 duplicate groups; 30 groups had conflicting labels;
96 extra copies and 62 conflicting-label images excluded; 3,504 retained.
Final splits: train 2,450; validation 523; test 531. Keep exclusion policy visible in the report.

## 5. Verify your current prepared data (START HERE after installation)

```powershell
& $py -I .\retina_project.py verify
```

## 6. Test the actual model on the selected hardware

```powershell
& $py -I .\retina_project.py smoke --device $device --mode baseline
& $py -I .\retina_project.py smoke --device $device --mode ordinal
```

Both should print that forward/backward/optimizer passed. These two steps use random input and random
weights only to test compatibility; they do not create a trained model or research results.
If the wrong GPU is selected, list adapters:

```powershell
& $py -I -c "import torch_directml as d; print([(i,d.device_name(i)) for i in range(d.device_count())])"
```

Add `--adapter 1` (or the correct number) to DirectML commands if needed.

## 7. Train a quick baseline

```powershell
& $py -I .\retina_project.py train --device $device --mode baseline --run baseline --epochs 3 --batch-size 8
```

First run downloads approximately 31 MB of ImageNet weights. The DenseNet121 backbone is frozen;
the five-grade classifier is trained with inverse-frequency class weights. This is real transfer learning,
but does not fine-tune the entire backbone. Batch progress and epoch duration are printed.
If memory is insufficient, use batch size 4 with a NEW run name and keep that batch size on resume.

`runs/baseline/last.pt` saves model, optimizer and RNG states after every COMPLETE epoch.
`runs/baseline/best.pt` is chosen by validation macro F1. Press Ctrl+C to stop; an interrupted epoch is replayed on resume.
Evaluation and UI use best.pt. Reusing a run name without --resume is blocked.

## 8. Evaluate validation and open the UI

```powershell
& $py -I .\retina_project.py evaluate --device $device --run baseline --split validation --batch-size 8
& $py -I .\retina_project.py report --split validation
& $py -I -m streamlit run .\app.py
```

Open http://localhost:8501 if the browser does not open automatically. Stop the app using Ctrl+C.
The UI also opens before training, but predictions are disabled until a checkpoint exists.
The app runs inference on CPU to avoid interrupting GPU training; uploads are processed in memory.

## 9. Train the ordinal comparison

Run this from another PowerShell window after setting `$py` and `$device` again, or stop the app first.

```powershell
& $py -I .\retina_project.py train --device $device --mode ordinal --run ordinal --epochs 3 --batch-size 8
& $py -I .\retina_project.py evaluate --device $device --run ordinal --split validation --batch-size 8
& $py -I .\retina_project.py report --split validation
```

This uses the same pretrained backbone, splits, seed, batch size, epoch budget and learning rate.
The ordinal head has one learned score and four ordered thresholds. Its cumulative probabilities cannot cross.
Its class-weighted binary cumulative loss uses ordered grades. It is not the paper's CBAM model or a claim of novel architecture.
Report whether it reduces large undergrading errors; it may perform worse or yield inconclusive results.

## 10. Continue training later (TOTAL epoch target)

```powershell
& $py -I .\retina_project.py train --device $device --mode baseline --run baseline --epochs 10 --batch-size 8 --resume
& $py -I .\retina_project.py train --device $device --mode ordinal --run ordinal --epochs 10 --batch-size 8 --resume
```

`--epochs 10` means ten total epochs, not ten additional epochs. Resume requires the original mode,
batch size, learning rate, freeze setting and identical split files. Device can change.
The saved optimizer enables continuation across CPU/DirectML/CUDA, subject to backend operator support;
bit-for-bit reproducibility across backends is not guaranteed.

For full backbone fine-tuning, use new, matched runs (these initialize from ImageNet, not your head-only run):

```powershell
& $py -I .\retina_project.py train --device $device --mode baseline --run baseline_finetune --epochs 10 --batch-size 4 --lr 0.0001 --finetune
& $py -I .\retina_project.py train --device $device --mode ordinal --run ordinal_finetune --epochs 10 --batch-size 4 --lr 0.0001 --finetune
```

Do not compare a head-only three-epoch baseline against a fully fine-tuned ten-epoch ordinal model as evidence
that the ordinal loss alone caused an improvement. Choose the pair and budget using validation results.

## 11. Final held-out evaluation (after fixing the experiment)

For the baseline/ordinal pair above, after choosing the final training budget:

```powershell
& $py -I .\retina_project.py evaluate --device $device --run baseline --split test --batch-size 8
& $py -I .\retina_project.py evaluate --device $device --run ordinal --split test --batch-size 8
& $py -I .\retina_project.py report --split test
& $py -I -m streamlit run .\app.py
```

Substitute the chosen fine-tuning run names and batch size if those are your final pair.
Do not use test results to tune hyperparameters. If evaluated early, describe those test results as exploratory
and disclose repeated test access. Longer training after looking at test results prevents treating it as an untouched test set.

Outputs: confusion matrices, learning curves, per-image probabilities, per-class recall/F1,
macro F1, quadratic weighted kappa, mean absolute grade error, and two-or-more-grade undergrading rates.
Undergrading is reported with both all-image and true-grade-2-plus denominators.

## 12. Report and presentation

Use `results/research_summary.md` as the starting draft and the comparison CSV for actual measured results.
Include: research question; reference/gap; dataset exclusions; fixed splits; architecture/loss;
hardware/training budget; baseline vs ordinal results; confusion matrices; severe undergrading;
privacy/provenance; fairness limits; uncertainty; limitations and next steps; AI assistance/prompts.
No results are prefilled or fabricated. Add confidence intervals and external validation as further work
unless you implement and actually measure them. The ethics and AIML submissions can emphasize different
parts of the same implementation; follow each instructor's submission rules.

## 13. Move to the GTX laptop

Copy the project folder, including `retina_project.py`, `app.py`, `requirements.txt`, `.streamlit`,
`data/processed`, `results`, and `runs`. Original images and papers may be archived separately.
Do NOT copy virtual environments; recreate the CUDA environment on the laptop using section 2.
The project code resolves paths relative to itself, so its parent folder may change.
Then set `$py`, `$device = 'cuda'`, run verify, smoke, and the matching resume command.
Keep your audit and split files together with the checkpoints.

## 14. Archive results for submission

```powershell
& $py -I -m pip freeze | Set-Content -Encoding utf8 .\results\environment.txt
Compress-Archive -Path .\retina_project.py,.\app.py,.\requirements.txt,.\START_HERE.md,.\results,.\runs -DestinationPath .\Retina_Submission.zip -Force
```

This archive may be large because it includes optimizer checkpoints. It omits raw data, prepared images and
virtual environments. Keep those locally for reproducibility and verify competition rules before sharing data.

## Troubleshooting

- `No module named...`: use the intended `$py`; run its `-I -m pip install -r requirements.txt`.
- DirectML `no matching distribution`: use the explicit preview version AND public PyPI command above.
- NumPy/Torch warning: requirements pins NumPy to 1.26.4; install requirements after torch-directml.
- DirectML unsupported operator or many CPU-fallback warnings: capture the error; switch to CUDA laptop or CPU.
- Slow first batches: initialization/warm-up can be slow. Measure the first complete epoch.
- Changed splits: restore the original split CSVs or start a new run. Never resume on changed splits.
- App has no prediction: complete an epoch, then click Refresh results.
- Do not load untrusted `.pt` files. The app only selects local run checkpoints, not uploaded models.

References: https://pypi.org/project/torch-directml/ · https://docs.pytorch.org/vision/0.19/models/generated/torchvision.models.densenet121.html
· https://doi.org/10.1109/ACCESS.2022.3165193
