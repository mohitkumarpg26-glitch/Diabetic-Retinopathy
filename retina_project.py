"""APTOS research pipeline. Run --help for commands. No clinical use."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import json
import random
import shutil
import time

ROOT = Path(__file__).resolve().parent
GRADES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def padded_image(image):
    from PIL import Image, ImageOps
    return ImageOps.pad(ImageOps.exif_transpose(image).convert('RGB'), (224, 224),
                        method=Image.Resampling.LANCZOS, color=(0, 0, 0))


def prepare(args):
    from PIL import Image
    raw = ROOT / 'data/raw'
    out = ROOT / 'data/processed'
    out.mkdir(parents=True, exist_ok=True)
    rows = read_rows(raw / 'train.csv')
    ids = [r['id_code'] for r in rows]
    assert len(ids) == len(set(ids)), 'Repeated IDs in train.csv'
    assert all(r['diagnosis'] in {'0', '1', '2', '3', '4'} for r in rows)
    audit_path = ROOT / 'results/image_audit.json'
    if args.reuse_audit:
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        assert audit['images_checked'] == len(rows), 'Audit size mismatch'
    else:
        hashes, bad = defaultdict(list), []
        for n, row in enumerate(rows, 1):
            p = raw / 'train_images' / (row['id_code'] + '.png')
            try:
                with Image.open(p) as im:
                    im.load()
                with p.open('rb') as f:
                    hashes[hashlib.file_digest(f, 'sha256').hexdigest()].append(row['id_code'])
            except Exception as e:
                bad.append({'id_code': row['id_code'], 'error': str(e)})
            if n % 250 == 0:
                print(f'Audited {n}/{len(rows)}', flush=True)
        audit = {'images_checked': len(rows), 'unreadable_images': bad,
                 'exact_duplicate_groups': [v for v in hashes.values() if len(v) > 1]}
        write_json(audit_path, audit)
    labels = {r['id_code']: r['diagnosis'] for r in rows}
    excluded = {v['id_code']: 'unreadable' for v in audit['unreadable_images']}
    for group in audit['exact_duplicate_groups']:
        assert set(group) <= set(ids), 'Unknown ID in audit'
        if len({labels[i] for i in group}) > 1:
            excluded.update({i: 'conflicting_labels' for i in group})
        else:
            excluded.update({i: 'extra_identical_copy' for i in sorted(group)[1:]})
    clean = [r for r in rows if r['id_code'] not in excluded]
    by_grade = defaultdict(list)
    for r in clean:
        by_grade[r['diagnosis']].append(r)
    rng = random.Random(42)
    splits = {'train': [], 'validation': [], 'test': []}
    for grade in sorted(by_grade, key=int):
        items = by_grade[grade]
        rng.shuffle(items)
        a = int(len(items) * .70)
        b = a + int(len(items) * .15)
        for name, part in zip(splits, (items[:a], items[a:b], items[b:])):
            splits[name].extend(part)
    for name, items in splits.items():
        rng.shuffle(items)
        path = out / f'{name}_split.csv'
        if path.exists():
            backup = out / 'before_bundle_splits' / path.name
            backup.parent.mkdir(exist_ok=True)
            if not backup.exists():
                shutil.copy2(path, backup)
        with path.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['id_code', 'diagnosis'])
            w.writeheader()
            w.writerows({k: r[k] for k in w.fieldnames} for r in items)
        print(name, len(items), dict(sorted(Counter(r['diagnosis'] for r in items).items())))
    included = {r['id_code'] for r in clean}
    assert all(len(included.intersection(g)) <= 1 for g in audit['exact_duplicate_groups'])
    target = out / 'images224'
    target.mkdir(exist_ok=True)
    for n, r in enumerate(clean, 1):
        dest = target / (r['id_code'] + '.png')
        # Recreate deterministically; originals never change.
        with Image.open(raw / 'train_images' / dest.name) as im:
            padded_image(im).save(dest)
        if n % 250 == 0:
            print(f'Prepared {n}/{len(clean)}', flush=True)
    write_json(ROOT / 'results/split_cleaning.json', {
        'original_images': len(rows), 'retained_images': len(clean),
        'exclusion_counts': dict(Counter(excluded.values())),
        'excluded_image_ids': excluded, 'seed': 42})
    print('Data prepared. Freeze these splits before model tuning.')


def verify_data():
    splits = {s: read_rows(ROOT / f'data/processed/{s}_split.csv')
              for s in ('train', 'validation', 'test')}
    all_ids = [r['id_code'] for rows in splits.values() for r in rows]
    assert len(all_ids) == len(set(all_ids)), 'An ID occurs in multiple splits'
    audit = json.loads((ROOT / 'results/image_audit.json').read_text(encoding='utf-8'))
    assert not ({r['id_code'] for r in audit['unreadable_images']} & set(all_ids))
    for group in audit['exact_duplicate_groups']:
        assert len(set(group) & set(all_ids)) <= 1, 'Run cleaning: duplicates remain'
    for image_id in all_ids:
        assert (ROOT / f'data/processed/images224/{image_id}.png').is_file(), image_id
    fingerprint = hashlib.sha256()
    for name in splits:
        fingerprint.update((ROOT / f'data/processed/{name}_split.csv').read_bytes())
    print('Verified cleaned splits and prepared image paths.', flush=True)
    return splits, fingerprint.hexdigest()


def get_device(name, adapter=0):
    import torch
    if name == 'directml':
        import torch_directml
        print('GPU:', torch_directml.device_name(adapter), flush=True)
        return torch_directml.device(adapter)
    if name == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Check the NVIDIA driver and CUDA PyTorch build.')
        print('GPU:', torch.cuda.get_device_name(0), flush=True)
    return torch.device(name)


def build_model(mode, pretrained=False):
    import torch
    from torch import nn
    from torchvision.models import densenet121, DenseNet121_Weights
    model = densenet121(weights=DenseNet121_Weights.DEFAULT if pretrained else None)
    if mode == 'baseline':
        model.classifier = nn.Linear(model.classifier.in_features, 5)
    else:
        class OrderedHead(nn.Module):
            def __init__(self, features):
                super().__init__()
                self.score = nn.Linear(features, 1)
                self.base = nn.Parameter(torch.tensor(-1.5))
                self.gaps = nn.Parameter(torch.zeros(3))

            def forward(self, x):
                gaps = torch.nn.functional.softplus(self.gaps) + 1e-4
                cuts = torch.cat((self.base.reshape(1), self.base + gaps.cumsum(0)))
                return self.score(x) - cuts.reshape(1, 4)
        model.classifier = OrderedHead(model.classifier.in_features)
    return model


def probabilities(logits, mode):
    import torch
    if mode == 'baseline':
        return logits.softmax(dim=1)
    q = logits.sigmoid()  # q[k] = P(grade > k); ordered thresholds prevent crossing.
    return torch.cat((1-q[:, :1], q[:, :-1]-q[:, 1:], q[:, -1:]), dim=1)


def loss_value(logits, y, mode, weights):
    import torch
    import torch.nn.functional as F
    if mode == 'baseline':
        return F.cross_entropy(logits, y, weight=weights)
    target = (y[:, None] > torch.arange(4, device=y.device)[None, :]).float()
    each = F.binary_cross_entropy_with_logits(logits, target, reduction='none').mean(1)
    sample_weights = weights[y]
    return (each * sample_weights).sum() / sample_weights.sum()


def image_transform(training=False):
    from torchvision import transforms as T
    ops = [T.RandomHorizontalFlip()] if training else []
    return T.Compose(ops + [T.ToTensor(), T.Normalize([.485, .456, .406], [.229, .224, .225])])


class RetinaDataset:
    def __init__(self, rows, training=False):
        self.rows = rows
        self.transform = image_transform(training)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from PIL import Image
        row = self.rows[i]
        with Image.open(ROOT / f'data/processed/images224/{row["id_code"]}.png') as im:
            x = self.transform(im.convert('RGB'))
        return x, int(row['diagnosis']), row['id_code']


def metrics(y, pred):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, mean_absolute_error
    y, pred = np.asarray(y), np.asarray(pred)
    eligible = y >= 2
    kappa = float(cohen_kappa_score(y, pred, labels=list(range(5)), weights='quadratic'))
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=list(range(5)), average='macro', zero_division=0)),
            'quadratic_weighted_kappa': kappa if np.isfinite(kappa) else None,
            'mean_absolute_grade_error': float(mean_absolute_error(y, pred)),
            'undergraded_by_2plus_all': float(((y-pred) >= 2).mean()),
            'undergraded_by_2plus_grade2plus': float(((y[eligible]-pred[eligible]) >= 2).mean()) if eligible.any() else None,
            'n_images': int(len(y))}


def cpu_tree(obj):
    import torch
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: cpu_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [cpu_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(cpu_tree(v) for v in obj)
    return obj


def run_epoch(model, loader, device, mode, weights, optimizer=None, frozen=True):
    import torch
    training = optimizer is not None
    model.train(training)
    if training and frozen:
        model.features.eval()
    total, ys, ps, ids, probs = 0., [], [], [], []
    start = time.perf_counter()
    with torch.set_grad_enabled(training):
        for n, (x, y, image_ids) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_value(logits, y, mode, weights)
            if training:
                loss.backward()
                optimizer.step()
            p = probabilities(logits.detach(), mode).cpu()
            total += loss.item() * len(y)
            ys.extend(y.cpu().tolist())
            ps.extend(p.argmax(1).tolist())
            probs.extend(p.tolist())
            ids.extend(image_ids)
            if n % 25 == 0 or n == len(loader):
                print(f'  {"train" if training else "validation/eval"} batch {n}/{len(loader)} '
                      f'elapsed {time.perf_counter()-start:.0f}s', flush=True)
    return total / len(ys), metrics(ys, ps), (ids, ys, ps, probs)


def smoke(args):
    import torch
    torch.manual_seed(42)
    device = get_device(args.device, args.adapter)
    model = build_model(args.mode).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, foreach=False)
    start = time.perf_counter()
    for step in range(2):
        x = torch.randn(2, 3, 224, 224).to(device)
        y = torch.tensor([1, 3], dtype=torch.long).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_value(model(x), y, args.mode, torch.ones(5).to(device))
        loss.backward()
        optimizer.step()
        value = loss.item()
        if not torch.isfinite(torch.tensor(value)):
            raise RuntimeError('Nonfinite loss')
        print(f'Step {step+1}: loss={value:.4f}', flush=True)
    print(f'DenseNet {args.mode} forward/backward/optimizer passed in {time.perf_counter()-start:.1f}s')


def train(args):
    import torch
    from torch.utils.data import DataLoader
    random.seed(42)
    torch.manual_seed(42)
    splits, fingerprint = verify_data()
    run = ROOT / 'runs' / args.run
    last = run / 'last.pt'
    if last.exists() and not args.resume:
        raise RuntimeError('Run already exists: use --resume or a new --run name')
    if args.resume and not last.exists():
        raise FileNotFoundError(last)
    state = torch.load(last, map_location='cpu', weights_only=False) if args.resume else None
    config = {'mode': args.mode, 'finetune': args.finetune, 'lr': args.lr,
              'batch_size': args.batch_size, 'seed': 42, 'split_fingerprint': fingerprint,
              'architecture': 'densenet121', 'preprocessing': 'RGB pad224 ImageNet normalization'}
    if state:
        for key in ('mode', 'finetune', 'lr', 'batch_size', 'split_fingerprint'):
            if state['config'][key] != config[key]:
                raise ValueError(f'Resume mismatch: {key}. Use original arguments or a new run.')
    device = get_device(args.device, args.adapter)
    model = build_model(args.mode, pretrained=not bool(state))
    for p in model.features.parameters():
        p.requires_grad = args.finetune
    if state:
        model.load_state_dict(state['model'])
    model.to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=args.lr, foreach=False)
    if state:
        optimizer.load_state_dict(state['optimizer'])
        # Adam scalar step counters remain on CPU for non-capturable Adam.
        for values in optimizer.state.values():
            for key, value in values.items():
                if torch.is_tensor(value):
                    values[key] = value.cpu() if key == 'step' else value.to(device)
        torch.set_rng_state(state['torch_rng'])
        random.setstate(state['python_rng'])
        if args.device == 'cuda' and state.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    else:
        # Match data order/augmentation seeds despite different head parameter counts.
        torch.manual_seed(42)
        random.seed(42)
    counts = Counter(int(r['diagnosis']) for r in splits['train'])
    if any(counts[k] == 0 for k in range(5)):
        raise ValueError('Training requires all five grades')
    weights = torch.tensor([len(splits['train'])/(5*counts[k]) for k in range(5)]).to(device)
    loaders = {s: DataLoader(RetinaDataset(splits[s], training=s == 'train'),
                            batch_size=args.batch_size, shuffle=s == 'train', num_workers=0)
               for s in ('train', 'validation')}
    history = state['history'] if state else []
    best = state['best_macro_f1'] if state else -1.
    start_epoch = state['epoch'] + 1 if state else 1
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / 'config.json', config)
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            print(f'Epoch {epoch}/{args.epochs}', flush=True)
            begin = time.perf_counter()
            tl, tm, _ = run_epoch(model, loaders['train'], device, args.mode, weights,
                                  optimizer, not args.finetune)
            vl, vm, _ = run_epoch(model, loaders['validation'], device, args.mode, weights)
            improved = vm['macro_f1'] > best
            best = max(best, vm['macro_f1'])
            history.append({'epoch': epoch, 'train_loss': tl, 'validation_loss': vl,
                            'train_macro_f1': tm['macro_f1'], 'validation_macro_f1': vm['macro_f1'],
                            'seconds': time.perf_counter()-begin})
            checkpoint = {'config': config, 'model': cpu_tree(model.state_dict()),
                          'optimizer': cpu_tree(optimizer.state_dict()), 'epoch': epoch,
                          'history': history, 'best_macro_f1': best,
                          'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(),
                          'cuda_rng': torch.cuda.get_rng_state_all() if args.device == 'cuda' else None}
            tmp = run / 'last.tmp'
            torch.save(checkpoint, tmp)
            tmp.replace(last)
            if improved:
                shutil.copy2(last, run / 'best.pt')
            write_json(run / 'history.json', history)
            write_json(run / 'validation_metrics.json', vm)
            print(f'Saved epoch {epoch}; validation macro F1={vm["macro_f1"]:.3f}; '
                  f'epoch time={history[-1]["seconds"]:.0f}s', flush=True)
    except KeyboardInterrupt:
        print('\nStopped. Last fully completed epoch is saved; resume from last.pt.')


def load_checkpoint(path):
    import torch
    # Load only locally generated, trusted checkpoints; never accept checkpoint uploads.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = build_model(checkpoint['config']['mode'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint


def evaluate(args):
    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import classification_report, ConfusionMatrixDisplay
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    splits, fingerprint = verify_data()
    run = ROOT / 'runs' / args.run
    model, checkpoint = load_checkpoint(run / 'best.pt')
    assert checkpoint['config']['split_fingerprint'] == fingerprint, 'Splits changed since training'
    device = get_device(args.device, args.adapter)
    model.to(device)
    loader = DataLoader(RetinaDataset(splits[args.split]), batch_size=args.batch_size, num_workers=0)
    _, result, values = run_epoch(model, loader, device, checkpoint['config']['mode'], torch.ones(5).to(device))
    ids, ys, ps, probs = values
    result.update({'split': args.split, 'epoch': checkpoint['epoch'], 'mode': checkpoint['config']['mode']})
    folder = run / f'evaluation_{args.split}'
    folder.mkdir(exist_ok=True)
    write_json(folder / 'metrics.json', result)
    write_json(folder / 'classification_report.json', classification_report(
        ys, ps, labels=list(range(5)), target_names=GRADES, output_dict=True, zero_division=0))
    with (folder / 'predictions.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['id_code', 'true_grade', 'predicted_grade'] + [f'p_grade_{k}' for k in range(5)])
        w.writerows([i, y, p] + prob for i, y, p, prob in zip(ids, ys, ps, probs))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(ys, ps, labels=list(range(5)), ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'{args.run}: {args.split} set, epoch {checkpoint["epoch"]}')
    fig.tight_layout()
    fig.savefig(folder / 'confusion_matrix.png', dpi=160)
    plt.close(fig)
    history = checkpoint['history']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name in ('train', 'validation'):
        axes[0].plot([h['epoch'] for h in history], [h[f'{name}_loss'] for h in history], label=name)
        axes[1].plot([h['epoch'] for h in history], [h[f'{name}_macro_f1'] for h in history], label=name)
    for ax, title in zip(axes, ('Loss', 'Macro F1')):
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
    fig.tight_layout()
    fig.savefig(folder / 'learning_curves.png', dpi=160)
    plt.close(fig)
    print(json.dumps(result, indent=2))


def report(args):
    rows = []
    for path in sorted((ROOT / 'runs').glob(f'*/evaluation_{args.split}/metrics.json')):
        value = json.loads(path.read_text())
        value['run'] = path.parent.parent.name
        rows.append(value)
    if not rows:
        raise RuntimeError('Evaluate at least one model first')
    dest = ROOT / 'results'
    dest.mkdir(exist_ok=True)
    with (dest / f'comparison_{args.split}.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    text = ['# Diabetic retinopathy grading: preliminary research results', '',
            'Author: Mohit Kumar, 2026PAD2529', '',
            '## Data integrity',
            'Byte-identical duplicate groups were checked before splitting. Extra copies were removed; '
            'all images in conflicting-label groups were excluded pending expert review. '
            'This changes the evaluation population and may make it easier; report exclusions explicitly. '
            'Near-duplicates and patient-level overlap have not been ruled out. No patient IDs or '
            'demographic attributes were available in the supplied CSV.', '', '## Model results']
    for row in rows:
        text.extend(['', f'### {row["run"]}', '```json', json.dumps(row, indent=2), '```'])
    text.extend(['', '## Interpretation and ethics',
                 'The two-or-more-grade undergrading rate is reported over all images and separately '
                 'over true grades 2-4. This is an error proxy, not a measured patient outcome.',
                 'Report class-specific recall alongside macro F1, weighted kappa and mean grade error. '
                 'No demographic fairness claim is supported. Softmax/ordinal scores are uncalibrated. '
                 'No clinical deployment, treatment recommendation, or guarantee of screening safety is justified.',
                 'Select models using validation only. Fix the training budget before evaluating the held-out test set; '
                 'do not tune models using test results. Compare matching seeds, backbone settings and epoch budgets.',
                 'A head-only run is transfer learning with a frozen backbone. Longer fine-tuning and external '
                 'validation remain future work. Report negative or inconclusive comparisons honestly.', '',
                 '## Reference',
                 'Farag et al. (2022), IEEE Access. https://doi.org/10.1109/ACCESS.2022.3165193', '',
                 'This implementation uses DenseNet121 and an ordered cumulative-logit head; it does not '
                 'reproduce the paper\'s CBAM architecture.', '',
                 '## AI assistance disclosure',
                 'Code and workflow were drafted with Codex. Keep the chat/prompts with the project, review '
                 'the implementation, and insert actual measured results into the final submission.'])
    (dest / 'research_summary.md').write_text('\n'.join(text), encoding='utf-8')
    print('Saved results/comparison CSV and results/research_summary.md')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='Audit, clean, split and resize data from scratch')
    prep.add_argument('--reuse-audit', action='store_true', help='Reuse your existing audit for unchanged source data')
    sub.add_parser('verify', help='Check existing cleaned splits and prepared images')
    for command in ('smoke', 'train', 'evaluate'):
        q = sub.add_parser(command)
        q.add_argument('--device', choices=['cpu', 'cuda', 'directml'], default='cpu')
        q.add_argument('--adapter', type=int, default=0)
        if command != 'evaluate':
            q.add_argument('--mode', choices=['baseline', 'ordinal'], default='baseline')
        if command != 'smoke':
            q.add_argument('--run', required=True)
            q.add_argument('--batch-size', type=int, default=8)
        if command == 'train':
            q.add_argument('--epochs', type=int, default=3, help='Total target epochs, including resumed epochs')
            q.add_argument('--lr', type=float, default=.001)
            q.add_argument('--finetune', action='store_true', help='Unfreeze the full backbone; use lr .0001')
            q.add_argument('--resume', action='store_true')
        if command == 'evaluate':
            q.add_argument('--split', choices=['validation', 'test'], default='validation')
    q = sub.add_parser('report')
    q.add_argument('--split', choices=['validation', 'test'], default='validation')
    args = p.parse_args()
    if args.command in ('train', 'evaluate') and (Path(args.run).name != args.run or args.run in ('.', '..')):
        p.error('--run must be a simple folder name')
    {'prepare': prepare, 'verify': lambda a: verify_data(), 'smoke': smoke,
     'train': train, 'evaluate': evaluate, 'report': report}[args.command](args)


if __name__ == '__main__':
    main()
