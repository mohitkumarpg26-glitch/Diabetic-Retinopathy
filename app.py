"""Run: python -I -m streamlit run app.py"""
from pathlib import Path
import sys
import json
import threading

import pandas as pd
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from retina_project import GRADES, padded_image, read_rows
from decision_adjustment import adjust_scores

ADJUSTMENT_DIR = ROOT / 'results/decoding_improvement'
adjustment = json.loads((ADJUSTMENT_DIR / 'frozen_selection.json').read_text()) if (ADJUSTMENT_DIR / 'frozen_selection.json').exists() else None
adjusted_results = json.loads((ADJUSTMENT_DIR / 'test_comparison.json').read_text()) if (ADJUSTMENT_DIR / 'test_comparison.json').exists() else None

st.set_page_config(page_title='Retina | Research Studio', page_icon='👁️', layout='wide')
st.title('Retina research studio')
st.caption('Five-grade retinal image analysis · Mohit Kumar & Malhaar Sinha · NSUT')

with st.sidebar:
    st.header('Research workspace')
    st.write('DenseNet121 · five ordered grades')
    checkpoints = sorted((ROOT / 'runs').glob('*/best.pt'))
    choices = {p.parent.name: p for p in checkpoints}
    run_names = list(choices)
    selected = st.selectbox('Model checkpoint', run_names, index=run_names.index('baseline_finetune') if 'baseline_finetune' in run_names else 0) if choices else None
    adjusted = st.radio('Prediction method', ['Adjusted baseline', 'Original baseline'], help='Adjustments were selected using validation predictions; model weights are unchanged.') == 'Adjusted baseline' if selected == 'baseline_finetune' and adjustment else False
    st.divider()
    st.caption('The best checkpoint is selected by validation macro F1.')
    if st.button('Refresh results'):
        st.cache_resource.clear()
        st.rerun()

if adjusted_results:
    st.subheader('Baseline improvement on the saved test sample')
    before, after = adjusted_results['raw'], adjusted_results['selected_test']
    cards = st.columns(4)
    cards[0].metric('Adjusted accuracy', f'{after["accuracy"]:.2%}', f'{100*(after["accuracy"]-before["accuracy"]):+.2f} pp', border=True)
    cards[1].metric('Weighted kappa', f'{after["qwk"]:.4f}', f'{after["qwk"]-before["qwk"]:+.4f}', border=True)
    cards[2].metric('Macro F1', f'{after["macro_f1"]:.4f}', f'{after["macro_f1"]-before["macro_f1"]:+.4f}', border=True)
    cards[3].metric('Severe undergrading', f'{after["severe_undergrading"]:.2%}', f'{100*(after["severe_undergrading"]-before["severe_undergrading"]):+.2f} pp', delta_color='inverse', border=True)
    st.caption('Exploratory comparison · 531 test images · severe undergrading denominator: 209 images with true grades 2–4. The test set was examined earlier; these results do not establish clinical benefit.')
    st.divider()


@st.cache_resource
def cached_model(path, modified_ns):
    from retina_project import load_checkpoint
    import torch
    torch.set_num_threads(2)
    model, checkpoint = load_checkpoint(path)
    return model, checkpoint, threading.Lock()


predict_tab, results_tab, data_tab = st.tabs(['Explore an image', 'Model results', 'Data & ethics'])

with predict_tab:
    left, right = st.columns([1, 1])
    with left:
        st.subheader('1 · Choose a retinal image')
        upload = st.file_uploader('PNG or JPEG fundus image', type=['png', 'jpg', 'jpeg'])
        st.caption('The image is processed in memory and is not saved by this app. Use de-identified research images.')
        image = None
        if upload:
            try:
                with Image.open(upload) as original:
                    image = padded_image(original)
                st.image(image, caption='Model input · 224 × 224 · proportions preserved', width=320,
                         alt='Uploaded retinal image padded to the model input size')
            except Exception:
                st.error('This file could not be read as an image.')
        pressed = st.button('Estimate research grade', type='primary', icon=':material/visibility:', width='stretch', disabled=image is None or selected is None)
    with right:
        st.subheader('2 · Review the estimate')
        st.caption('Active method: adjusted baseline' if adjusted else 'Active method: original model decoding')
        if selected is None:
            st.warning('No trained checkpoint yet. Complete a training epoch to enable predictions.')
        elif not pressed:
            st.write('Choose an image and run the estimate to view model scores.')
        if pressed and image is not None:
            try:
                import torch
                from retina_project import image_transform, probabilities
                path = choices[selected]
                with st.spinner('Running model on CPU…'):
                    model, checkpoint, lock = cached_model(str(path), path.stat().st_mtime_ns)
                    with lock, torch.inference_mode():
                        scores = probabilities(model(image_transform()(image).unsqueeze(0)),
                                               checkpoint['config']['mode'])[0].tolist()
                raw_scores = list(scores)
                raw_grade = max(range(5), key=lambda k: raw_scores[k])
                if adjusted:
                    scores = adjust_scores(scores, adjustment).tolist()
                grade = max(range(5), key=lambda k: scores[k])
                st.metric('Predicted grade', f'{grade} · {GRADES[grade]}')
                if adjusted:
                    st.caption(f'Original baseline: grade {raw_grade} · Adjusted baseline: grade {grade}')
                chart_data = {'Original score': raw_scores, 'Adjusted score': scores} if adjusted else {'Model score': scores}
                st.bar_chart(pd.DataFrame(chart_data, index=GRADES),
                             alt='Uncalibrated model scores for the five severity grades')
                st.caption('Scores are uncalibrated model outputs, not a reliable estimate of diagnostic confidence. '
                           'The app does not verify that the uploaded image is a usable retinal photograph.')
                st.write(f'Checkpoint: **{selected}** · epoch **{checkpoint["epoch"]}** · '
                         f'**{checkpoint["config"]["mode"]}**')
                st.download_button('Download this research result', json.dumps({
                    'predicted_grade': grade, 'grade_name': GRADES[grade], 'scores': scores,
                    'raw_scores': raw_scores, 'raw_grade': raw_grade,
                    'decision_rule': 'class_logit_offsets' if adjusted else 'raw_argmax',
                    'decision_offsets': adjustment['params'] if adjusted else None,
                    'run': selected, 'epoch': checkpoint['epoch'],
                    'status': 'Experimental; uncalibrated; not for clinical use'}, indent=2),
                    file_name='research_prediction.json', mime='application/json')
            except Exception as error:
                st.error(f'Prediction could not run: {error}')

with results_tab:
    st.subheader('Measured performance')
    evaluation_split = st.radio('Evaluation set', ['validation', 'test'], horizontal=True)
    if adjusted_results and evaluation_split == 'test':
        with st.container(border=True):
            st.markdown('**Original vs adjusted baseline**')
            display_rows = []
            for label, key in [('Original baseline', 'raw'), ('Adjusted baseline', 'selected_test')]:
                value = adjusted_results[key]
                display_rows.append({'Method': label, 'Accuracy (%)': 100*value['accuracy'], 'Macro F1': value['macro_f1'], 'QWK': value['qwk'], 'Mean grade error': value['mae'], 'Severe undergrading (%)': 100*value['severe_undergrading']})
            st.dataframe(pd.DataFrame(display_rows).set_index('Method').round(4), alt='Original and adjusted baseline test performance')
            st.download_button('Download adjusted evaluation', json.dumps(adjusted_results, indent=2), 'adjusted_baseline_evaluation.json', 'application/json')
    elif adjustment and evaluation_split == 'validation':
        with st.expander('Adjustment selection on validation data'):
            st.caption('Selected by five-fold out-of-fold validation QWK. The checkpoint had already been selected on this validation set, so this is exploratory rather than independent nested validation.')
            st.dataframe(pd.DataFrame({name: value['oof'] for name, value in adjustment['candidates'].items()}).T, alt='Cross-validated decision rule comparison')
    all_results = []
    for p in sorted((ROOT / 'runs').glob(f'*/evaluation_{evaluation_split}/metrics.json')):
        result = json.loads(p.read_text())
        result['run'] = p.parent.parent.name
        all_results.append(result)
    if not all_results:
        st.info(f'No {evaluation_split} evaluation saved yet. Run the evaluate command from the guide.')
    else:
        comparison = pd.DataFrame(all_results).set_index('run')
        st.dataframe(comparison, width='stretch', alt='Measured model evaluation metrics by run')
        st.download_button('Download metrics CSV', comparison.to_csv(), 'model_metrics.csv', 'text/csv')
        st.caption('Lower undergrading and grade error are better. Higher macro F1 and weighted kappa are better. '
                   'The all-image undergrading rate includes grades 0–1, which cannot be undergraded by two levels.')
    if selected:
        run = ROOT / 'runs' / selected
        history_path = run / 'history.json'
        if history_path.exists():
            history = pd.DataFrame(json.loads(history_path.read_text())).set_index('epoch')
            a, b = st.columns(2)
            with a:
                st.write('Training and validation loss')
                st.line_chart(history[['train_loss', 'validation_loss']], alt='Training and validation loss by epoch')
            with b:
                st.write('Macro F1 by epoch')
                st.line_chart(history[['train_macro_f1', 'validation_macro_f1']], alt='Training and validation macro F1 by epoch')
        matrix = run / f'evaluation_{evaluation_split}/confusion_matrix.png'
        if matrix.exists():
            st.image(str(matrix), caption='Actual grade vs predicted grade', width=650,
                     alt='Confusion matrix of actual and predicted severity grades')
    st.caption('Early runs are preliminary. Use validation for model selection and reserve the test set for the fixed final comparison.')

with data_tab:
    st.subheader('Data provenance and quality')
    cleaning_path = ROOT / 'results/split_cleaning.json'
    if cleaning_path.exists():
        cleaning = json.loads(cleaning_path.read_text())
        a, b, c = st.columns(3)
        a.metric('Original labeled images', cleaning['original_images'])
        b.metric('Retained images', cleaning['retained_images'])
        c.metric('Excluded images', cleaning['original_images'] - cleaning['retained_images'])
        st.write('Recorded exclusions', cleaning['exclusion_counts'])
    records = []
    for split in ('train', 'validation', 'test'):
        path = ROOT / f'data/processed/{split}_split.csv'
        if path.exists():
            records.extend({'split': split, 'grade': int(row['diagnosis'])} for row in read_rows(path))
    if records:
        counts = pd.crosstab(pd.DataFrame(records)['grade'], pd.DataFrame(records)['split'])
        st.bar_chart(counts, alt='Image counts by grade and dataset split')
        st.dataframe(counts, width='stretch', alt='Exact image counts by severity grade and split')
    st.markdown('''
**Safety question:** Does accounting for ordered grades reduce undergrading by two or more levels?

**Data integrity:** Exact duplicate files were audited. Extra identical copies were removed and conflicting-label
groups excluded pending expert review. Those exclusions affect the population evaluated. Similar images and
patient-level overlap have not been fully audited.

**Fairness:** Class-specific performance can be measured. Demographic fairness cannot be established from the
available image-ID and diagnosis columns.

**Limits:** Single-dataset results, small minority classes, 224-pixel images, possible label noise, uncalibrated
scores, and no external or clinical validation. A low predicted grade cannot rule out disease.

**Accountability:** Review failures, document AI assistance, and retain clinician oversight in any future research.
''')
    st.link_button('APTOS dataset source', 'https://www.kaggle.com/competitions/aptos2019-blindness-detection/data')
    st.link_button('Reference paper · IEEE Access', 'https://doi.org/10.1109/ACCESS.2022.3165193')
