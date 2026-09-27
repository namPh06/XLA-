"""Small structural check for the hand-off Colab notebook."""
import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image


notebook_path = Path("notebooks/LinCIR_Colab.ipynb")
assert notebook_path.is_file(), f"Missing {notebook_path}"
notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
assert notebook["nbformat"] == 4

cells = notebook["cells"]
source = "\n".join("".join(cell.get("source", [])) for cell in cells)
for marker in [
    "navervision/lincir.git",
    "navervision/zeroshot-cir-models",
    "lincir_large.pt",
    "5ba98d52db7a5e9a78f9ba7436991235477b89544caf246e4659642bf9b409bd",
    "drive.mount",
    "except Exception as exc",
    "Path('/content/LinCIR')",
    "CIRR_ROOT",
    "validate.py",
    "train_phi.py",
    "--max_train_steps",
    "RUN_FULL_TRAINING",
]:
    assert marker in source, f"Notebook is missing {marker}"

assert "    └── dev/" not in source, "CIRR dev/ must be beside cirr/, not inside it"

for cell in cells:
    if cell.get("cell_type") == "code":
        ast.parse("".join(cell.get("source", [])))

readiness = next("".join(cell["source"]) for cell in cells
                 if "caption_file = CIRR_ROOT" in "".join(cell.get("source", [])))
with TemporaryDirectory() as directory:
    root = Path(directory)
    captions = root / "cirr/captions/cap.rc2.val.json"
    split = root / "cirr/image_splits/split.rc2.val.json"
    for path in (captions, split):
        path.parent.mkdir(parents=True, exist_ok=True)
    captions.write_text(json.dumps([{}] * 4181), encoding="utf-8")
    # One image shared by the fixture's entries keeps this check small and offline.
    split.write_text(json.dumps({str(i): "dev/example.png" for i in range(2297)}), encoding="utf-8")
    namespace = {"CIRR_ROOT": root, "RUN_DIR": root, "json": json}
    exec(readiness, namespace)
    assert namespace["CIRR_READY"] is False, "Missing images must clear the readiness flag"
    report = root / "cirr_readiness.json"
    assert len(json.loads(report.read_text())["missing_images"]) == 2297
    photo = root / "dev/example.png"
    photo.parent.mkdir()
    Image.new("RGB", (2, 2)).save(photo)
    exec(readiness, namespace)
    assert namespace["CIRR_READY"] is True
    photo.write_bytes(b"broken image")
    exec(readiness, namespace)
    assert namespace["CIRR_READY"] is False, "Corrupt images must stop evaluation"
    Image.new("RGB", (2, 2)).save(photo)
    captions.write_text("[]", encoding="utf-8")
    exec(readiness, namespace)
    assert namespace["CIRR_READY"] is False, "A partial query set is not a complete baseline"

    # Execute the optional test export cell, with only subprocess inference
    # replaced by deterministic predictions. Check the actual output validation
    # and Drive-copy path without requiring GPU, model downloads or CIRR images.
    test_records = [dict(pairid=i, reference='0', img_set={'members': [str(j) for j in range(6)]})
                    for i in range(4148)]
    (captions.parent / 'cap.rc2.test1.json').write_text(json.dumps(test_records), encoding='utf-8')
    (split.parent / 'split.rc2.test1.json').write_text(
        json.dumps({str(i): 'dev/example.png' for i in range(2315)}), encoding='utf-8')
    checkpoint = root / 'phi.pt'
    checkpoint.touch()
    def fake_inference(command, log_path):
        destination = root / 'submission/cirr'
        destination.mkdir(parents=True)
        for prefix, metric, k in [('', 'recall', 50), ('subset_', 'recall_subset', 3)]:
            result = dict(version='rc2', metric=metric)
            result.update({str(i): [str(j) for j in range(1, k + 1)] for i in range(4148)})
            (destination / f'{prefix}vit_l_official.json').write_text(json.dumps(result), encoding='utf-8')
    namespace.update(WORK=root, CHECKPOINT=checkpoint, sys=sys, shutil=shutil,
                     CACHE_DIR=root, run_logged=fake_inference, Path=Path)
    export = next(''.join(cell['source']) for cell in cells
                  if 'EXPORT_CIRR_TEST = False' in ''.join(cell.get('source', [])))
    exec(export.replace('EXPORT_CIRR_TEST = False', 'EXPORT_CIRR_TEST = True'), namespace)
    assert (root / 'submissions/vit_l_official/subset_vit_l_official.json').is_file()
    try:
        exec(export.replace('EXPORT_CIRR_TEST = False', 'EXPORT_CIRR_TEST = True'), namespace)
    except FileExistsError:
        pass
    else:
        raise AssertionError('Export overwrote an existing submission')

    logging_cell = next(''.join(cell['source']) for cell in cells
                        if 'def run_logged(' in ''.join(cell.get('source', [])))
    tree = ast.parse(logging_cell)
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    namespace.update(subprocess=subprocess, UPSTREAM_COMMIT='test', gpu_name='CPU test', EFFECTIVE_BATCH_SIZE=4)
    exec(compile(tree, 'notebook_logging', 'exec'), namespace)
    with patch('importlib.metadata.version', return_value='test'):
        run_logged = namespace['run_logged']
        log = root / 'success.log'
        run_logged([sys.executable, '-c', "print('completed')"], log)
        assert 'completed' in log.read_text() and json.loads(log.with_suffix('.json').read_text())['returncode'] == 0
        try:
            run_logged([sys.executable, '-c', 'raise SystemExit(7)'], root / 'failure.log')
        except RuntimeError:
            pass
        else:
            raise AssertionError('Subprocess failure was hidden')
        assert json.loads((root / 'failure.json').read_text())['returncode'] == 7
        try:
            run_logged([sys.executable, '-c', "print('overwritten')"], log)
        except FileExistsError:
            pass
        else:
            raise AssertionError('Experiment log was overwritten')

print("PASS: notebook syntax/flow, CIRR guards, test export, streamed logs, subprocess failure and overwrite guards")
