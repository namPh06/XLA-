"""Small structural check for the hand-off Colab notebook."""
import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory

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

print("PASS: notebook syntax/flow; CIRR missing, complete, corrupt, and partial-data guards")
