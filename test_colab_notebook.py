"""Small structural check for the hand-off Colab notebook."""
import ast
import json
from pathlib import Path


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

print("PASS: notebook JSON, Python syntax, setup/checkpoint/eval/train flow")
