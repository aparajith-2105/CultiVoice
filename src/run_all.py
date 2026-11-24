"""Run the whole pipeline in order: EDA -> models -> export -> verification.
Usage (from the project root):  python src/run_all.py
Outputs: results/ (charts, tables), models/ (.joblib), frontend/model/ (JSON + predict.js).
verify_export.py needs Node.js; it is skipped with a message if node is not installed."""
import shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
steps = ["eda.py", "train_crop.py", "train_yield.py", "train_fert.py", "export_models.py", "verify_export.py"]
for s in steps:
    if s == "verify_export.py" and shutil.which("node") is None:
        print("Skipping verify_export.py (Node.js not found). Install Node.js to check the JS predictor.")
        continue
    print(f"\n=== {s} ===", flush=True)
    r = subprocess.run([sys.executable, str(HERE / s)], cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"{s} failed")
print("\nAll steps finished.")
