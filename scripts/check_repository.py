"""Dataset-free integrity checks for the research hand-off repository."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    ROOT / "README.md",
    ROOT / "docs" / "report" / "KESSOUAR_rapport.pdf",
    ROOT / "docs" / "RESULTS.md",
    ROOT / "docs" / "AUDIT.md",
    ROOT / "src" / "model.py",
    ROOT / "src" / "model_tdbn.py",
    ROOT / "experiments" / "mad" / "04_train_chain" / "eval_test_tdbn.py",
)
IGNORED_PARTS = {".git", ".venv", "venv", "__pycache__"}
FORBIDDEN_PATHS = (
    "/users/abdekess61",
    "/data/abdekess61",
    r"C:\Users\Raouf",
)
MAINTAINED_ROOTS = (
    ROOT / "src",
    ROOT / "experiments" / "mad" / "common",
    ROOT / "experiments" / "mad" / "04_train_chain",
    ROOT / "experiments" / "mad" / "06_fall_detect",
    ROOT / "experiments" / "dvsgc_overlap",
)
MAINTAINED_FILES = (
    ROOT / "experiments" / "dvsgc_standard" / "train.py",
    ROOT / "experiments" / "dvsgc_standard" / "test.py",
    ROOT / "experiments" / "dvsgc_standard" / "val.py",
)


def python_files():
    for path in ROOT.rglob("*.py"):
        if not any(part in IGNORED_PARTS for part in path.parts):
            yield path


def verify_syntax(errors):
    for path in python_files():
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            errors.append(f"syntaxe invalide: {path.relative_to(ROOT)}: {exc}")


def verify_hashes(errors):
    manifest = ROOT / "artifacts" / "SHA256SUMS"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"artefact absent: {relative}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            errors.append(f"SHA-256 incorrect: {relative}")


def verify_portability(errors):
    paths = list(MAINTAINED_FILES)
    for base in MAINTAINED_ROOTS:
        paths.extend(base.rglob("*.py"))
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_PATHS:
            if forbidden in text:
                errors.append(
                    f"chemin personnel dans un script maintenu: {path.relative_to(ROOT)}"
                )
                break


def verify_evaluation_protocol(errors):
    for path in (
        ROOT / "experiments" / "mad" / "04_train_chain" / "eval_test_tdbn.py",
        ROOT / "experiments" / "mad" / "04_train_chain" / "eval_test_tdbn_L4.py",
    ):
        text = path.read_text(encoding="utf-8")
        for disallowed in (
            "recalibrate_running_stats(model, test_loader",
            "update_bntt_running_stats(model, test_loader",
        ):
            if disallowed in text:
                errors.append(f"adaptation sur le test interdite: {path.relative_to(ROOT)}")

    for path in (ROOT / "experiments" / "mad" / "04_train_chain").glob("train*.py"):
        text = path.read_text(encoding="utf-8")
        if "update_bntt_running_stats(model, val_loader" in text:
            errors.append(f"recalibration sur validation: {path.relative_to(ROOT)}")


def main():
    errors = []
    for path in REQUIRED:
        if not path.exists():
            errors.append(f"fichier requis absent: {path.relative_to(ROOT)}")

    verify_syntax(errors)
    verify_hashes(errors)
    verify_portability(errors)
    verify_evaluation_protocol(errors)

    if errors:
        print("ÉCHEC")
        for error in errors:
            print(f"- {error}")
        return 1

    print("OK: syntaxe, artefacts, portabilité et protocole vérifiés.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
