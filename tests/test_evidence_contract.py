import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (ROOT / "foundations" / "mnist_numpy" / "src", ROOT / "gpt" / "src")


def source_paths() -> list[Path]:
    return [path for root in SOURCE_ROOTS for path in sorted(root.glob("*.py"))]


def active_not_implemented_raises(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        target = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
        if isinstance(target, ast.Name) and target.id == "NotImplementedError":
            lines.append(node.lineno)
    return lines


def test_required_evidence_and_completed_sources():
    active_stubs = {
        str(path.relative_to(ROOT)): active_not_implemented_raises(path)
        for path in source_paths()
        if active_not_implemented_raises(path)
    }
    assert active_stubs == {}

    markers = ("TODO:", "!TODO", "# raise NotImplementedError")
    assignment_markers = {
        str(path.relative_to(ROOT)): [
            marker
            for marker in markers
            if marker in path.read_text(encoding="utf-8")
        ]
        for path in source_paths()
        if any(marker in path.read_text(encoding="utf-8") for marker in markers)
    }
    assert assignment_markers == {}

    required = [
        "ATTRIBUTION.md",
        "docs/architecture.md",
        "docs/results.md",
        "docs/contribution-map.md",
        "artifacts/current/smoke-result.json",
        "artifacts/pretraining/run_config.json",
        "artifacts/pretraining/summary_by_batch_size.md",
        "artifacts/pretraining/loss_comparison_val.png",
    ]
    missing = [path for path in required if not (ROOT / path).is_file()]
    assert missing == []


def test_attribution_names_sources_and_license_boundary():
    attribution = (ROOT / "ATTRIBUTION.md").read_text(encoding="utf-8")

    assert "https://github.com/Soldbone/gpt-lab" in attribution
    assert "https://github.com/devhyun05/group4-mnist-lab" in attribution
    assert "별도 LICENSE가 없으므로" in attribution


def test_results_separate_historical_and_current_evidence():
    results = (ROOT / "docs" / "results.md").read_text(encoding="utf-8")

    assert "Historical result" in results
    assert "Current reproduction" in results
    assert "4fe533e" in results
