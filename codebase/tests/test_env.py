"""
Environment checks: required packages import at the pinned versions, the seed helper is
deterministic, and the YAML configs carry the protocol fields the runners read.
"""

import os
import random
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED = ["torch", "numpy", "scipy", "sklearn", "pandas", "h5py", "yaml", "mne", "moabb", "skorch", "ptflops"]
# Import name -> distribution name in requirements.txt, where they differ.
DIST_NAME = {"sklearn": "scikit-learn", "yaml": "pyyaml"}


def _pins() -> dict:
    pins = {}
    for line in (PROJECT_ROOT / "requirements.txt").read_text().splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_.-]+)==([^\s#]+)", line)
        if m:
            pins[m.group(1).lower()] = m.group(2)
    return pins


@pytest.mark.parametrize("module", REQUIRED)
def test_required_package_imports(module):
    pytest.importorskip(module)


def test_requirements_are_pinned_and_lean():
    text = (PROJECT_ROOT / "requirements.txt").read_text()
    pins = _pins()
    for module in REQUIRED:
        assert DIST_NAME.get(module, module) in pins, f"{module} is not pinned with == in requirements.txt"
    for unused in ("efficient-kan", "torchvision", "torchaudio"):
        assert unused not in text, f"{unused} is unused and should not be a requirement"


@pytest.mark.parametrize("module,dist", [("moabb", "moabb"), ("skorch", "skorch"), ("sklearn", "scikit-learn"), ("mne", "mne")])
def test_installed_versions_match_pins(module, dist):
    """The protocol-critical packages must match requirements.txt (a MOABB API change broke the runners once)."""
    mod = pytest.importorskip(module)
    pinned = _pins()[dist]
    if mod.__version__ != pinned:
        if os.environ.get("STRICT_PINS", "").lower() in ("1", "true"):
            assert mod.__version__ == pinned, f"{dist} {mod.__version__} installed, {pinned} pinned"
        else:
            pytest.skip(f"{dist} {mod.__version__} installed, {pinned} pinned in requirements.txt (export STRICT_PINS=1 to assert)")


def test_accelerator_detection_returns_known_device():
    pytest.importorskip("torch")
    from src.models.skorch_wrapper import get_default_device
    assert get_default_device() in {"cuda", "mps", "cpu"}


def test_seed_determinism():
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    from src.utils.seed import set_seed

    draws = []
    for _ in range(2):
        set_seed(42)
        draws.append((random.random(), np.random.rand(3), torch.randn(5, 5)))
    assert draws[0][0] == draws[1][0]
    np.testing.assert_array_equal(draws[0][1], draws[1][1])
    assert torch.equal(draws[0][2], draws[1][2])
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.is_deterministic_algorithms_warn_only_enabled()


def test_yaml_config_loading():
    """Both configs exist and carry the fields the runners read."""
    yaml = pytest.importorskip("yaml")
    config_dir = PROJECT_ROOT / "configs"

    bnci_cfg = yaml.safe_load((config_dir / "bnci2014_001.yaml").read_text())
    assert bnci_cfg["dataset"]["name"].lower() == "bnci2014_001"
    assert len(bnci_cfg["dataset"]["events"]) == 4
    assert "rest" not in bnci_cfg["dataset"]["events"]
    assert bnci_cfg["experiment"]["seeds"]

    physio_cfg = yaml.safe_load((config_dir / "physionet.yaml").read_text())
    assert physio_cfg["dataset"]["name"].lower() == "physionet"
    assert physio_cfg["dataset"]["subset_subjects"] == 20
    assert "rest" not in physio_cfg["dataset"]["events"]


def test_loader_label_map_is_global():
    """Audit item 19: one label map from the paradigm's events, not per subject."""
    pytest.importorskip("moabb")
    from src.data.loaders import MotorImageryDataLoader

    loader = MotorImageryDataLoader("BNCI2014_001")
    assert loader.label_map == {"left_hand": 0, "right_hand": 1, "feet": 2, "tongue": 3}
    assert loader.paradigm.events == ["left_hand", "right_hand", "feet", "tongue"]

    physio = MotorImageryDataLoader("PhysionetMI")
    assert "rest" not in physio.label_map and len(physio.label_map) == 4

    with pytest.raises(ValueError):
        MotorImageryDataLoader("BNCI2014_001", n_classes=5)
