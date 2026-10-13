import argparse
import sys
import yaml
from pathlib import Path


def prefetch_bnci2014_001():
    """
    Download and locally cache the BNCI2014_001 motor imagery dataset via MOABB.
    """
    try:
        from moabb.datasets import BNCI2014_001
    except ImportError:
        print("[ERROR] MOABB is not installed in the environment. Please run: pip install moabb")
        sys.exit(1)

    print("[EEG-KAN] Initializing BNCI2014_001 dataset pre-fetch...")
    dataset = BNCI2014_001()
    subjects = dataset.subject_list
    print(f"[EEG-KAN] Found {len(subjects)} subjects in BNCI2014_001: {subjects}")

    # Download data for all subjects into local MNE cache
    data = dataset.get_data(subjects=subjects)
    print(f"[EEG-KAN] Successfully cached BNCI2014_001 data for all {len(subjects)} subjects!")
    return data


def prefetch_physionet(subset_size: int = 20):
    """
    Download and locally cache a subset of the PhysionetMI dataset via MOABB.
    """
    try:
        from moabb.datasets import PhysionetMI
    except ImportError:
        print("[ERROR] MOABB is not installed in the environment. Please run: pip install moabb")
        sys.exit(1)

    print(f"[EEG-KAN] Initializing PhysionetMI dataset pre-fetch (subset size: {subset_size})...")
    dataset = PhysionetMI()
    subjects = dataset.subject_list[:subset_size]
    print(f"[EEG-KAN] Selected {len(subjects)} subjects for pre-fetching: {subjects}")

    data = dataset.get_data(subjects=subjects)
    print(f"[EEG-KAN] Successfully cached PhysionetMI data for {len(subjects)} subjects!")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-fetch and cache MOABB EEG datasets locally.")
    parser.add_argument("--dataset", type=str, default="bnci2014_001", choices=["bnci2014_001", "physionet", "all"])
    parser.add_argument("--physionet_subset", type=int, default=20)
    args = parser.parse_args()

    if args.dataset in ["bnci2014_001", "all"]:
        prefetch_bnci2014_001()
    if args.dataset in ["physionet", "all"]:
        prefetch_physionet(subset_size=args.physionet_subset)
