"""
Data Ingestion and Signal Filtering Module using MOABB & MNE
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import mne
    from moabb.datasets import BNCI2014_001, PhysionetMI
    from moabb.paradigms import LeftRightImagery, MotorImagery
except ImportError:
    mne = None
    BNCI2014_001 = None
    PhysionetMI = None
    MotorImagery = None


class MotorImageryDataLoader:
    """
    Data ingestion pipeline for Motor Imagery EEG benchmarks.
    Extracts epoched EEG signals with bandpass filtering and label encoding.
    """

    def __init__(
        self,
        dataset_name: str = "BNCI2014_001",
        fmin: float = 8.0,
        fmax: float = 30.0,
        resample_rate: Optional[float] = None,
        n_classes: int = 4,
    ):
        """
        Args:
            dataset_name (str): Name of dataset ('BNCI2014_001' or 'PhysionetMI').
            fmin (float): Lower frequency cutoff for bandpass filter (default: 8.0 Hz for mu rhythm).
            fmax (float): Upper frequency cutoff for bandpass filter (default: 30.0 Hz for beta rhythm).
            resample_rate (float, optional): Optional resampling frequency in Hz.
            n_classes (int): Number of motor imagery classes (default: 4).
        """
        self.dataset_name = dataset_name
        self.fmin = fmin
        self.fmax = fmax
        self.resample_rate = resample_rate
        self.n_classes = n_classes

        self.dataset = self._init_dataset(dataset_name)
        self.paradigm = self._init_paradigm(fmin, fmax, resample_rate, n_classes)

    def _init_dataset(self, name: str):
        if name.upper() == "BNCI2014_001":
            if BNCI2014_001 is None:
                raise ImportError("MOABB is required. Please run: pip install moabb")
            return BNCI2014_001()
        elif name.upper() in ["PHYSIONET", "PHYSIONETMI"]:
            if PhysionetMI is None:
                raise ImportError("MOABB is required. Please run: pip install moabb")
            return PhysionetMI()
        else:
            raise ValueError(f"Unsupported dataset name: {name}")

    def _init_paradigm(
        self, fmin: float, fmax: float, resample: Optional[float], n_classes: int
    ):
        if MotorImagery is None:
            return None
        # Same explicit event list as the runners (never ``rest``), so every subject shares
        # one label map instead of one built from whichever classes that subject happens to have.
        from src.utils.experiment import EVENTS, canonical_dataset_name

        events = list(EVENTS[canonical_dataset_name(self.dataset_name)])
        if n_classes != len(events):
            raise ValueError(
                f"{self.dataset_name} defines {len(events)} motor-imagery classes {events}; got n_classes={n_classes}"
            )
        self.events = events
        self.label_map = {lbl: idx for idx, lbl in enumerate(events)}
        return MotorImagery(
            events=events,
            fmin=fmin,
            fmax=fmax,
            n_classes=n_classes,
            resample=resample,
        )

    def get_subject_data(
        self, subject_id: int
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, str]]:
        """
        Load and pre-process epoched EEG data for a single subject.

        Args:
            subject_id (int): Subject ID integer (e.g. 1 to 9 for BNCI2014_001).

        Returns:
            X (np.ndarray): Epoched EEG tensor of shape (N_trials, 1, Channels, Time).
            y (np.ndarray): Integer class labels of shape (N_trials,).
            metadata (Dict): Metadata dict containing channel names, sampling rate, subject ID.
        """
        if self.paradigm is None:
            raise RuntimeError("MOABB MotorImagery paradigm failed to initialize.")

        X_raw, y_raw, metadata_df = self.paradigm.get_data(
            dataset=self.dataset, subjects=[subject_id]
        )

        # Map string labels with the dataset-wide label map built from the paradigm's event list,
        # so a subject missing a class does not shift every other class index.
        label_map = self.label_map
        unknown = sorted(set(np.unique(y_raw)) - set(label_map))
        if unknown:
            raise ValueError(f"Labels {unknown} are not in the paradigm's events {self.events}")
        y = np.array([label_map[lbl] for lbl in y_raw], dtype=np.int64)

        # Enforce tensor shape contract: (N_trials, 1, Channels, Time Samples)
        # MOABB default return shape: (N_trials, Channels, Time) -> expand channel dim
        if X_raw.ndim == 3:
            X = np.expand_dims(X_raw, axis=1).astype(np.float32)
        else:
            X = X_raw.astype(np.float32)

        meta = {
            "subject_id": subject_id,
            "n_trials": len(y),
            "channels": X.shape[2],
            "time_samples": X.shape[3],
            "label_map": label_map,
        }

        return X, y, meta

    def get_loso_splits(self, subjects: Optional[List[int]] = None):
        """
        Generator producing Leave-One-Subject-Out (LOSO) train/test subject splits.

        Yields:
            train_subjects (List[int]): Subject IDs for training fold.
            test_subject (int): Subject ID for unseen test fold.
        """
        all_subjects = subjects or self.dataset.subject_list
        for i, test_sub in enumerate(all_subjects):
            train_subs = [s for s in all_subjects if s != test_sub]
            yield train_subs, test_sub
