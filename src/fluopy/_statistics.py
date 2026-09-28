"""Shared statistical operations for simulated and predicted results."""

from __future__ import annotations

import re

import numpy as np
import numpy.typing as npt
import pandas as pd

_ENERGY_TRANSFER_LABEL = re.compile(
    r"D:\s*([^,]+),\s*A:\s*([^,]+),\s*dist:\s*(\d+(?:\.\d+)?)\s*"
)


def parse_energy_transfer_label(label: str) -> tuple[str, str, str] | None:
    match = _ENERGY_TRANSFER_LABEL.fullmatch(label)
    return None if match is None else match.groups()


def normalize_transition_frequencies(
    frequencies: npt.ArrayLike, transition_df: pd.DataFrame
) -> npt.NDArray[np.float64]:
    normalized = np.asarray(frequencies, dtype=np.float64).copy()
    transition_ids_by_donor: dict[str, list[int]] = {}
    for group_label_raw, group in transition_df.groupby(level=0, sort=False):
        group_label = str(group_label_raw)
        energy_transfer = parse_energy_transfer_label(group_label)
        donor = group_label if energy_transfer is None else energy_transfer[0]
        transition_ids_by_donor.setdefault(donor, []).extend(
            group.index.get_level_values(1).tolist()
        )

    for transition_ids in transition_ids_by_donor.values():
        total = np.sum(normalized[transition_ids])
        if total > 0:
            normalized[transition_ids] /= total

    return normalized


def calculate_state_occupations(
    frequency_states: dict[str, npt.NDArray[np.float64]],
    mean_lifetimes: dict[str, npt.NDArray[np.float64]],
) -> dict[str, npt.NDArray[np.float64]]:
    state_occupations: dict[str, npt.NDArray[np.float64]] = {}
    for fluorophore, frequencies in frequency_states.items():
        lifetimes = mean_lifetimes[fluorophore]
        occupations = np.multiply(
            frequencies,
            lifetimes,
            where=np.isfinite(lifetimes),
            out=np.zeros(frequencies.size),
        )
        total = occupations.sum()
        if total > 0:
            occupations /= total
        state_occupations[fluorophore] = occupations

    return state_occupations
