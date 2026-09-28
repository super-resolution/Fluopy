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
    """
    Parse an energy-transfer transition-group label.

    Parameters
    ----------
    label
        Transition-group label.

    Returns
    -------
    tuple[str, str, str] | None
        Donor name, acceptor name, and distance if label describes an energy transfer;
        otherwise None.
    """
    match = _ENERGY_TRANSFER_LABEL.fullmatch(label)
    return None if match is None else match.groups()


def normalize_transition_frequencies(
    frequencies: npt.ArrayLike, transition_df: pd.DataFrame
) -> npt.NDArray[np.float64]:
    """
    Normalize transition frequencies separately for each donor fluorophore.

    Energy-transfer transitions are assigned to the donor's transition group.

    Parameters
    ----------
    frequencies
        Transition frequencies ordered by transition identity.
    transition_df
        Dataframe of transitions with transition-group labels and transition identities
        as its index.

    Returns
    -------
    npt.NDArray[np.float64]
        Copy of frequencies normalized separately for each donor fluorophore. Values
        remain 0 for a fluorophore whose total frequency is 0.
    """
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
    """
    Calculate relative state occupations from frequencies and mean lifetimes.

    Non-finite mean lifetimes contribute 0 to the state occupations.

    Parameters
    ----------
    frequency_states
        Relative number of visits to each state, grouped by fluorophore name.
    mean_lifetimes
        Mean lifetime of each state, grouped by fluorophore name.

    Returns
    -------
    dict[str, npt.NDArray[np.float64]]
        Relative time spent in each state, normalized separately for each fluorophore.
    """
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
