"""
Photophysical constants for specific fluorophores.

This module provides a dataclass container to hold photophysical constants.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

__all__: list[str] = ["Spectrum", "FluorophoreData", "cy5_dna", "atto643"]


@dataclass
class Spectrum:
    wavelengths: npt.ArrayLike
    values: npt.ArrayLike

    def __post_init__(self) -> None:
        self.wavelengths = np.asarray(self.wavelengths, dtype=float).copy()
        self.values = np.asarray(self.values, dtype=float).copy()

        if self.wavelengths.ndim != 1:
            raise ValueError("spectrum wavelengths must be one-dimensional.")
        if self.values.ndim != 1:
            raise ValueError("spectrum values must be one-dimensional.")
        if self.wavelengths.size != self.values.size:
            raise ValueError(
                "spectrum wavelengths and values must have the same length."
            )
        if self.wavelengths.size < 2:
            raise ValueError("a spectrum must contain at least two data points.")
        if not np.all(np.isfinite(self.wavelengths)):
            raise ValueError("spectrum wavelengths must be finite.")
        if not np.all(np.isfinite(self.values)):
            raise ValueError("spectrum values must be finite.")
        if not np.all(np.diff(self.wavelengths) > 0):
            raise ValueError("spectrum wavelengths must be strictly increasing.")
        if np.any(self.values < 0):
            raise ValueError("spectrum values must be non-negative.")


@dataclass
class FluorophoreData:
    """
    Container for all constant photophysical attributes of a fluorophore.
    The naming of constants is closely related to TransitionType.

    Attributes
    ----------
    data_files : str | Path | None
        The name of the folder containing the spectra data files. The folder should be
        located in src/fluopy/fluorophore_spectra. Needed to infer excitation rate
        and energy transfer rates. If None, no automatic inference of rates will be
        performed.
    QUANTUM_YIELD : float
        The fluorescence quantum yield of the fluorophore. Should be between 0 and 1.
    FLUORESCENCE_LIFETIME : float
        The fluorescence lifetime of the fluorophore in s.
    ISC_ST_RATE : float
        The intersystem crossing rate from S1 to T1 in 1/s.
    ISC_TS_RATE : float
        The intersystem crossing rate from T1 to S0 in 1/s.
    RISC_RATE : float
        The reverse intersystem crossing rate from T1 to S1 in 1/s.
    STA_EFFICIENCY : float
        The efficiency of STA (singlet-triplet annihilation) resulting in an effective
        transition of the acceptor state: S1|T1 -> S0|T2 -> S0|S1. The step in between
        (S0|T2) is not explicitly modeled, but the overall efficiency of the process is
        captured in this constant. Should be between 0 and 1.
    PHOTOBLEACH_T1_RATE : float
        The photobleaching rate from T1 to B in 1/s.
    CROSS_SECTION_WAVELENGTH : int | None
        The wavelength in nm at which absorption cross sections are defined. The
        standard excitation of S0 is handled via data_files (entire absorption
        spectrum), but for other transitions (e.g., cis absorption to define
        photoinduced back-isomerization), a single cross section should be provided.
        The cross_section_wavelength is used to check whether the provided cross
        sections are given for the same wavelength as a specified wavelength.
    DSTORM_PET_T_RATE_MOL : float
        The concentration-dependent PET rate that targets T1 in 1/(M*s).
    DSTORM_PET_S_RATE_MOL : float
        The concentration-dependent PET rate that targets S1 in 1/(M*s).
    DSTORM_PET_SUCCESS_RATE : float
        The efficiency of PET resulting in the long-living OFF state in dSTORM.
        Should be between 0 and 1.
    DSTORM_TH_EL_RATE_1 : float
        The rate of thermal elimination, returning OFF to S0 in 1/s.
    DSTORM_P_EL_CROSS_SECTION : float
        The cross section of the photoinduced uncaging, returning OFF to S0 in
        cm^2.
    RAD_ESCAPE_EFFICIENCY: float
        The efficiency of radical escape, resulting in the radical anion following
        PET. Should be between 0 and 1.
    RAD_RELAX_RATE: float
        The rate of relaxation of the radical anion back to S0 in 1/s.
    OFRET_EFFICIENCY: float
        The efficiency of OET (FRET to OFF state) resulting in an effective transition
        of the acceptor state: S1|OFF -> S0|OFF* -> S0|S0. The step in between (S0|OFF*)
        is not explicitly modeled, but the overall efficiency of the process is captured
        in this constant. Should be between 0 and 1.
    ISO_RATE: float
        The rate of trans S1 to cis isomerization in 1/s.
    BISO_CROSS_SECTION: float
        The cross section of the photoinduced back-isomerization, returning cis to S0 in
        cm^2.
    BISO_THERMAL_RATE: float
        The rate of thermal back-isomerization, returning cis to S0 in 1/s.
    BISO_EFFICIENCY: float
        The efficiency of CET (FRET to cis state) resulting in an effective transition
        of the acceptor state: S1|cis -> S0|cis* -> S0|S0. The step in between (S0|cis*)
        is not explicitly modeled, but the overall efficiency of the process is captured
        in this constant. Should be between 0 and 1.
    """

    # spectra
    data_files: str | Path | None = None

    # general
    QUANTUM_YIELD: float = 0
    FLUORESCENCE_LIFETIME: float = 0
    S1_QUENCH_RATE: float = 0
    ISC_ST_RATE: float = 0
    ISC_TS_RATE: float = 0
    RISC_RATE: float = 0
    STA_EFFICIENCY: float = 0
    PHOTOBLEACH_T1_RATE: float = 0
    CROSS_SECTION_WAVELENGTH: int | None = None

    # dstorm
    DSTORM_PET_T_RATE_MOL: float = 0
    DSTORM_PET_S_RATE_MOL: float = 0
    DSTORM_PET_SUCCESS_RATE: float = 0
    DSTORM_TH_EL_RATE_1: float = 0
    DSTORM_P_EL_CROSS_SECTION: float = 0
    RAD_ESCAPE_EFFICIENCY: float = 0
    RAD_RELAX_RATE: float = 0
    OFRET_EFFICIENCY: float = 0

    # cis trans isomerization
    ISO_RATE: float = 0
    BISO_CROSS_SECTION: float = 0
    BISO_THERMAL_RATE: float = 0
    BISO_EFFICIENCY: float = 0

    # rhodamines
    H2O_ATTACK_S: float = 0
    H2O_ATTACK_T: float = 0
    BACK_REACTION: float = 0


cy5_dna = FluorophoreData(
    data_files="cy5_data",
    QUANTUM_YIELD=0.27,
    FLUORESCENCE_LIFETIME=1.7e-9,
    ISC_ST_RATE=8.3e5,
    ISC_TS_RATE=5e3,
    RISC_RATE=0,
    STA_EFFICIENCY=0,
    PHOTOBLEACH_T1_RATE=1e1,
    CROSS_SECTION_WAVELENGTH=640,
    DSTORM_PET_T_RATE_MOL=1e8,
    DSTORM_PET_S_RATE_MOL=1e9,
    DSTORM_PET_SUCCESS_RATE=1e-3,
    DSTORM_TH_EL_RATE_1=1e-2,
    DSTORM_P_EL_CROSS_SECTION=6e-24,
    RAD_ESCAPE_EFFICIENCY=0.01,
    RAD_RELAX_RATE=1.3e3,
    OFRET_EFFICIENCY=0.001,
    ISO_RATE=4e6,
    BISO_CROSS_SECTION=0.6e-17,
    BISO_THERMAL_RATE=5e3,
    BISO_EFFICIENCY=0.04,
)
cy5_dna.__doc__ += (
    "\nConstant photophysical attributes of Cy5 on DNA. "
    "\nAssumes that the buffer is oxygen-depleted."
)


atto643 = FluorophoreData(
    data_files="atto643_data",
    QUANTUM_YIELD=0.6,
    FLUORESCENCE_LIFETIME=3e-9,
    S1_QUENCH_RATE=0,  # to be updated
    ISC_ST_RATE=1e6,  # to be updated
    ISC_TS_RATE=1e5,  # to be updated
    RISC_RATE=0,  # to be updated
    PHOTOBLEACH_T1_RATE=1,  # to be updated
    H2O_ATTACK_S=3e4,  # to be updated
    H2O_ATTACK_T=0,  # to be updated
    BACK_REACTION=1e-1,  # to be updated
)
atto643.__doc__ += "\nConstant photophysical attributes of Atto643."


testfluo_1 = FluorophoreData(
    data_files="testing_data_1",
    QUANTUM_YIELD=0.27,
    FLUORESCENCE_LIFETIME=1e-9,
    ISC_ST_RATE=8.3e5,
    ISC_TS_RATE=5e3,
    RISC_RATE=0,
    PHOTOBLEACH_T1_RATE=1,
    DSTORM_PET_T_RATE_MOL=1e8,
    DSTORM_PET_S_RATE_MOL=1e9,
    DSTORM_PET_SUCCESS_RATE=1e-3,
    DSTORM_TH_EL_RATE_1=2e-2,
    ISO_RATE=2e7,
    BISO_CROSS_SECTION=1.7e-17,
)
testfluo_1.__test__ = False
testfluo_1.__doc__ += "\nConstant photophysical attributes of testing fluorophore 1."


testfluo_2 = FluorophoreData(
    data_files="testing_data_2",
    QUANTUM_YIELD=0.6,
    FLUORESCENCE_LIFETIME=3e-9,
    S1_QUENCH_RATE=0,
    ISC_ST_RATE=1e6,
    ISC_TS_RATE=1e5,
    RISC_RATE=0,
    PHOTOBLEACH_T1_RATE=1,
    H2O_ATTACK_S=3e4,
    H2O_ATTACK_T=0,
    BACK_REACTION=1e-1,
)
testfluo_2.__test__ = False
testfluo_2.__doc__ += "\nConstant photophysical attributes of testing fluorophore 2."
