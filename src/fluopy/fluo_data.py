"""
Photophysical constants for specific fluorophores.

This module provides a dataclass container to hold photophysical constants.
"""

from dataclasses import dataclass
from pathlib import Path

__all__: list[str] = ["FluorophoreData", "cy5_dna", "atto643"]


@dataclass
class FluorophoreData:
    """
    Container for all constant photophysical attributes of a fluorophore.
    The naming of constants is closely related to TransitionType.
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
    PHOTOBLEACH_T2_RATE: float = 0

    # dstorm
    DSTORM_PET_T_RATE_MOL: float = 0
    DSTORM_PET_S_RATE_MOL: float = 0
    DSTORM_PET_SUCCESS_RATE: float = 0
    DSTORM_TH_EL_RATE_1: float = 0
    DSTORM_TH_EL_RATE_2: float = 0
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
    PHOTOBLEACH_T2_RATE=0,
    DSTORM_PET_T_RATE_MOL=1e8,
    DSTORM_PET_S_RATE_MOL=1e9,
    DSTORM_PET_SUCCESS_RATE=1e-3,
    DSTORM_TH_EL_RATE_1=1e-2,
    DSTORM_TH_EL_RATE_2=0,
    DSTORM_P_EL_CROSS_SECTION=6e-24,  # 640 nm
    RAD_ESCAPE_EFFICIENCY=0.01,
    RAD_RELAX_RATE=1.3e3,
    OFRET_EFFICIENCY=0.001,
    ISO_RATE=4e6,
    BISO_CROSS_SECTION=0.6e-17,  # 640 nm
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
    PHOTOBLEACH_T2_RATE=0,
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
    PHOTOBLEACH_T2_RATE=0,
    DSTORM_PET_T_RATE_MOL=1e8,
    DSTORM_PET_S_RATE_MOL=1e9,
    DSTORM_PET_SUCCESS_RATE=1e-3,
    DSTORM_TH_EL_RATE_1=2e-2,
    DSTORM_TH_EL_RATE_2=0,
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
    PHOTOBLEACH_T2_RATE=0,
    H2O_ATTACK_S=3e4,
    H2O_ATTACK_T=0,
    BACK_REACTION=1e-1,
)
testfluo_2.__test__ = False
testfluo_2.__doc__ += "\nConstant photophysical attributes of testing fluorophore 2."
