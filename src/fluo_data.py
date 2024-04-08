"""
Module fluo_data
"""

from dataclasses import dataclass


@dataclass
class FluorophoreData:
    """
    Contains all constant photophysical attributes of fluorophores.
    Closely related to TransitionType.
    """

    # spectra
    data_files: str = None

    # general
    MAXIMUM_EXTINCTION_COEFFICIENT: float = 0
    QUANTUM_YIELD: float = 0
    FLUORESCENCE_LIFETIME: float = 0
    S1_QUENCH_RATE: float = 0
    ISC_ST_RATE: float = 0
    ISC_TS_RATE: float = 0
    RISC_RATE: float = 0
    PHOTOBLEACH_T1_RATE: float = 0
    PHOTOBLEACH_T2_RATE: float = 0

    # dstorm
    DSTORM_PET_T_RATE_MOL: float = 0
    DSTORM_PET_S_RATE_MOL: float = 0
    DSTORM_PET_SUCCESS_RATE: float = 0
    DSTORM_TH_EL_RATE_1: float = 0
    DSTORM_TH_EL_RATE_2: float = 0

    # cis trans isomerization
    ISO_RATE: float = 0
    BISO_CROSS_SECTION: float = 0

    # rhodamines
    H2O_ATTACK_S: float = 0
    H2O_ATTACK_T: float = 0
    BACK_REACTION: float = 0


@dataclass
class Cy5(FluorophoreData):
    """
    Contains constant attributes of the fluorophore Cy5.
    """

    data_files: str = "cy5_data"

    MAXIMUM_EXTINCTION_COEFFICIENT: float = 2.5e5
    QUANTUM_YIELD: float = 0.27
    FLUORESCENCE_LIFETIME: float = 1e-9
    ISC_ST_RATE: float = 8.3e5
    ISC_TS_RATE: float = 5e3
    RISC_RATE: float = 0
    PHOTOBLEACH_T1_RATE: float = 1
    PHOTOBLEACH_T2_RATE: float = 0

    DSTORM_PET_T_RATE_MOL: float = 1e8
    DSTORM_PET_S_RATE_MOL: float = 1e9
    DSTORM_PET_SUCCESS_RATE: float = 1e-3
    DSTORM_TH_EL_RATE_1: float = 2e-2
    DSTORM_TH_EL_RATE_2: float = 0

    ISO_RATE: float = 2e7
    BISO_CROSS_SECTION: float = 1.7e-17


@dataclass
class Atto643(FluorophoreData):
    """
    Contains constant attributes of the fluorophore Atto643.
    """

    data_files: str = "atto643_data"

    MAXIMUM_EXTINCTION_COEFFICIENT: float = 1.5e5
    QUANTUM_YIELD: float = 0.6
    FLUORESCENCE_LIFETIME: float = 3e-9
    S1_QUENCH_RATE: float = 0  # TBU
    ISC_ST_RATE: float = 1e6  # TBU
    ISC_TS_RATE: float = 1e5  # TBU
    RISC_RATE: float = 0  # TBU
    PHOTOBLEACH_T1_RATE: float = 1  # TBU
    PHOTOBLEACH_T2_RATE: float = 0

    H2O_ATTACK_S: float = 3e4  # TBU
    H2O_ATTACK_T: float = 0  # TBU
    BACK_REACTION: float = 1e-1  # TBU


@dataclass
class TestFluo_1(FluorophoreData):
    """
    Contains constant attributes of the testing fluorophore 1.
    """

    __test__ = False

    data_files: str = "cy5_data"

    MAXIMUM_EXTINCTION_COEFFICIENT: float = 2.5e5
    QUANTUM_YIELD: float = 0.27
    FLUORESCENCE_LIFETIME: float = 1e-9
    ISC_ST_RATE: float = 8.3e5
    ISC_TS_RATE: float = 5e3
    RISC_RATE: float = 0
    PHOTOBLEACH_T1_RATE: float = 1
    PHOTOBLEACH_T2_RATE: float = 0

    DSTORM_PET_T_RATE_MOL: float = 1e8
    DSTORM_PET_S_RATE_MOL: float = 1e9
    DSTORM_PET_SUCCESS_RATE: float = 1e-3
    DSTORM_TH_EL_RATE_1: float = 2e-2
    DSTORM_TH_EL_RATE_2: float = 0

    ISO_RATE: float = 2e7
    BISO_CROSS_SECTION: float = 1.7e-17


@dataclass
class TestFluo_2(FluorophoreData):
    """
    Contains constant attributes of the testing fluorophore 2.
    """

    __test__ = False

    data_files: str = "atto643_data"

    MAXIMUM_EXTINCTION_COEFFICIENT: float = 1.5e5
    QUANTUM_YIELD: float = 0.6
    FLUORESCENCE_LIFETIME: float = 3e-9
    S1_QUENCH_RATE: float = 0  # TBU
    ISC_ST_RATE: float = 1e6  # TBU
    ISC_TS_RATE: float = 1e5  # TBU
    RISC_RATE: float = 0  # TBU
    PHOTOBLEACH_T1_RATE: float = 1  # TBU
    PHOTOBLEACH_T2_RATE: float = 0

    H2O_ATTACK_S: float = 3e4  # TBU
    H2O_ATTACK_T: float = 0  # TBU
    BACK_REACTION: float = 1e-1  # TBU
