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
    RAD_ESCAPE_RATE: float = 0
    RAD_RELAX_RATE: float = 0

    # cis trans isomerization
    ISO_RATE: float = 0
    BISO_CROSS_SECTION: float = 0
    BISO_THERMAL_RATE: float = 0
    BISO_EFFICIENCY: float = 0

    # rhodamines
    H2O_ATTACK_S: float = 0
    H2O_ATTACK_T: float = 0
    BACK_REACTION: float = 0


@dataclass
class Cy5_DNA(FluorophoreData):
    """
    The buffer is oxygen-depleted.
    """
    data_files: str = "cy5_data"

    QUANTUM_YIELD: float = 0.27
    FLUORESCENCE_LIFETIME: float = 1e-9
    ISC_ST_RATE: float = 8.3e5
    ISC_TS_RATE: float = 5e3
    RISC_RATE: float = 0
    PHOTOBLEACH_T1_RATE: float = 1e1  # estimation
    PHOTOBLEACH_T2_RATE: float = 0

    DSTORM_PET_T_RATE_MOL: float = 1e8
    DSTORM_PET_S_RATE_MOL: float = 1e9
    DSTORM_PET_SUCCESS_RATE: float = 1e-3
    DSTORM_TH_EL_RATE_1: float = 2e-2
    DSTORM_TH_EL_RATE_2: float = 0
    RAD_ESCAPE_RATE: float = 8e5
    RAD_RELAX_RATE: float = 8e2

    ISO_RATE: float = 4e6
    BISO_CROSS_SECTION: float = 0.6e-17
    BISO_THERMAL_RATE: float = 5e3
    BISO_EFFICIENCY: float = 0.03


@dataclass
class Cy5_Widengren(FluorophoreData):
    """
    Contains constant attributes of the fluorophore Cy5 as defined in Characterization 
    of photoinduced isomerization and back-isomerization of the cyanine dye Cy5 by FCS 
    (Widengren, Schwille).
    The buffer is air-saturated.
    """

    data_files: str = "cy5_data"

    QUANTUM_YIELD: float = 0.27
    FLUORESCENCE_LIFETIME: float = 1e-9
    ISC_ST_RATE: float = 8.3e5
    ISC_TS_RATE: float = 5e5
    RISC_RATE: float = 0
    PHOTOBLEACH_T1_RATE: float = 2.5e3
    PHOTOBLEACH_T2_RATE: float = 0

    DSTORM_PET_T_RATE_MOL: float = 1e8
    DSTORM_PET_S_RATE_MOL: float = 1e9
    DSTORM_PET_SUCCESS_RATE: float = 1e-3
    DSTORM_TH_EL_RATE_1: float = 2e-2
    DSTORM_TH_EL_RATE_2: float = 0

    ISO_RATE: float = 2e7
    BISO_CROSS_SECTION: float = 1.7e-17
    BISO_THERMAL_RATE: float = 5e3
    BISO_EFFICIENCY: float = 0.03


@dataclass
class Cy5_Widengren_DNA(FluorophoreData):
    """
    Contains constant attributes of the fluorophore Cy5 as defined in Characterization
    of photoinduced isomerization and back-isomerization of the cyanine dye Cy5 by FCS
    (Widengren, Schwille).
    The buffer is air-saturated.
    Here, Cy5 is bound to DNA. 
    The ratios of 632 nm (DNA to noDNA) are applied to 647 nm.
    """

    data_files: str = "cy5_data"

    QUANTUM_YIELD: float = 0.27
    FLUORESCENCE_LIFETIME: float = 1e-9
    ISC_ST_RATE: float = 4e5
    ISC_TS_RATE: float = 1e5
    RISC_RATE: float = 0
    PHOTOBLEACH_T1_RATE: float = 5e2  # since ISC_TS_RATE is 1/5 of the original
    PHOTOBLEACH_T2_RATE: float = 0

    DSTORM_PET_T_RATE_MOL: float = 1e8
    DSTORM_PET_S_RATE_MOL: float = 1e9
    DSTORM_PET_SUCCESS_RATE: float = 1e-3
    DSTORM_TH_EL_RATE_1: float = 2e-2
    DSTORM_TH_EL_RATE_2: float = 0

    ISO_RATE: float = 4e6
    BISO_CROSS_SECTION: float = 0.6e-17
    BISO_THERMAL_RATE: float = 5e3
    BISO_EFFICIENCY: float = 0.2


@dataclass
class Cy5_Gidi_DNA(FluorophoreData):
    """
    Contains constant attributes of the fluorophore Cy5 as defined in Unifying 
    Mechanism for Thiol-Induced Photoswitching and Photostability of Cyanine Dyes (Gidi 
    et al.).
    The buffer is oxygen-depleted.
    """

    data_files: str = "cy5_data"

    QUANTUM_YIELD: float = 0.27
    FLUORESCENCE_LIFETIME: float = 1e-9
    ISC_ST_RATE: float = 8.3e5
    ISC_TS_RATE: float = 5e3
    RISC_RATE: float = 0
    PHOTOBLEACH_T1_RATE: float = 1e1  # estimation
    PHOTOBLEACH_T2_RATE: float = 0

    DSTORM_PET_T_RATE_MOL: float = 1e8
    DSTORM_PET_S_RATE_MOL: float = 1e9
    DSTORM_PET_SUCCESS_RATE: float = 1e-3
    DSTORM_TH_EL_RATE_1: float = 2e-2
    DSTORM_TH_EL_RATE_2: float = 0

    ISO_RATE: float = 4e6
    BISO_CROSS_SECTION: float = 0.6e-17
    BISO_THERMAL_RATE: float = 5e3
    BISO_EFFICIENCY: float = 0.2


@dataclass
class Atto643(FluorophoreData):
    """
    Contains constant attributes of the fluorophore Atto643.
    """

    data_files: str = "atto643_data"

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

    data_files: str = "testing_data_1"

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

    data_files: str = "testing_data_2"

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
