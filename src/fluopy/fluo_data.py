"""
Photophysical constants for specific fluorophores.

This module provides a dataclass container to hold photophysical constants.
"""

from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__: list[str] = ["Spectrum", "FluorophoreData", "cy5_dna", "atto643"]


@dataclass
class Spectrum:
    """
    Contains wavelength-dependent spectral data.

    Spectrum values can be provided directly as array-like objects or loaded from a csv
    file. Wavelengths are given in nm and must be strictly increasing.

    Attributes
    ----------
    wavelengths : 1-D array_like
        The wavelength values in nm.
    values : 1-D array_like
        Spectrum values corresponding to wavelengths.
    """

    wavelengths: npt.ArrayLike
    values: npt.ArrayLike

    @classmethod
    def from_arrays(
        cls,
        wavelengths: npt.ArrayLike,
        values: npt.ArrayLike,
    ) -> Self:
        """
        Create a spectrum from wavelength and value arrays.

        Parameters
        ----------
        wavlengths
            Wavelengths in nm.
        values
            Spectrum values corresponding to wavelenghts.

        Returns
        -------
        Spectrum
            Spectrum object containing copies of the input arrays.
        """
        return cls(wavelengths=wavelengths, values=values)

    @classmethod
    def from_csv(
        cls,
        path: str | PathLike[str],
        wavelength_column: str = "Wavelengths",
        value_column: str = "y",
    ) -> Self:
        """
        Create a spectrum from a CSV file.

        Parameters
        ----------
        path
            Path to the CSV file.
        wavelength_column
            Name of the column containing wavelengths in nm.
        value_column
            Name of the column containing spetrum values.

        Returns
        -------
        Spectrum
            Spectrum object loaded from the CSV file.
        """
        data = pd.read_csv(path)

        missing_columns = {
            wavelength_column,
            value_column,
        }.difference(data.columns)

        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"spectrum CSV is missing columns: {missing}.")

        return cls(
            wavelengths=data[wavelength_column].to_numpy(),
            values=data[value_column].to_numpy(),
        )

    def at(self, wavelength: float) -> float:
        """
        Return the spectrum value at a wavelength.

        Values between given wavelengths are linearily interpolated. Extrapolation
        outside the spectrum range is not supported.

        Parameters
        ----------
        wavelength
            Wavelength in nm.

        Returns
        -------
        value
            Spectrum value at the specified wavelength.
        """
        if not np.isfinite(wavelength):
            raise ValueError("wavelength must be finite.")

        minimum = self.wavelengths[0]
        maximum = self.wavelengths[-1]

        if wavelength < minimum or wavelength > maximum:
            raise ValueError(
                f"wavelength {wavelength} nm is outside the spectrum range "
                f"{minimum}-{maximum} nm."
            )

        value = float(np.interp(wavelength, self.wavelengths, self.values))

        return value

    def integral(
        self,
        lower: float | None = None,
        upper: float | None = None,
    ) -> float:
        """
        Integrate the spectrum over a wavelength interval.

        Integration limits outside the available spectrum are clipped to the spectrum
        range. An interval without overlap has an integral of zero.

        Parameters
        ----------
        lower
            Lower integration limit in nm. If None, use the lowest available wavelength.
        upper
            Upper integration limit in nm. If None, use the highest available
            wavelength.

        Returns
        -------
        integral
            Trapezoidal integral of the spectrum.
        """
        if lower is None:
            lower = float(self.wavelengths[0])
        if upper is None:
            upper = float(self.wavelengths[-1])

        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("integration limits must be finite.")
        if lower >= upper:
            raise ValueError(
                "the lower integration limit must be smaller than the upper limit."
            )

        lower = max(lower, float(self.wavelengths[0]))
        upper = min(upper, float(self.wavelengths[-1]))

        if lower >= upper:
            return 0.0

        inside = (self.wavelengths > lower) & (self.wavelengths < upper)
        wavelengths = np.concatenate(([lower], self.wavelengths[inside], [upper]))
        values = np.interp(
            wavelengths,
            self.wavelengths,
            self.values,
        )
        integral = float(np.trapezoid(values, wavelengths))

        return integral

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


def _load_bundled_spectra(
    directory_name: str,
) -> tuple[Spectrum, dict[str, Spectrum]]:
    directory = Path(__file__).parent / "fluorophore_spectra" / directory_name

    emission_spectrum = Spectrum.from_csv(directory / "emission.csv")
    absorption_spectra = {
        path.stem.removeprefix("absorption_"): Spectrum.from_csv(path)
        for path in sorted(directory.glob("absorption_*.csv"))
    }

    return emission_spectrum, absorption_spectra


@dataclass
class FluorophoreData:
    """
    Container for all constant photophysical attributes of a fluorophore.
    The naming of constants is closely related to TransitionType.

    Attributes
    ----------
    emission_spectrum : Spectrum | None
        Emission spectrum used for bandpass filtering and as the donor spectrum in
        energy-transfer calculations.
    absorption_spectra : dict[str, Spectrum]
        Absorption spectra indexed by lowercase acceptor-state names, for example
        's0', 't1', 'cis' or 'off'. The S0 spectrum is also used to infer the excitation
        rate.
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
        The wavelength in nm at which individual absorption cross sections are defined.
        Standard excitation from S0 is calculated using the S0 absorption spectrum in
        absorption_spectra. For other transitions, such as photoinduced
        back-isomerization from cis, an individual cross section can be provided.
        CROSS_SECTION_WAVELENGTH is used to check whether these cross sections
        correspond to the specified excitation wavelength.
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
    emission_spectrum: Spectrum | None = None
    absorption_spectra: dict[str, Spectrum] = field(default_factory=dict)

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

    def __post_init__(self) -> None:
        if self.emission_spectrum is not None and not isinstance(
            self.emission_spectrum, Spectrum
        ):
            raise TypeError("emission_spectrum must be a Spectrum or None.")

        for state, spectrum in self.absorption_spectra.items():
            if not isinstance(state, str) or not state:
                raise TypeError("absorption_spectra keys must be non-empty strings.")
            if not isinstance(spectrum, Spectrum):
                raise TypeError(
                    f"absorption spectrum for state {state!r} must be a Spectrum."
                )


_cy5_emission, _cy5_absorption = _load_bundled_spectra("cy5_data")
_atto643_emission, _atto643_absorption = _load_bundled_spectra("atto643_data")
_testfluo_1_emission, _testfluo_1_absorption = _load_bundled_spectra("testing_data_1")
_testfluo_2_emission, _testfluo_2_absorption = _load_bundled_spectra("testing_data_2")


cy5_dna = FluorophoreData(
    emission_spectrum=_cy5_emission,
    absorption_spectra=_cy5_absorption,
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
    emission_spectrum=_atto643_emission,
    absorption_spectra=_atto643_absorption,
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
    emission_spectrum=_testfluo_1_emission,
    absorption_spectra=_testfluo_1_absorption,
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
    emission_spectrum=_testfluo_2_emission,
    absorption_spectra=_testfluo_2_absorption,
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
