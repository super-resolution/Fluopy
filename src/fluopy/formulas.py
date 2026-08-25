"""
Mathematical formulas for optical and spectral properties.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import constants

__all__: list[str] = [
    "convert_wavenumber_wavelength_frequency",
    "calculate_photon_flux",
    "calculate_excitation_rate",
    "calculate_emission_rate",
    "calculate_internal_conversion_rate",
    "henderson_hasselbalch_equation",
    "calculate_pet_rate",
    "calculate_spectral_overlap_integral",
    "calculate_fret_rate",
    "calculate_fret_efficiency",
    "calculate_photon_collection_rate",
]


_REDUCING_AGENT_PKA = {
    "mea": 9.0,
    "betaME": 9.6,
}


def convert_wavenumber_wavelength_frequency(
    wavenumber: float | npt.ArrayLike | None = None,
    wavelength: float | npt.ArrayLike | None = None,
    frequency: float | npt.ArrayLike | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Convert either wavenumber, wavelength or frequency into the other two.

    Parameters
    ----------
    wavenumber
        In 1/cm.
    wavelength
        In nm.
    frequency
        In Hz.

    Returns
    -------
    tuple[npt.NDArray[np.float64]]
        (wavenumber in 1/cm, wavelength in nm, frequency in Hz)
    """
    if sum(x is not None for x in [wavelength, wavenumber, frequency]) != 1:
        raise ValueError(
            "One and only one of wavenumber, wavelength and frequency must not be None."
        )
    if wavenumber is not None:
        wavenumber = np.asarray(wavenumber, dtype=np.float64)
        if np.any(~np.isfinite(wavenumber)) or np.any(wavenumber <= 0):
            raise ValueError("wavenumber must contain positive finite values.")
        wavelength = np.asarray(1 / (wavenumber * 1e2) * 1e9, dtype=np.float64)
        frequency = np.asarray(wavenumber * 1e2 * constants.c, dtype=np.float64)

    elif wavelength is not None:
        wavelength = np.asarray(wavelength, dtype=np.float64)
        if np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0):
            raise ValueError("wavelength must contain positive finite values.")
        wavenumber = np.asarray(1 / (wavelength * 1e-9) * 1e-2, dtype=np.float64)
        frequency = np.asarray(constants.c / (wavelength * 1e-9), dtype=np.float64)

    else:  # frequency is not None:
        frequency = np.asarray(frequency, dtype=np.float64)
        if np.any(~np.isfinite(frequency)) or np.any(frequency <= 0):
            raise ValueError("frequency must contain positive finite values.")
        wavenumber = np.asarray(frequency / constants.c * 1e-2, dtype=np.float64)
        wavelength = np.asarray(constants.c / frequency * 1e9, dtype=np.float64)

    return wavenumber, wavelength, frequency


def calculate_photon_flux(
    irradiance: float | npt.ArrayLike = 2, frequency: float | npt.ArrayLike = 4.5e14
) -> npt.NDArray[np.float64]:
    """
    Calculates the photon flux based on the irradiance and the frequency of the light.

    Parameters
    ----------
    irradiance
        The irradiance in kW/cm².
    frequency
        The frequency in Hz.

    Returns
    -------
    npt.NDArray[np.float64]
        The photon flux in 1/(m² s).
    """
    irradiance = np.asarray(irradiance, dtype=np.float64)
    frequency = np.asarray(frequency, dtype=np.float64)
    if np.any(~np.isfinite(irradiance)) or np.any(irradiance < 0):
        raise ValueError("irradiance must contain finite, non-negative values.")
    if np.any(~np.isfinite(frequency)) or np.any(frequency <= 0):
        raise ValueError("frequency must contain positive finite values.")
    irradiance = irradiance * 1e3 * 1e4
    photon_flux = np.asarray(irradiance / (constants.h * frequency))

    return photon_flux


def calculate_excitation_rate(
    photon_flux: float | npt.ArrayLike = 8e25,
    extinction_coefficient: float | npt.ArrayLike | None = None,
    absorption_cross_section: float | npt.ArrayLike | None = None,
) -> float | npt.NDArray[np.float64]:
    """
    Returns the excitation rate for a given irradiance and an extinction coefficient or
    an absorption cross section.

    Parameters
    ----------
    photon_flux
        The photon flux in 1/(m² s).
    extinction_coefficient
        Extinction coefficient of fluorophore at wavelength in 1/(cm M).
    absorption_cross_section
        Absorption cross section of fluorophore at wavelength in cm².
        The scattering cross section is assumed to be negligible, hence the absorption
        cross section equals the excitation cross section.

    Returns
    -------
    float | npt.NDArray[np.float64]
        The excitation rate in 1/s.
    """
    if (
        sum(x is not None for x in [extinction_coefficient, absorption_cross_section])
        != 1
    ):
        raise ValueError(
            "One and only one of extinction_coefficient and absorption_cross_section "
            "must not be None."
        )
    if extinction_coefficient is not None:
        extinction_coefficient = np.asarray(
            extinction_coefficient,
            dtype=np.float64,
        )
        if np.any(~np.isfinite(extinction_coefficient)) or np.any(
            extinction_coefficient < 0
        ):
            raise ValueError(
                "extinction_coefficient must contain finite, non-negative values."
            )
        cross_section = extinction_coefficient * 1e3 * np.log(10) / constants.Avogadro
    else:
        if absorption_cross_section is None:
            raise RuntimeError("absorption_cross_section unexpectedly missing")
        cross_section = np.asarray(
            absorption_cross_section,
            dtype=np.float64,
        )
        if np.any(~np.isfinite(cross_section)) or np.any(cross_section < 0):
            raise ValueError(
                "absorption_cross_section must contain finite, non-negative values."
            )
    cross_section = cross_section * 1e-4
    photon_flux_array = np.asarray(photon_flux, dtype=np.float64)
    excitation_rate = np.asarray(
        photon_flux_array * cross_section,
        dtype=np.float64,
    )

    return excitation_rate


def calculate_emission_rate(
    quantum_yield: float | npt.ArrayLike = 0.5,
    fluorescence_lifetime: float | npt.ArrayLike = 1e-9,
) -> float | npt.NDArray[np.float64]:
    """
    Returns the rate of fluorescent emission based on the quantum yield and the
    fluorescence lifetime.

    Parameters
    ----------
    quantum_yield
        Number between 0 and 1.
    fluorescence_lifetime
        The fluorescence lifetime in s.

    Returns
    -------
    float | npt.NDArray[np.float64]
        The rate of emission in 1/s.
    """
    quantum_yield = np.asarray(quantum_yield, dtype=np.float64)
    fluorescence_lifetime = np.asarray(
        fluorescence_lifetime,
        dtype=np.float64,
    )

    if (
        np.any(~np.isfinite(quantum_yield))
        or np.any(quantum_yield < 0)
        or np.any(quantum_yield > 1)
    ):
        raise ValueError("quantum_yield must contain finite values between 0 and 1.")

    if np.any(~np.isfinite(fluorescence_lifetime)) or np.any(
        fluorescence_lifetime <= 0
    ):
        raise ValueError("fluorescence_lifetime must contain positive finite values.")
    emission_rate = np.asarray(quantum_yield / fluorescence_lifetime, dtype=np.float64)

    return emission_rate


def calculate_internal_conversion_rate(
    quantum_yield: float | npt.ArrayLike = 0.5,
    emission_rate: float | npt.ArrayLike = 5e8,
    *other_outgoing_rates_args: float,
    **other_outgoing_rates_kwargs: float,
) -> float | npt.NDArray[np.float64]:
    """
    Calculates the rate of internal conversion from the first excited state to the
    vibrationally excited but electronic ground state.

    Parameters
    ----------
    quantum_yield
        Number between 0 and 1.
    emission_rate
        The rate of emission in 1/s.
    other_outgoing_rates_args
        Rates of all other transitions (except fluorescence emission) that leave the
        first excited state in 1/s.
    other_outgoing_rates_kwargs
        Rates of all other transitions (except fluorescence emission) that leave the
        first excited state in 1/s.

    Returns
    -------
    float | npt.NDArray[np.float64]
        The rate of internal conversion in 1/s.
    """
    quantum_yield = np.asarray(quantum_yield, dtype=np.float64)
    if (
        np.any(~np.isfinite(quantum_yield))
        or np.any(quantum_yield <= 0)
        or np.any(quantum_yield > 1)
    ):
        raise ValueError(
            "quantum_yield must contain finite values greater than 0 and at most 1."
        )
    emission_rate = np.asarray(emission_rate, dtype=np.float64)
    if np.any(~np.isfinite(emission_rate)) or np.any(emission_rate < 0):
        raise ValueError("emission_rate must contain finite, non-negative values.")
    internal_conversion_rate = np.asarray(
        emission_rate / quantum_yield - emission_rate, dtype=np.float64
    )

    for outgoing_rate in other_outgoing_rates_args:
        internal_conversion_rate -= outgoing_rate
    for _, outgoing_rate in other_outgoing_rates_kwargs.items():
        internal_conversion_rate -= outgoing_rate
    if np.any(internal_conversion_rate < 0):
        raise ValueError(
            "emission rate is too low to produce the given quantum yield while "
            "competing with other given transitions."
        )
    return internal_conversion_rate


def henderson_hasselbalch_equation(
    ph: float, pka: float, concentration: float
) -> float:
    """
    Returns the estimated concentration of the base given the total concentration.

    Parameters
    ----------
    ph
        The pH as indicator of acidity or basicity.
    pka
        Acid dissociation constant.
    concentration
        Total concentration of the agent in mM.

    Returns
    -------
    float
        Concentration of the base in mM.
    """
    base_to_acid = 10 ** (ph - pka)
    base_concentration = base_to_acid * concentration / (base_to_acid + 1)

    return base_concentration


def calculate_pet_rate(
    reducing_agent: Literal["mea", "betaME"] = "mea",
    concentration: float = 143,
    k_pet: float = 1,
    ph: float = 8.0,
) -> float:
    """
    Returns the dSTORM reduction rate for a given reducing agent and its concentration.

    Parameters
    ----------
    reducing_agent
        One of 'mea' (mercaptoethylamine), 'betaME' (mercaptoethanol).
    concentration
        Concentration of the reducing agent in mM.
    k_pet
        The rate of photoinduced electron transfer in 1/(s M).
    ph
        The pH as indicator of acidity or basicity.

    Returns
    -------
    float
        The PeT rate in 1/s.
    """
    try:
        pka = _REDUCING_AGENT_PKA[reducing_agent]
    except KeyError:
        raise ValueError('reducing_agent has to be one of "betaME", "mea".') from None

    concentration = (
        henderson_hasselbalch_equation(ph=ph, pka=pka, concentration=concentration)
        * 1e-3
    )
    pet_rate = k_pet * concentration

    return pet_rate


def calculate_spectral_overlap_integral(
    donor: npt.ArrayLike,
    acceptor: npt.ArrayLike,
    wavelengths: npt.ArrayLike,
    donor_area: float | None = None,
) -> float:
    """
    Calculates the spectral overlap integral defined as the integral of the
    multiplication of the donor emission spectrum normalized to an area of 1, the
    acceptor molar extinction coefficient as a function of wavelength and the
    wavelength to the power of 4.

    Parameters
    ----------
    donor : 1-D array_like
        Contains emission values of the donor - they don't have to be normalized yet.
    acceptor : 1-D array_like
        Contains the acceptors molar extinction coefficients in 1/(M cm).
    wavelengths : 1-D array_like
        The wavelength values in nm, that correspond to the respective donor and
        acceptor values.
    donor_area
        Area of the complete donor emission spectrum. If None, calculate the are from
        donor and wavelengths.

    Returns
    -------
    float
        The value of the spectral overlap integral in (nm**4)/(M cm).
    """
    donor = np.asarray(donor, dtype=np.float64)
    acceptor = np.asarray(acceptor, dtype=np.float64)
    wavelengths = np.asarray(wavelengths, dtype=np.float64)

    if donor.ndim != 1 or acceptor.ndim != 1 or wavelengths.ndim != 1:
        raise ValueError("donor, acceptor and wavelengths must be one-dimensional.")

    if donor.shape != acceptor.shape or donor.shape != wavelengths.shape:
        raise ValueError("donor, acceptor and wavelengths must have matching shapes.")

    if donor.size < 2:
        raise ValueError(
            "donor, acceptor and wavelengths must contain at least two values."
        )

    if (
        np.any(~np.isfinite(donor))
        or np.any(~np.isfinite(acceptor))
        or np.any(~np.isfinite(wavelengths))
    ):
        raise ValueError("donor, acceptor and wavelengths must contain finite values.")

    if np.any(np.diff(wavelengths) <= 0):
        raise ValueError("wavelengths must be strictly increasing.")
    if np.any(donor < 0) or np.any(acceptor < 0):
        raise ValueError("donor and acceptor values must be non-negative.")

    if donor_area is None:
        donor_area = float(np.trapezoid(donor, x=wavelengths))

    if not np.isfinite(donor_area) or donor_area <= 0:
        raise ValueError("donor emission spectrum must have positive finite area.")

    normalized_donor = donor / donor_area
    integrand = normalized_donor * acceptor * wavelengths**4
    spectral_overlap_integral = float(np.trapezoid(integrand, x=wavelengths))

    return spectral_overlap_integral


def calculate_fret_rate(
    distance: float = 10,
    emission_rate: float = 5e8,
    spectral_overlap_integral: float = 1e16,
    dipole_orientation_factor: float = 2 / 3,
    refractive_index: float = 1.33,
) -> float:
    """
    Calculates the Förster resonance energy transfer rate.

    Parameters
    ----------
    distance
        In nm.
    emission_rate
        In 1/s.
    spectral_overlap_integral
        In (nm**4)/(M cm).
    dipole_orientation_factor
        The dipole orientation factor κ².
    refractive_index
        The refractive index of the medium.

    Returns
    -------
    float
        fret rate in 1/s.
    """
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("distance must be positive and finite.")
    if not np.isfinite(emission_rate) or emission_rate < 0:
        raise ValueError("emission_rate must be finite and non-negative.")
    if not np.isfinite(spectral_overlap_integral) or spectral_overlap_integral < 0:
        raise ValueError("spectral_overlap_integral must be finite and non-negative.")
    if (
        not np.isfinite(dipole_orientation_factor)
        or not 0 <= dipole_orientation_factor <= 4
    ):
        raise ValueError(
            "dipole_orientation_factor must be finite and between 0 and 4."
        )
    if not np.isfinite(refractive_index) or refractive_index <= 0:
        raise ValueError("refractive_index must be positive and finite.")
    fret_rate = (
        8.785
        * 1e-11
        * (
            (dipole_orientation_factor * emission_rate)
            / (refractive_index**4 * distance**6)
        )
        * spectral_overlap_integral
    )

    return fret_rate


def calculate_fret_efficiency(
    fret_rate: float = 1e8, fluorescence_lifetime: float = 1e-9
) -> float:
    """
    Calculates the FRET efficiency.

    Parameters
    ----------
    fret_rate
        In 1/s.
    fluorescence_lifetime
        The fluorescence lifetime of the donor in absence of the acceptor in s.

    Returns
    -------
    float
        The FRET efficiency (dimensionless). Between 0 and 1.
    """
    if not np.isfinite(fret_rate) or fret_rate < 0:
        raise ValueError("fret_rate must be finite and non-negative.")
    if not np.isfinite(fluorescence_lifetime) or fluorescence_lifetime <= 0:
        raise ValueError("fluorescence_lifetime must be positive and finite.")
    tau_1 = fluorescence_lifetime
    tau_2 = 1 / (1 / fluorescence_lifetime + fret_rate)
    efficiency = 1 - tau_2 / tau_1

    return efficiency


def calculate_photon_collection_rate(NA: float = 1.45, n1: float = 1.51) -> float:
    """
    Calculates the photon collection rate based on the numerical aperture of the
    objective.

    Parameters
    ----------
    NA
        Numerical aperture of the objective.
    n1
        Refractive index of the medium.

    Returns
    -------
    float
        The photon collection rate.
    """
    if not np.isfinite(n1) or n1 <= 0:
        raise ValueError("n1 must be positive and finite.")
    if not np.isfinite(NA) or not 0 <= NA <= n1:
        raise ValueError("NA must be finite and between 0 and n1.")
    half_angle = np.arcsin(NA / n1)
    cone = 2 * np.pi * (1 - np.cos(half_angle))
    photon_collection_rate = float(cone / (4 * np.pi))

    return photon_collection_rate
