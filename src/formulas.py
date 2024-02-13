"""
Module formulas
"""
import numpy as np
from scipy import constants


def convert_wavenumber_wavelength_frequency(wavenumber=None, wavelength=None, frequency=None):
    """
    Convert either wavenumber, wavelength or frequency into the other two.

    Parameters
    ----------
    wavenumber : float, 1-D array_like
        In 1/cm.
    wavelength : float, 1-D array_like
        In nm.
    frequency : float, 1-D array_like
        In Hz.

    Returns
    -------
    wavenumber : np.ndarray
        In 1/cm.
    wavelength : np.ndarray
        In nm.
    frequency : np.ndarray
        In Hz.
    """
    if sum(x is not None for x in [wavelength, wavenumber, frequency]) != 1:
        raise ValueError('One and only one of wavenumber, wavelength and frequency must not be None.')
    if wavenumber is not None:
        wavenumber = np.asarray(wavenumber)
        wavelength = np.asarray(1/(wavenumber * 1e2) * 1e9)
        frequency = np.asarray(wavenumber * 1e2 * constants.c)

    elif wavelength is not None:
        wavelength = np.asarray(wavelength)
        wavenumber = np.asarray(1/(wavelength * 1e-9) * 1e-2)
        frequency = np.asarray(constants.c / (wavelength * 1e-9))

    elif frequency is not None:
        frequency = np.asarray(frequency)
        wavenumber = np.asarray(frequency / constants.c * 1e-2)
        wavelength = np.asarray(constants.c / frequency * 1e9)

    else:
        pass

    return wavenumber, wavelength, frequency


def calculate_photon_flux(irradiance=2, frequency=4.5e14):
    """
    Calculates the photon flux based on the irradiance and the frequency of the light.

    Parameters
    ----------
    irradiance : float, 1-D array_like
        The irradiance in kW/cm².
    frequency : float, 1-D array_like
        The frequency in Hz.

    Returns
    -------
    photon_flux : np.ndarray
        The photon flux in 1/(m² s).
    """
    irradiance = np.asarray(irradiance)
    frequency = np.asarray(frequency)
    irradiance = irradiance * 1e3 * 1e4
    photon_flux = np.asarray(irradiance / (constants.h * frequency))

    return photon_flux


def calculate_excitation_rate(photon_flux=8e25, extinction_coefficient=None, absorption_cross_section=None):
    """
    Returns the excitation rate for a given irradiance and an extinction coefficient or an absorption cross section.

    Parameters
    ----------
    photon_flux : float, 1-D array_like
        The photon flux in 1/(m² s).
    extinction_coefficient : float, 1-D array_like
        Extinction coefficient of fluorophore at wavelength in 1/(cm M).
    absorption_cross_section : float, 1-D array_like
        Absorption cross section of fluorophore at wavelength in cm².
        The scattering cross section is assumed to be negligible, hence the absorption cross section equals the
        excitation cross section.

    Returns
    -------
    excitation_rate : float, np.ndarray
        The excitation rate in 1/s.
    """
    if sum(x is not None for x in [extinction_coefficient, absorption_cross_section]) != 1:
        raise ValueError('One and only one of extinction_coefficient and absorption_cross_section must not be None.')
    if extinction_coefficient is not None:
        absorption_cross_section = np.asarray(extinction_coefficient) * 1e3 * np.log(10) / constants.Avogadro

    absorption_cross_section = np.asarray(absorption_cross_section) * 1e-4
    excitation_rate = np.asarray(photon_flux) * np.asarray(absorption_cross_section)

    return excitation_rate


def calculate_emission_rate(quantum_yield=0.5, fluorescence_lifetime=1e-9):
    """
    Returns the rate of fluorescent emission based on the quantum yield and the fluorescence lifetime.

    Parameters
    ----------
    quantum_yield : float, 1-D array_like
        Number between 0 and 1.
    fluorescence_lifetime : float, 1-D array_like
        The fluorescence lifetime in s.

    Returns
    -------
    emis_rate : float, np.ndarray
        The rate of emission in 1/s.
    """
    emis_rate = np.asarray(quantum_yield) / np.asarray(fluorescence_lifetime)

    return emis_rate


def calculate_internal_conversion_rate(quantum_yield=0.5, emission_rate=5e8, *other_outgoing_rates_args,
                                       **other_outgoing_rates_kwargs):
    """
    Calculates the rate of internal conversion from the first excited state to the vibrationally excited but
    electronic ground state.

    Parameters
    ----------
    quantum_yield : float, 1-D array_like
        Number between 0 and 1.
    emission_rate : float, 1-D array_like
        The rate of emission in 1/s.
    other_outgoing_rates_args : floats
        Rates of all other transitions (except fluorescence emission) that leave the first excited state in 1/s.
    other_outgoing_rates_kwargs : floats
        Rates of all other transitions (except fluorescence emission) that leave the first excited state in 1/s.

    Returns
    -------
    internal_conversion_rate : float, np.ndarray
        The rate of internal conversion in 1/s.
    """
    quantum_yield = np.asarray(quantum_yield)
    if np.any(quantum_yield < 0) or np.any(quantum_yield > 1):
        raise ValueError('Quantum yield has to be between 0 and 1.')
    internal_conversion_rate = np.asarray(emission_rate)/quantum_yield - np.asarray(emission_rate)
    for outgoing_rate in other_outgoing_rates_args:
        internal_conversion_rate -= outgoing_rate
    for _, outgoing_rate in other_outgoing_rates_kwargs.items():
        internal_conversion_rate -= outgoing_rate
    if np.any(internal_conversion_rate < 0):
        raise ValueError('emission rate is too low to produce the given quantum yield while competing with other given '
                         'transitions.')
    return internal_conversion_rate


def calculate_back_isomerization_rate(photon_flux=8e25, absorption_cross_section=1e-17):
    """
    Returns the back-isomerization rate for a given irradiance and an absorption cross section of the molecule/isomer.

    Parameters
    ----------
    photon_flux : float, 1-D array_like
        The photon flux in 1/(m² s).
    absorption_cross_section : float, 1-D array_like
        Absorption cross section of fluorophore isomer at wavelength w in cm².

    Returns
    -------
    back_isomerization_rate : float, np.ndarray
        The back-isomerization rate in 1/s.
    """
    absorption_cross_section = np.asarray(absorption_cross_section) * 1e-4
    back_isomerization_rate = np.asarray(photon_flux) * absorption_cross_section

    return back_isomerization_rate


def henderson_hasselbalch_equation(ph, pka, concentration):
    """
    Returns the estimated concentration of the base given the total concentration.

    Parameters
    ----------
    ph : float
        The pH as indicator of acidity or basicity.
    pka : float
        Acid dissociation constant.
    concentration : float
        Total concentration of the agent in mM.

    Returns
    -------
    base_concentration : float
        Concentration of the base in mM.
    """
    base_to_acid = 10**(ph - pka)
    base_concentration = base_to_acid * concentration / (base_to_acid + 1)

    return base_concentration


def calculate_pet_rate(reducing_agent='mea', concentration=143, k_pet=1, ph=8):
    """
    Returns the dSTORM reduction rate for a given reducing agent and its concentration.

    Parameters
    ----------
    reducing_agent : str
        One of 'mea', 'betaME'.
    concentration : float
        Concentration of the reducing agent in mM.
    k_pet : float
        The rate of photoinduced electron transfer in 1/(s M).
    ph : float
        The pH as indicator of acidity or basicity.

    Returns
    -------
    reduction_rate : float
        The reduction rate in 1/s.
    """
    # the factor 1/7 (or 7) comes from protocols stating to either use 100 µl 100 mM MEA or 10 µl 143 mM beta-ME
    if reducing_agent == 'betaME':
        pka = 9.6
    elif reducing_agent == 'mea':
        pka = 9.5
    else:
        raise ValueError('reducing_agent has to be one of "betaME", "mea".')

    concentration = henderson_hasselbalch_equation(ph=ph, pka=pka, concentration=concentration) * 1e-3
    reduction_rate = k_pet * concentration

    return reduction_rate


def calculate_spectral_overlap_integral(donor=None, acceptor=None, wavelengths=None):
    """
    Calculates the spectral overlap integral defined as the integral of the multiplication of the donor emission
    spectrum normalized to an area of 1, the acceptor molar extinction coefficient as a function of wavelength and the
    wavelength to the power of 4.

    Parameters
    ----------
    donor : 1-D array_like
        Contains emission values of the donor - they don't have to be normalized yet.
    acceptor : 1-D array_like
        Contains the acceptors molar extinction coefficients in 1/(M cm).
    wavelengths : 1-D array_like
        The wavelength values in nm, that correspond to the respective donor and acceptor values.

    Returns
    -------
    spectral_overlap_integral : float
        The value of the spectral overlap integral in (nm**4)/(M cm).
    """
    donor = np.asarray(donor)
    acceptor = np.asarray(acceptor)
    wavelengths = np.asarray(wavelengths)
    if donor.size != acceptor.size or donor.size != wavelengths.size:
        raise AttributeError('donor, acceptor and wavelengths have to be of the same size.')

    donor = donor / np.trapz(donor)  # normalize spectrum to area of 1
    not_integrated = donor * acceptor * wavelengths**4
    spectral_overlap_integral = np.trapz(not_integrated)

    return spectral_overlap_integral


def calculate_fret_rate(distance=10, emission_rate=5e8, spectral_overlap_integral=1e16, dipole_orientation_factor=2/3,
                        refractive_index=1):
    """
    Calculates the Förster resonance energy transfer rate.

    Parameters
    ----------
    distance : float
        In nm.
    emission_rate : float
        In 1/s.
    spectral_overlap_integral : float
        In (nm**4)/(M cm).
    dipole_orientation_factor : float
        The dipole orientation factor κ².
    refractive_index : float
        The refractive index.

    Returns
    -------
    fret_rate : float
        In 1/s.
    """
    fret_rate = 8.785*1e-11 * ((dipole_orientation_factor * emission_rate) /
                               (refractive_index**4 * distance**6)) * spectral_overlap_integral

    return fret_rate


def calculate_fret_efficiency(fret_rate=1e8, fluorescence_lifetime=1e-9):
    """
    Calculates the FRET efficiency.

    Parameters
    ----------
    fret_rate : float
        In 1/s.
    fluorescence_lifetime : float
        The fluorescence lifetime of the donor in absence of the acceptor in s.

    Returns
    -------
    efficiency : float
        The FRET efficiency (dimensionless). Between 0 and 1.
    """
    tau_1 = fluorescence_lifetime
    tau_2 = 1 / (1/fluorescence_lifetime + fret_rate)
    efficiency = 1 - tau_2 / tau_1

    return efficiency
