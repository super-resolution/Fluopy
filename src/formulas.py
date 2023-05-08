import numpy as np
from scipy import constants


def convert_wavenumber_wavelength_frequency(wavenumber=None, wavelength=None, frequency=None):
    """
    Convert either wavenumber, wavelength or frequency into the other two.

    Parameters
    ----------
    wavenumber : float
        In 1/cm.
    wavelength : float
        In nm.
    frequency : float
        In Hz.

    Returns
    -------
    wavenumber : float
        In 1/cm.
    wavelength : float
        In nm.
    frequency : float
        In Hz.
    """
    if wavenumber is not None:
        wavelength = 1/(wavenumber * 1e2) * 1e9
        frequency = wavenumber * 1e2 * constants.c

    elif wavelength is not None:
        wavenumber = 1/(wavelength * 1e-9) * 1e-2
        frequency = constants.c / (wavelength * 1e-9)

    elif frequency is not None:
        wavenumber = frequency / constants.c * 1e-2
        wavelength = constants.c / frequency * 1e9

    else:
        raise ValueError('One of wavenumber, wavelength and frequency must not be None.')

    return wavenumber, wavelength, frequency


def calculate_photon_flux(irradiance, frequency):
    """
    Calculates the photon flux based on the irradiance and the frequency of the light.

    Parameters
    ----------
    irradiance : float
        The irradiance in kW/cm².
    frequency : float
        The frequency in Hz.

    Returns
    -------
    photon_flux : float
        The photon flux in 1/(m² s).
    """
    irradiance = irradiance * 1e3 * 1e4
    photon_flux = irradiance / (constants.h * frequency)

    return photon_flux


def calculate_excitation_rate(photon_flux, extinction_coefficient=None, absorption_cross_section=None):
    """
    Returns the excitation rate for a given irradiance and an extinction coefficient or an absorption cross section.

    Parameters
    ----------
    photon_flux : float
        The photon flux in 1/(m² s).
    extinction_coefficient : float
        Extinction coefficient of fluorophore at wavelength in 1/(cm M).
    absorption_cross_section : float
        Absorption cross section of fluorophore at wavelength in cm².
        The scattering cross section is assumed to be negligible, hence the absorption cross section equals the
        excitation cross section.

    Returns
    -------
    excitation_rate : float
        The excitation rate in 1/s.
    """
    if extinction_coefficient is not None:
        absorption_cross_section = extinction_coefficient * 1e3 * np.log(10) / constants.Avogadro
    absorption_cross_section = absorption_cross_section * 1e-4

    excitation_rate = photon_flux * absorption_cross_section

    return excitation_rate


def calculate_emission_rate(quantum_yield, fluorescence_lifetime):
    """
    Returns the rate of fluorescent emission based on the quantum yield and the fluorescence lifetime.

    Parameters
    ----------
    quantum_yield : float
        Number between 0 and 1.
    fluorescence_lifetime : float
        The fluorescence lifetime in s.

    Returns
    -------
    emis_rat : float
        The rate of emission in 1/s.
    """
    emis_rate = quantum_yield / fluorescence_lifetime

    return emis_rate


def calculate_internal_conversion_rate(quantum_yield, emission_rate, *other_outgoing_rates):
    """
    Calculates the rate of internal conversion from the first excited state to the vibrationally excited but
    electronic ground state.

    Parameters
    ----------
    quantum_yield : float
        Number between 0 and 1.
    emission_rate : float
        The rate of emission in 1/s.
    other_outgoing_rates : floats
        Rates of all other transitions (except fluorescence emission) that leave the first excited state in 1/s.

    Returns
    -------
    internal_conversion_rate : float
        The rate of internal conversion in 1/s.
    """
    internal_conversion_rate = emission_rate / quantum_yield - emission_rate
    for outgoing_rate in other_outgoing_rates:
        internal_conversion_rate -= outgoing_rate

    return internal_conversion_rate


def calculate_back_isomerization_rate(photon_flux, absorption_cross_section):
    """
    Returns the back-isomerization rate for a given irradiance and an absorption cross section of the isomer.

    Parameters
    ----------
    photon_flux : float
        The photon flux in 1/(m² s).
    absorption_cross_section : float
        Absorption cross section of fluorophore isomer at wavelength w in cm².

    Returns
    -------
    back_isomerization_rate : float
        The back-isomerization rate in 1/s.
    """
    absorption_cross_section = absorption_cross_section * 1e-4

    back_isomerization_rate = photon_flux * absorption_cross_section

    return back_isomerization_rate


def calculate_reduction_rate(reducing_agent='mea', concentration=100, k_pet=None, ph=8, same=True):
    """
    Returns the dSTORM reduction rate for a given reducing agent and its concentration.

    Parameters
    ----------
    reducing_agent : str
        One of 'mea', 'βME'.
    concentration : float
        Concentration of the reducing agent in mM.
    k_pet : float
        The rate of photoinduced electron transfer in 1/(s M).
    ph : float
        The pH as indicator of acidity or basicity.
    same : bool
        Whether the reducing_agent corresponds to the k_pet.

    Returns
    -------
    reduction_rate : float
        The reduction rate in 1/s.
    """
    # the factor 1/7 (or 7) comes from protocols stating to either use 100 µl 100 mM MEA or 10 µl 143 mM beta-ME
    if ph != 8:
        print('Not implemented yet')
        # some function (inverse sigmoid) to adjust the rates

    if not same:
        if reducing_agent == 'mea':
            k_pet = k_pet * 1/7
        elif reducing_agent == 'βME':
            k_pet = k_pet * 7

    concentration = concentration * 1e-3
    reduction_rate = k_pet * concentration

    return reduction_rate


def calculate_fret_rate(distance, emission_rate, spectral_overlap_integral, dipole_orientation_factor, constant=1):
    """
    Calculates the Förster resonance energy transfer rate.

    Parameters
    ----------
    distance : float
        In nm.
    emission_rate : float
        In 1/s.
    spectral_overlap_integral : float
        In 1/(M cm).
    dipole_orientation_factor : float
        The dipole orientation factor κ².
    constant : float
        To adjust for the given units.

    Returns
    -------
    fret_rate : float
        In 1/s.
    """
    fret_rate = constant * ((dipole_orientation_factor**2 * emission_rate) / distance**6) * spectral_overlap_integral

    return fret_rate
