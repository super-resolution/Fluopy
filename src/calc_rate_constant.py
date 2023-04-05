import numpy as np
from scipy import constants


def excitation_rate(irradiance, wavelength, extinction_coefficient=None, absorption_cross_section=None):
    """
    Returns the excitation rate for a given irradiance and an extinction coefficient or an absorption cross section.

    Parameters
    ----------
    irradiance : float
        Laser irradiance of wavelength in kW/cm².
    wavelength : float
        Wavelength of laser in nm.
    extinction_coefficient : float
        Extinction coefficient of fluorophore at wavelength in 1/(cm M).
    absorption_cross_section : float
        Absorption cross section of fluorophore at wavelength in cm².
        The scattering cross section is assumed to be negligible, hence the absorption cross section equals the
        excitation cross section.

    Returns
    -------
    exc_rate : float
        The excitation rate in 1/s.
    """
    if extinction_coefficient is not None:
        absorption_cross_section = extinction_coefficient * 1e3 * np.log(10) / constants.Avogadro
    absorption_cross_section = absorption_cross_section * 1e-4
    irradiance = irradiance * 1e3 * 1e4
    frequency = constants.speed_of_light/(wavelength * 1e-9)
    photon_flux = irradiance / (constants.h * frequency)

    exc_rate = photon_flux * absorption_cross_section

    return exc_rate


def back_isomerization_rate(irradiance, wavelength, absorption_cross_section):
    """
    Returns the back-isomerization rate for a given irradiance and an absorption cross section of the isomer.

    Parameters
    ----------
    irradiance : float
        Laser irradiance of wavelength w in kW/cm².
    wavelength : float
        Wavelength of laser in nm.
    absorption_cross_section : float
        Absorption cross section of fluorophore isomer at wavelength w in cm².

    Returns
    -------
    back_iso_rate : float
        The back-isomerization rate in 1/s.
    """
    absorption_cross_section = absorption_cross_section * 1e-4
    irradiance = irradiance * 1e3 * 1e4
    frequency = constants.speed_of_light/(wavelength * 1e-9)
    photon_flux = irradiance / (constants.h * frequency)

    back_iso_rate = photon_flux * absorption_cross_section

    return back_iso_rate


def reduction_rate(reducing_agent, concentration, k_pet, ph, same=True):
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
    reduc_rate : float
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
    reduc_rate = k_pet * concentration

    return reduc_rate


def emission_rate(quantum_yield, fluorescence_lifetime):
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
