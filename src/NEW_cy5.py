import src.NEW_transitions as tr
import src.formulas as fo
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

path_absorption = Path(__file__).parent / r'fluorophores\cy5_absorption.xlsx'
dataframe_absorption = pd.read_excel(path_absorption, sheet_name=0, index_col='wavelength [nm]')


def construct_transition_set(irradiance, wavelength, distances, **dstorm_parameters):
    wavenumber, wavelength, frequency = fo.convert_wavenumber_wavelength_frequency(wavelength=wavelength)
    photon_flux = fo.calculate_photon_flux(irradiance, frequency)
    relative_extinction = dataframe_absorption.loc[int(wavelength), 'relative extinction']
    effective_extinction = relative_extinction * Cy5.maximum_extinction_coefficient

    excitation_rate = fo.calculate_excitation_rate(photon_flux=photon_flux, extinction_coefficient=effective_extinction)
    excitation = tr.Transition(rate=excitation_rate, transition_type=tr.TransitionType.EXCITATION)

    emission_rate = fo.calculate_emission_rate(Cy5.quantum_yield, Cy5.fluorescence_lifetime)
    emission = tr.Transition(rate=emission_rate, transition_type=tr.TransitionType.FLUORESCENT_EMISSION)

    isc_st = tr.Transition(rate=Cy5.isc_st_rate, transition_type=tr.TransitionType.INTERSYSTEM_CROSSING_ST)

    isc_ts = tr.Transition(rate=Cy5.isc_ts_rate, transition_type=tr.TransitionType.INTERSYSTEM_CROSSING_TS)

    isomerization = tr.Transition(rate=Cy5.iso_rate, transition_type=tr.TransitionType.ISOMERIZATION)

    biso_rate = fo.calculate_back_isomerization_rate(photon_flux, Cy5.biso_cross_section)
    bisomerization = tr.Transition(rate=biso_rate, transition_type=tr.TransitionType.BACKISOMERIZATION)

    internal_conversion_rate = fo.calculate_internal_conversion_rate(Cy5.quantum_yield, emission_rate,
                                                                     Cy5.iso_rate, Cy5.isc_st_rate)
    internal_conversion = tr.Transition(rate=internal_conversion_rate,
                                        transition_type=tr.TransitionType.INTERNAL_CONVERSION_S)

    dstorm_red_rate = fo.calculate_reduction_rate(k_pet=Cy5.dstorm_red_rate_mol, **dstorm_parameters)
    dstorm_red = tr.Transition(rate=dstorm_red_rate, transition_type=tr.TransitionType.REDUCTION)

    dstorm_oxi = tr.Transition(rate=Cy5.dstorm_oxi_rate, transition_type=tr.TransitionType.OXIDATION)

    calculate_homo_fret = {'emission_rate': emission_rate, 'spectral_overlap_integral': Cy5.j_homo_fret,
                           'dipole_orientation_factor': Cy5.fret_kappa_sq}
    homo_frets = tr.get_energy_transfer_transitions(transition_type=tr.TransitionType.HOMO_FRET, distances=distances,
                                                    rates=None, calculate_rates=calculate_homo_fret)

    calculate_cis_fret = {'emission_rate': emission_rate, 'spectral_overlap_integral': Cy5.j_cis_fret,
                           'dipole_orientation_factor': Cy5.fret_kappa_sq}
    cis_frets = tr.get_energy_transfer_transitions(transition_type=tr.TransitionType.CIS_FRET, distances=distances,
                                                    rates=None, calculate_rates=calculate_cis_fret)

    calculate_triplet_fret = {'emission_rate': emission_rate, 'spectral_overlap_integral': Cy5.j_triplet_fret,
                           'dipole_orientation_factor': Cy5.fret_kappa_sq}
    triplet_frets = tr.get_energy_transfer_transitions(transition_type=tr.TransitionType.TRIPLET_FRET, distances=distances,
                                                    rates=None, calculate_rates=calculate_triplet_fret)

    calculate_off_fret = {'emission_rate': emission_rate, 'spectral_overlap_integral': Cy5.j_off_fret,
                           'dipole_orientation_factor': Cy5.fret_kappa_sq}
    off_frets = tr.get_energy_transfer_transitions(transition_type=tr.TransitionType.OFF_FRET, distances=distances,
                                                    rates=None, calculate_rates=calculate_off_fret)

    transitions = [excitation, emission, isc_st, isc_ts, isomerization, bisomerization, internal_conversion, dstorm_red,
                   dstorm_oxi] + homo_frets + cis_frets + triplet_frets + off_frets
    transition_set = tr.TransitionSet(transitions=transitions)

    return transition_set


@dataclass
class Cy5:
    maximum_extinction_coefficient: float = 2.5e5
    quantum_yield: float = 0.27
    fluorescence_lifetime: float = 1e-9
    isc_st_rate: float = 8.3e5
    isc_ts_rate: float = 5e5
    photobleach_s1_rate: float = 0
    photobleach_t1_rate: float = 0
    photobleach_t2_rate: float = 0
    iso_rate: float = 2e7
    biso_cross_section: float = 1.7e-17

    fret_kappa_sq: float = 2/3

    j_homo_fret: float = 1.55e16
    j_cis_fret: float = 3e16
    j_triplet_fret: float = 9e15
    j_off_fret: float = 1e15

    dstorm_red_rate_mol: float = 9.6e7
    dstorm_oxi_rate: float = 2e-1
