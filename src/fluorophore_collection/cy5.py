from dataclasses import dataclass, field
import src.transitions as tr
from pathlib import Path
import os
import src.formulas as fo
import pandas as pd
from src.abstract_classes import FluorophoreData


MAXIMUM_EXTINCTION_COEFFICIENT = 2.5e5
QUANTUM_YIELD = 0.27
FLUORESCENCE_LIFETIME = 1e-9

PHOTOBLEACHING = 1


# energy transfers
KAPPA_SQUARED = 2/3
J_HOMO_FRET = 1.55e16  # donor: S1, acceptor: S0
J_CIS_FRET = 3e16  # donor: S1, acceptor: Cis
J_TRIPLET_FRET = 9e15  # donor: S1, acceptor: T1
J_OFF_FRET = 1e15  # donor: S1, acceptor: OFF


# dSTORM
PET_T_RATE_MOL = 1e8
PET_S_RATE_MOL = 1e9
THERMAL_ELIM_RATE = 2e-2


@dataclass
class Cy5(FluorophoreData):
    """
    Contains constant attributes of the fluorophore Cy5. The values correspond to Set1 as defined in [...].
    """
    # define parameter set
    parameter_set: str = field()

    maximum_extinction_coefficient: float = MAXIMUM_EXTINCTION_COEFFICIENT
    quantum_yield: float = QUANTUM_YIELD
    fluorescence_lifetime: float = FLUORESCENCE_LIFETIME

    # intersystem crossing
    isc_st_rate: float = field(init=False)
    isc_ts_rate: float = field(init=False)

    # photobleaching
    photobleach_t1_rate: float = PHOTOBLEACHING

    # cis/trans isomerization
    iso_rate: float = field(init=False)
    biso_cross_section: float = field(init=False)

    # dipole orientation factor
    fret_kappa_sq: float = KAPPA_SQUARED
    # spectral overlap integral
    j_homo_fret: float = J_HOMO_FRET
    j_cis_fret: float = J_CIS_FRET
    j_triplet_fret: float = J_TRIPLET_FRET
    j_off_fret: float = J_OFF_FRET

    # dstorm-specific attributes
    dstorm_pet_t_rate_mol: float = PET_T_RATE_MOL
    dstorm_pet_s_rate_mol: float = PET_S_RATE_MOL
    dstorm_th_el_rate: float = THERMAL_ELIM_RATE

    def __post_init__(self):
        if self.parameter_set == 'test':
            self.maximum_extinction_coefficient = 2.5e5
            self.quantum_yield = 0.27
            self.fluorescence_lifetime = 1e-9
            self.isc_st_rate = 8.3e5
            self.isc_ts_rate = 5e3
            self.photobleach_t1_rate = 0
            self.iso_rate = 2e7
            self.biso_cross_section = 1.7e-17
            self.fret_kappa_sq = 2 / 3
            self.j_homo_fret = 1.55e16
            self.j_cis_fret = 3e16
            self.j_triplet_fret = 9e15
            self.j_off_fret = 1e15
            self.dstorm_pet_t_rate_mol = 1e8
            self.dstorm_pet_s_rate_mol = 1e9
            self.dstorm_th_el_rate = 2e-2
        elif self.parameter_set == 'set 1':
            self.isc_st_rate = 8.3e5
            self.isc_ts_rate = 5e5
            self.iso_rate = 2e7
            self.biso_cross_section = 1.7e-17
        elif self.parameter_set == 'set 2':
            self.isc_st_rate = 8.3e5
            self.isc_ts_rate = 5e5
            self.iso_rate = 2e7
            self.biso_cross_section = 1.7e-17
        else:
            raise AttributeError("parameter_set has to be one of 'test', 'set 1' and 'set 2'.")

    def derive_transitions(self, irradiance=2, wavelength=600, bleaching=False, energy_transfer=True, distances=None,
                           dstorm=True, **dstorm_parameters):
        """
        Derives transitions based on the experimental conditions to be mimicked.

        Parameters
        ----------
        irradiance : float
            Irradiance in kW/cm².
        wavelength : float
            Wavelength in nm.
        bleaching : bool
            Whether to incooperate bleaching as a possible transition.
        energy_transfer : bool
            Whether to incooperate energy transfers as possible transitions.
        distances : dict
            Contains tuples of 2 fluorophore ids as keys and their distance as values.
            Only needed if energy_transfer is True.
        dstorm : bool
            Whether to incooperate dstorm photoswitching as possible transitions.
        dstorm_parameters : dict
            May contain the following keys: reducing_agent, concentration, k_pet, ph.
            Only needed if dstorm is True.

        Returns
        -------
        transitions : Collection
            Contains transitions of type Transition.
        """
        path_absorption = os.path.join(Path(__file__).parent, 'cy5_data', 'cy5_rel_absorption.csv')
        dataframe_absorption = pd.read_csv(filepath_or_buffer=path_absorption, index_col=0)

        wavenumber, wavelength, frequency = fo.convert_wavenumber_wavelength_frequency(wavelength=wavelength)
        photon_flux = fo.calculate_photon_flux(irradiance=irradiance, frequency=frequency)
        relative_extinction = dataframe_absorption.loc[int(wavelength), 'relative extinction']
        effective_extinction = relative_extinction * self.maximum_extinction_coefficient

        excitation_rate = fo.calculate_excitation_rate(photon_flux=photon_flux,
                                                       extinction_coefficient=effective_extinction)
        excitation = tr.Transition(rate=excitation_rate, transition_type=tr.TransitionType.EXCITATION)

        emission_rate = fo.calculate_emission_rate(quantum_yield=self.quantum_yield,
                                                   fluorescence_lifetime=self.fluorescence_lifetime)
        emission = tr.Transition(rate=emission_rate, transition_type=tr.TransitionType.FLUORESCENT_EMISSION)

        isc_st = tr.Transition(rate=self.isc_st_rate, transition_type=tr.TransitionType.INTERSYSTEM_CROSSING_ST)

        isc_ts = tr.Transition(rate=self.isc_ts_rate, transition_type=tr.TransitionType.INTERSYSTEM_CROSSING_TS)

        isomerization = tr.Transition(rate=self.iso_rate, transition_type=tr.TransitionType.ISOMERIZATION)

        biso_rate = fo.calculate_back_isomerization_rate(photon_flux=photon_flux,
                                                         absorption_cross_section=self.biso_cross_section)
        bisomerization = tr.Transition(rate=biso_rate, transition_type=tr.TransitionType.BACKISOMERIZATION)

        internal_conversion_rate = fo.calculate_internal_conversion_rate(quantum_yield=self.quantum_yield,
                                                                         emission_rate=emission_rate,
                                                                         iso_rate=self.iso_rate,
                                                                         isc_st_rate=self.isc_st_rate)
        internal_conversion = tr.Transition(rate=internal_conversion_rate,
                                         transition_type=tr.TransitionType.INTERNAL_CONVERSION_S)

        dstorm_transitions = []
        if dstorm:
            dstorm_pet_t_rate = fo.calculate_pet_rate(k_pet=self.dstorm_pet_t_rate_mol, **dstorm_parameters)
            dstorm_pet_s_rate = fo.calculate_pet_rate(k_pet=self.dstorm_pet_s_rate_mol, **dstorm_parameters)
            dstorm_pet_t = tr.Transition(rate=dstorm_pet_t_rate, transition_type=tr.TransitionType.ET_CYCLE_T)
            dstorm_pet_s = tr.Transition(rate=dstorm_pet_s_rate, transition_type=tr.TransitionType.ET_CYCLE_S)
            dstorm_red_t_rate = dstorm_pet_t_rate / 1000
            dstorm_red_s_rate = dstorm_pet_s_rate / 1000
            dstorm_red_t = tr.Transition(rate=dstorm_red_t_rate, transition_type=tr.TransitionType.REDUCTION_T)
            dstorm_red_s = tr.Transition(rate=dstorm_red_s_rate, transition_type=tr.TransitionType.REDUCTION_S)
            dstorm_oxi = tr.Transition(rate=self.dstorm_th_el_rate, transition_type=tr.TransitionType.OXIDATION)
            dstorm_transitions = [dstorm_pet_t, dstorm_pet_s, dstorm_red_t, dstorm_red_s, dstorm_oxi]

        energy_transfers = []
        if energy_transfer:
            calculate_homo_fret = {'emission_rate': emission_rate, 'spectral_overlap_integral': self.j_homo_fret,
                                   'dipole_orientation_factor': self.fret_kappa_sq}
            homo_frets = tr.get_energy_transfer_transitions(transition_type=tr.TransitionType.HOMO_FRET, distances=distances,
                                                         rates=None, calculate_rates=calculate_homo_fret)

            calculate_cis_fret = {'emission_rate': emission_rate, 'spectral_overlap_integral': self.j_cis_fret,
                                  'dipole_orientation_factor': self.fret_kappa_sq}
            cis_frets = tr.get_energy_transfer_transitions(transition_type=tr.TransitionType.CIS_FRET_1, distances=distances,
                                                        rates=None, calculate_rates=calculate_cis_fret)

            calculate_triplet_fret = {'emission_rate': emission_rate, 'spectral_overlap_integral': self.j_triplet_fret,
                                      'dipole_orientation_factor': self.fret_kappa_sq}
            triplet_frets = tr.get_energy_transfer_transitions(transition_type=tr.TransitionType.S_T_ANNIHILATION,
                                                            distances=distances, rates=None,
                                                            calculate_rates=calculate_triplet_fret)

            calculate_off_fret = {'emission_rate': emission_rate, 'spectral_overlap_integral': self.j_off_fret,
                                  'dipole_orientation_factor': self.fret_kappa_sq}
            off_frets = tr.get_energy_transfer_transitions(transition_type=tr.TransitionType.OFF_FRET, distances=distances,
                                                        rates=None, calculate_rates=calculate_off_fret)
            energy_transfers = (homo_frets + cis_frets + triplet_frets + off_frets)

        bleach = []
        if bleaching:
            bleach = [tr.Transition(rate=self.photobleach_t1_rate, transition_type=tr.TransitionType.PHOTOBLEACHING_1)]

        transitions = ([excitation, emission, isc_st, isc_ts, isomerization, bisomerization, internal_conversion] +
                       dstorm_transitions + bleach + energy_transfers)

        return transitions
