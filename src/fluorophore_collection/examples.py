from dataclasses import dataclass, field
import src.transitions as tr
from pathlib import Path
import os
import src.formulas as fo
import pandas as pd
from src.abstract_classes import FluorophoreData


@dataclass
class Example(FluorophoreData):
    """
    Contains constant attributes of a fluorophore. 
    """
    # define parameter set
    parameter_set: str = field()

    maximum_extinction_coefficient: float = field(init=False)
    quantum_yield: float = field(init=False)
    fluorescence_lifetime: float = field(init=False)

    # intersystem crossing
    isc_st_rate: float = field(init=False)
    isc_ts_rate: float = field(init=False)

    # photobleaching
    photobleach_t1_rate: float = field(init=False)

    # OFF
    s_to_off_rate: float = field(init=False)
    off_to_s_rate: float = field(init=False)


    def __post_init__(self):
        if self.parameter_set == 'set 1':
            self.maximum_extinction_coefficient = 2.5e5
            self.quantum_yield = 0.27
            self.fluorescence_lifetime = 1e-9
            self.isc_st_rate = 8.3e5
            self.isc_ts_rate = 5e3
            self.photobleach_t1_rate = 1e1
            self.s_to_off_rate = 3e4
            self.off_to_s_rate = 2e-2
            self.j_homo_fret = 1.5e16
            self.fret_kappa_sq = 2/3

        else:
            raise AttributeError("parameter_set has to be one of set 1.")

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

        Returns
        -------
        transitions : Collection
            Contains transitions of type Transition.
        """
        
        wavenumber, wavelength, frequency = fo.convert_wavenumber_wavelength_frequency(wavelength=wavelength)
        photon_flux = fo.calculate_photon_flux(irradiance=irradiance, frequency=frequency)
        excitation_rate = fo.calculate_excitation_rate(photon_flux=photon_flux,
                                                       extinction_coefficient=self.maximum_extinction_coefficient)
        excitation = tr.Transition(rate=excitation_rate, transition_type=tr.TransitionType.EXCITATION)

        emission_rate = fo.calculate_emission_rate(quantum_yield=self.quantum_yield,
                                                   fluorescence_lifetime=self.fluorescence_lifetime)
        emission = tr.Transition(rate=emission_rate, transition_type=tr.TransitionType.FLUORESCENT_EMISSION)

        isc_st = tr.Transition(rate=self.isc_st_rate, transition_type=tr.TransitionType.INTERSYSTEM_CROSSING_ST)

        isc_ts = tr.Transition(rate=self.isc_ts_rate, transition_type=tr.TransitionType.INTERSYSTEM_CROSSING_TS)

        
        internal_conversion_rate = fo.calculate_internal_conversion_rate(quantum_yield=self.quantum_yield,
                                                                         emission_rate=emission_rate,
                                                                         isc_st_rate=self.isc_st_rate)
        internal_conversion = tr.Transition(rate=internal_conversion_rate,
                                         transition_type=tr.TransitionType.INTERNAL_CONVERSION_S)

        s0_off = tr.Transition(tr.TransitionType.H2O_ATTACK_S, self.s_to_off_rate)
        off_s0 = tr.Transition(tr.TransitionType.BACK_REACTION, self.off_to_s_rate)

        energy_transfers = []
        if energy_transfer:
            calculate_homo_fret = {'emission_rate': emission_rate, 'spectral_overlap_integral': self.j_homo_fret,
                                   'dipole_orientation_factor': self.fret_kappa_sq}
            homo_frets = tr.get_energy_transfer_transitions(transition_type=tr.TransitionType.HOMO_FRET, distances=distances,
                                                            rates=None, calculate_rates=calculate_homo_fret)
            energy_transfers = homo_frets
        
        bleach = []
        if bleaching:
            bleach = [tr.Transition(rate=self.photobleach_t1_rate, transition_type=tr.TransitionType.PHOTOBLEACHING_1)]

        transitions = ([excitation, emission, isc_st, isc_ts, internal_conversion, s0_off, off_s0] +energy_transfers + bleach)

        return transitions
