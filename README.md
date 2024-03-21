# Overview
Continuous time markov chains (CTMC) are used to model the emission behavior of fluorophores. \
The number of fluorophores is limited (e.g., 4) and they are in fixed positions. 
> [!NOTE]
> This resembles e.g., a protein dimer labeled with two different fluorophores. Even though the protein is likely mobile, the relative positions of the fluorophores are expected to be constant. Rotational movement or restrictions of such can be handled with the dipole orientation factor &kappa;².

The package offers: 
- very detailed simulation (information about all transitions and states)
- different simulation options depending on concerns
  - detailed: ✔️ information ❌ time and memory
  - frames: ✔️ low memory ❌ time, only emission data
  - approx: ✔️ information, time ❌ statistically incorrect
  - parallel: ✔️ via Ray ✔️ via GPU
- virtually all transitions possible
- unlimited amount of different energy transfers simultaneously

Parameters needed for accurate simulation:
- emission and absorption data of involved fluorophores
- photophysical model (states, transitions, rates)
- positions of the fluorophores
- laser irradiance [W/m²], wavelength [m]
- frame integration time [s]
- bandpass range
- EMCCD gain
- photon collection rate objective
- quantum efficiency detector
- dark noise, readout noise
  
> [!IMPORTANT]
> Sometimes, a transition can 'partially' happen, meaning it depletes a certain initial state but doesn't end up in the intended final state. The same is true for energy transfers: if the acceptor is only roughly defined (e.g., with its name, but without electronic energy levels like S0, S1) it may be needed to define two transitions where one only depletes the donor. 

> [!NOTE]
> Markov chain topology and the photophysical context share some terms and one should be aware of the differences: due to computational advantage (explicit information), the photophysical transitions represent the Markovian states during simulation. The Markovian absorbing states correspond to the photophysically bleached states. 

*Code structure*

![code structure](https://github.com/super-resolution/Photoswitching/blob/main/images/code_structure.png)

# Customization
The package provides spectroscopic data for some fluorophores. If the fluorophore of interest is not among them, add data files to src\fluorophore_collection (note the file names, see derive_energy_transfer_transitions) and create a fluorophore dataclass in fluo_data.py. \
If needed transitions are not available, the following steps have to be taken: 
- the rate constant has to be defined in fluo_data.FluorophoreData and in the specific fluorophore dataclass
  - if the rate depends on other factors, define a function in formulas.py and call it within transitions.py\derive_transitions()
- the involved photophysical states have to be available in transitions.py\SingleState, if not - add them
  - if the transition is an energy transfer, the involved initial and final state combination has to available in transitions.py\PairedState
- the transition has to be defined in transitions.py\TransitionType
