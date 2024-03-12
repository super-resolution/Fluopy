# Photoswitching
Continuous time markov chains (CTMC) are used to model the emission behavior of fluorophores. \
The number of fluorophores is limited (e.g., 4) and they are in fixed positions. 
> [!NOTE]
> This resembles e.g., a protein dimer labeled with two different fluorophores [...]

The advantages of this package are: 
- very detailed simulation (information about all transitions and states)
- different simulation options depending on concerns
  - detailed: ✔️ information ❌ time and memory
  - frames: ✔️ fast ❌ only emission data
  - approx: ✔️ information, time ❌ statistically incorrect
  - parallel: ✔️ via Ray ✔️ via GPU
- virtually all transitions possible
- unlimited amount of different energy transfers simultaneously

Parameters needed for accurate simulation:
- Emission and absorption data of involved fluorophores
- photophysical model (states, transitions, rates)
- positions of the fluorophores
- [...]
  
> [!IMPORTANT]
> Sometimes, a transition can 'partially' happen, meaning it deplets a certain initial state but doesn't end up in the intended final state. The same is true for energy transfers: if the acceptor is only roughly defined (e.g., with its name, but without electronic energy levels like S0, S1) it may be needed to define two transitions where one only deplets the donor. 

> [!NOTE]
> In the photophysical context, states and transitions [...]
>
