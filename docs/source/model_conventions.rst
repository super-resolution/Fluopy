.. _model_conventions:

Interpreting events and statistics
=================================

The :doc:`tutorials <../tutorials/tutorials>` demonstrate the main workflows.
This page explains conventions that matter when defining unusual transitions or
interpreting statistics. Times and lifetimes are in seconds, rates in 1/s, and
positions and distances in nm.

Names, identities, and initial states
------------------------------------

A fluorophore's physical identity is its index in
FluorophoreSystem.fluorophores. Use FluorophoreSystem.get_identity() to resolve
an object to that index. Fluorophore names identify shared kinetic groups:
Prediction and Analysis pool their statistics by name. Molecules with the same
name must share the same FluorophoreData object and single-state transition
definitions and rates. Use distinct names for different kinetics.

Ordered pairs with matching donor and acceptor names must define the same paired
transition types at every distance, although their rates may differ. To retain
an inactive record for one such pair, use a zero rate and keep_zero_rates=True.
Pass that option through subsequent filtering or rate adjustment when it is
needed; those operations remove zero-rate records by default. Retained zero-rate
records affect the state space and structural absorbing_states, while
terminal_state_combinations is determined by positive-rate outgoing processes.

start_at is a tuple of SingleState values in physical identity order. The default
is S0 for every molecule. Simulation.run() follows that initial configuration.
For non-absorbing predictions, it selects the reachable component whose
stationary statistics are computed. Initial states with the same reachable
ergodic dynamics produce the same limiting statistics. Transient or absorbing
predictions can depend on start_at.

Events and state changes
------------------------

A transition is an event. It need not change a state. An ordinary single-state
transition concerns one fluorophore, while a paired transition has two ordered
components. The components can both change, only one can change, or neither
can change. A self-transition advances event time and contributes an event
occurrence without ending the current state residence interval.

.. list-table:: Examples of state changes
   :header-rows: 1

   * - Event
     - Initial state
     - Final state
     - Changed components
   * - Single-state excitation
     - S0
     - S1
     - One
   * - Single-state self-transition
     - S0
     - S0
     - None
   * - Paired BOTH
     - S0, S0
     - S1, S1
     - Both
   * - Paired FIRST
     - S0, S0
     - S1, S0
     - First only
   * - Paired SECOND
     - S0, S0
     - S0, S1
     - Second only
   * - Paired NONE
     - S0, S0
     - S0, S0
     - None

This small artificial model makes all four paired outcomes explicit. Its rates
and mechanism name are illustrative. The final table summarizes events actually
observed in a seeded trajectory; its counts fluctuate between realizations.

.. code-block:: python

   import numpy as np
   import pandas as pd

   from fluopy import fluo_data as fd
   from fluopy import fluorophores as fl
   from fluopy import simulation as si
   from fluopy import transitions as tr

   data = fd.FluorophoreData()
   system = fl.FluorophoreSystem([
       fl.Fluorophore("first", [0, 0], constants=data),
       fl.Fluorophore("second", [5, 0], constants=data),
   ])
   self_event = tr.TransitionType("SELF", tr.SingleState.S0, tr.SingleState.S0, False)
   both = tr.TransitionType("BOTH", tr.PairedState.S0_S0, tr.PairedState.S1_S1, False, "EVENT")
   first = tr.TransitionType("FIRST", tr.PairedState.S0_S0, tr.PairedState.S1_S0, False, "EVENT")
   second = tr.TransitionType("SECOND", tr.PairedState.S0_S0, tr.PairedState.S0_S1, False, "EVENT")
   neither = tr.TransitionType("NONE", tr.PairedState.S0_S0, tr.PairedState.S0_S0, False, "EVENT")
   model = tr.TransitionSet({
       "first": [
           tr.Transition(tr.TransitionType.EXCITATION, 3, [0]),
           tr.Transition(tr.TransitionType.FLUORESCENT_EMISSION, 4, [0]),
           tr.Transition(self_event, 2, [0]),
       ],
       "second": [
           tr.Transition(tr.TransitionType.EXCITATION, 3, [1]),
           tr.Transition(tr.TransitionType.FLUORESCENT_EMISSION, 4, [1]),
       ],
       "D: first, A: second, dist: 5.0": [
           tr.Transition(both, 1, [(0, 1)]),
           tr.Transition(first, 1, [(0, 1)]),
           tr.Transition(second, 1, [(0, 1)]),
           tr.Transition(neither, 1, [(0, 1)]),
       ],
   }, system)
   simulation = si.Simulation(model)
   simulation.run(size=500, seed=51)
   combined = model.combined_state_transitions_df
   events = pd.DataFrame({
       "event": combined.loc[simulation.transition_series, "abbreviation"].to_numpy(),
       "changed_components": np.count_nonzero(np.diff(simulation.state_series, axis=1), axis=0),
   })
   events.groupby("event").agg(
       occurrences=("changed_components", "size"),
       changed_components=("changed_components", "max"),
   )

Donor and acceptor are ordered roles
-----------------------------------

The donor and acceptor fields in PairedState identify the first and second
components of an ordered pair. They are also used for interactions where those
physical labels are imperfect. Fluopy does not infer the mechanism from a state
change or abbreviation: paired TransitionType objects require an explicit
mechanism. Single-state transitions use mechanism=None.

The spectral helper derive_energy_transfer_rate() implements a FRET rate law.
Other mechanisms require rates supplied from a suitable model or measurements.
The :doc:`Extending Fluopy notebook <../tutorials/notebooks/Tutorial_extending_fluopy>`
demonstrates a paired PET process and charge recombination.

With kap_sq_var=True, Simulation.run() samples an orientation factor for each
event step and holds it fixed during that waiting interval. Only transitions
whose mechanism is FRET and whose rate exceeds minimum_rate are rescaled.
Their input rates must use the reference orientation factor of 2/3. Other
mechanisms keep their supplied rates but their observed frequencies can change
through competition. This mode requires a fixed event count; end_time cannot
be supplied.

What the statistics count
-------------------------

Transition frequency counts events and normalizes them separately for each
fluorophore name. A paired event counts once and is assigned to the first
component's donor group. State frequencies count visits separately for each
physical fluorophore and are then pooled by name. An event that changes both
components can therefore contribute two state visits while remaining one event.

State occupations measure fractions of observed time. They are not state-visit
frequencies. Analysis includes the final residence in occupations, but excludes
the final, right-censored interval from completed lifetime distributions. A
self-event does not end a state lifetime. For paired events, transition waiting
times are collected from the donor's viewpoint; an unchanged acceptor's lifetime
is not interrupted by the interaction.

Simulation.transition_series indexes combined-transition rows, not the original
transition definitions. The transition_id column of
TransitionSet.combined_state_transitions_df maps those rows back to identities
in TransitionSet.transition_df. A definition can produce multiple combined rows.

Absorbing behavior and method limits
------------------------------------

An individual fluorophore can reach an absorbing state while others remain
active. The joint trajectory ends through absorption when no positive-rate
outgoing event remains. Self-events can keep producing events in an unchanged
configuration. Simulation.run() can also stop at an observation time; its last
residence interval is then censored.

.. list-table:: Choosing a method
   :header-rows: 1

   * - Method
     - Scope and interpretation
   * - Simulation.run()
     - Direct stochastic simulation, including paired transitions, initial
       conditions, and absorption; subject to finite memory and numerical precision.
   * - Prediction
     - At most two physical fluorophores. Paired models provide frequencies but
       not lifetime distributions, mean transition times, lifetimes, or occupations.
       Non-absorbing reachable dynamics need a unique stationary distribution.
   * - Simulation.approximate()
     - A suitable single-fluorophore cycle topology and a matching Prediction.
       Each simple cycle must contain the most frequent state-changing transition.
       Absorbing events are omitted. The generated trajectory may start in a
       different configuration from prediction.start_at.
   * - Dense or sparse matrices
     - Equivalent representations of the same dynamics. Sparse storage reduces
       matrix storage for sufficiently sparse models; it does not remove combined
       state-space growth or expand Prediction's supported scope.

For absorbing predictions, the reachable subchain must lead to absorption with
certainty. The reported transition and occupation statistics describe behavior
before terminal absorption: terminal transitions have frequency zero, and
absorbing states have lifetime inf and frequency and occupation zero. These
conventions differ from counting the absorbing event in a finite simulation.

Simulation can emit FloatingPointPrecisionWarning with an estimate based on
the largest outgoing rate and the resolution of floating-point event times.
Compare the indicated timescale with the actual observation duration when
assessing a model with very fast and very slow processes. The warning does not
mean that the simulation stopped. The diagnostic is enabled by default in
Simulation.run(). Set warn_precision=False for an individual run when it is
unnecessary to repeat the estimate. This controls the diagnostic only; it does
not change the dynamics or improve numerical precision.

Detected photons and pulsed excitation
--------------------------------------

Detection channels are named optical paths. Filtering, routing, and detection
efficiency determine which photons reach them. Restricting fluorophore_ids
represents an emitter-specific collection path; ordinary wavelength-resolved
detection should use spectral responses with that restriction unset. Each
photon can be detected in at most one channel. Detection of paired transitions
that emit a photon is not supported: emitting records must belong to one
fluorophore.

Frame-count series use frame end times in seconds. Extraction retains complete
frames only; photons in a final incomplete frame can remain in arrival times
without appearing in the count series. Added count noise does not create matching
arrival-time events. Binning and thresholds affect blinking statistics.

TCSPC delays are measured relative to the most recent excitation pulse, not
necessarily the pulse that excited the molecule. They are not S1 residence
times. A short repetition interval can change their distribution.

The paired and unpaired delay arrays classify availability immediately before
emission: a positive-rate paired process must be available with the emitter as
its first, active component. The classification does not indicate whether a
paired event occurred earlier. It applies to all paired mechanisms. Only S0
excitation is converted to pulse excitation; other light-dependent rates retain
their supplied values and should correspond to the mean irradiance.
