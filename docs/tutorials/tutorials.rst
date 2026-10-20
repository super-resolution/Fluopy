.. _tutorials:

===========================
Tutorials
===========================

Start with a single simulation, then choose a tutorial for the question you want
to explore. Each Jupyter notebook runs independently and uses explicit random
seeds. The examples demonstrate Fluopy; their illustrative parameters should be
adapted to the experimental system before quantitative interpretation.


.. toctree::
   :maxdepth: 1
   :caption: Guided tutorials
   :name: toc_tutorials_data

   Introduction to Fluopy <./notebooks/01_Introduction.ipynb>
   Interacting fluorophores <./notebooks/02_Interacting_fluorophores.ipynb>
   From emitted photons to measured signals <./notebooks/03_Detected_photons.ipynb>
   Pulsed excitation and TCSPC <./notebooks/04_TCSPC.ipynb>

The first tutorial covers model setup, simulation, prediction, and a photon trace.
The interacting example adds competing pathways and optional sparse storage and
orientation sampling. The signal and TCSPC tutorials focus on what detectors
observe under continuous and pulsed excitation, respectively.

For custom models and precise interpretation rules:

.. toctree::
   :maxdepth: 1
   :caption: Advanced modeling

   Extending Fluopy <./notebooks/Tutorial_extending_fluopy.ipynb>

See :doc:`../source/model_conventions` for event counting, state lifetimes,
paired-transition roles, and the limits of prediction and approximation.


.. toctree::
   :maxdepth: 1
   :caption: Build information
   :name: notebook_execution_table

   Notebook execution status <notebook_execution_table>
