.. _logging:

===========================
Logging
===========================

Fluopy uses Python's logging module for progress and diagnostic messages.
Each module has its own logger, such as fluopy.simulation, under the fluopy logger.

To show informational messages in a script::

    import logging
    import sys

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

To reduce Fluopy's logging to warnings and errors::

    logging.getLogger("fluopy").setLevel(logging.WARNING)

Python warnings are separate from logging. For example, the numerical precision
diagnostic from Simulation.run() is controlled by warn_precision, which defaults
to True. Set warn_precision=False for a run when that diagnostic is unnecessary;
changing the logging level does not suppress it. See
:doc:`model_conventions` for its interpretation.
