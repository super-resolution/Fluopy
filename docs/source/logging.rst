.. _logging:

===========================
Logging
===========================

Fluopy allows logging of various messages that are mostly related to inform
and warn about critical configuration and simulation issues.

A specific logger is defined for each individual package
with a name identical to the module name.

You can set up an individual logging configuration.

To start with you might want to use something like::

    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

