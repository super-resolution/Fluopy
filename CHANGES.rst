=======================
Changelog
=======================

0.4.x - dev
=================

New Features
------------
- allow fluorophore spectra to be supplied as arrays or loaded from CSV files
- derive transitions and FRET rates from user-defined fluorophore data
- allow custom single states, paired states and transition types
- add a public function for deriving energy-transfer rates from spectra

API Changes
-----------
- store emission and absorption spectra directly in FluorophoreData
- replace fixed state and transition-type enums with extensible value objects

Bug Fixes
---------
- include states occurring only in paired state transitions in the transition set state
space
- reject different fluorophore data assigned to the same fluorophore name

Other Changes and Additions
---------------------------
- expand the extending-Fluopy tutorial with customization examples


0.3.0 - 2026-06-02
=================

API Changes
-----------
- include more functions in __all__

Bug Fixes
---------
- fix tests on linux by sorting file names from iterdir()

Other Changes and Additions
---------------------------
- some changes to tutorials
- add tutorial on distributed simulation
- some changes to GitHub actions
- modify Dockerfile and add .dockerignore

0.2.0 - 2026-05-19
=====================

Other Changes and Additions
---------------------------
- fix for pandas 3.0 and python 3.14
- activate read-the-docs
- activate coverage
- refactor GitHub Action workflows

0.1.0 - 2026-05-13
=====================

Other Changes and Additions
---------------------------
- initial release
