Data
=====


This section outlines procedures for how to go about doing the quality control, the processing and the generation of data products.

Data Directory Structure
------------------------

Usually dropsonde data has a ``platform`` from which it was meassured and a ``flight_id`` from the exact flight it was taken from. After the processing each flight will have data on different levels of processing. ``pydropsonde`` uses the following levels.

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Level
     - Description
   * - Level_0
     - raw files from the AVAPS software (A-files, D-files, ...). Some files contain auxiliary data for a measurement sequence (usually a flight) and others are per sonde.
   * - Level_1
     - ASPEN-processed netCDF files per sonde. First QC applied.
   * - :doc:`Level_2 <level2>`
     - Sondes that passed additional QC tests. Still one file per sonde. All soundings with no usable data are excluded
   * - :doc:`Level_3 <level3>`
     - One dataset (file) containing all Level_2 soundings gridded on a uniform, vertical grid, with some derived variables
   * - Level_4
     - Circle products from all circles flown during flight or measurement campaign



You can define your exact folder structure as shown in :doc:`example configs <../tutorial/configs>`

.. toctree::
    :caption: Level Description

    level2
    level3
