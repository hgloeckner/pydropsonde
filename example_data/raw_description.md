# Sonde description for example data


## Duplicate Sonde id

The two sondes in `duplicate_sonde_id` have the same vaisala sonde id 233211701. both contain data that is not the same.

## empty a-file

The a-file in folder `empty_afile` is empty. The sonde however contains data that should show up in higher levels.

## Circles

The folders `HALO-20240818a` and `HALO-20240831a` each contain sondes belonging to the same circle.
However, in `HALO-20240818a` are only two sondes, so that circle variables should not be calculated.
Folder `HALO-20240818a` contains six sondes with valid data and one sonde without a launch detect - this was added manually and is different in the ORCESTRA original files.
Additionally, sonde 233814536 was erroreously recognized as a minisonde, but should be processed normally. Overall, the Level 4 dataset should contain 2 circles with 8 sondes combined. Both should have data for the sonde variables for all sondes, but only the one with 6 sondes is supposed to have circle data.

## Other Platform

There is another platform folder called P3. There is one sonde in that folder to check whether `pydropsonde` can handle the folder structure.

## overall expectation

The example data level 3 should contain 14 sondes with data (one is dropped because all data fields are empty). The level 4 data should contain 3 circles, one of which should include circle products.
