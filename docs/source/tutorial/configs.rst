Config Files
============


Each config file has a mandatory section that contains the data directory of the dropsonde data like

.. code-block:: ini

        [MANDATORY]
        data_directory = ./example_data

Technically, this is everything a valid config file needs. However, you can specify the place of your flight folders and of the actual dropsonde data in an optional section.
For the example data this would be

.. code-block:: ini

        [OPTIONAL]
        path_to_flight_ids = {platform}/Level_0
        path_to_l0_files = {platform}/Level_0/{flight_id}

If there are any global attributes that should be added to any dataset that is saved, those can be added to the optional section. For example

.. code-block:: ini

        [OPTIONAL]
        global_attrs = author:Geet George,author_email:g.george@tudelft.nl

would adds the attributes ``author : Geet George`` and ``author_email : g.george@tudelft.nl``. Be careful to not add : or , into the attrubutes you want added. Those are used as separators.

Other optional function parameters can be specified with their function as a section. To specify the L2 and L3 filenames this could be

.. code-block:: ini

        [processor.Sonde.get_l2_filename]
        l2_filename_template = PERCUSION_{platform}_{launch_time}_{flight_id}_{serial_id}_Level_2.nc
        [processor.Gridded.get_l3_filename]
        l3_filename_template = PERCUSION_{platform}_{flight_id}_Level_3.nc
