import numpy as np
import warnings

import pydropsonde.helper.xarray_helper as hx


class QualityControl:
    """
    Helper class to handle quality control functions and flags in a sonde object
    """

    def __init__(
        self,
    ) -> None:
        self.qc_vars = []
        self.qc_flags = {}
        self.qc_details = {}
        self.qc_by_var = {}
        self.alt_dim = "time"

    def set_qc_variables(self, qc_variables):
        self.qc_vars = self.qc_vars + list(qc_variables)
        for variable in self.qc_vars:
            self.qc_by_var.update({variable: dict(qc_flags={}, qc_details={})})

    def get_is_floater(
        self,
        aspen_ds,
        gpsalt_threshold: float = 25,
        consecutive_time_steps: int = 3,
    ):
        """
        Add a qc flag, whether a sonde is a floater to a qc_flag object

        Parameters
        ----------
        gpsalt_threshold : float, optional
            The gpsalt altitude below which the sonde will check for time periods when gpsalt and pres have not changed. Default is 25.
        consecutive_time_steps : float, optional
            The number of timestapes that have to be at roughly  the same height and pressure to set landing time. default is 3

        Return
        ------
        Estimated landing time for floater or None
        """
        gpsalt_threshold = float(gpsalt_threshold)

        surface_ds = (
            aspen_ds.where(aspen_ds.gpsalt < gpsalt_threshold, drop=True)
            .sortby("time")
            .dropna(dim="time", how="any", subset=["pres", "gpsalt"])
        )
        gpsalt_diff = np.diff(surface_ds.gpsalt)
        pressure_diff = np.diff(surface_ds.pres)
        gpsalt_diff_below_threshold = (
            np.abs(gpsalt_diff) < 1
        )  # GPS altitude value at surface shouldn't change by more than 1 m
        pressure_diff_below_threshold = (
            np.abs(pressure_diff) < 1
        )  # Pressure value at surface shouldn't change by more than 1 hPa
        floater = gpsalt_diff_below_threshold & pressure_diff_below_threshold
        if np.any(floater):
            self.is_floater = True
        else:
            self.is_floater = False
            return None
        for time_index in range(len(floater) - consecutive_time_steps + 1):
            if np.all(floater[time_index : time_index + consecutive_time_steps]):
                landing_time = surface_ds.time[time_index - 1].values
                print(
                    f"{aspen_ds.attrs['SondeId']}: Floater detected! The landing time is estimated as {landing_time}."
                )
                return landing_time
        print(
            f"{aspen_ds.attrs['SondeId']}: Floater detected! However, the landing time could not be estimated. Therefore setting landing time as {surface_ds.time[0].values}"
        )
        return surface_ds.time[0].values

    def profile_fullness(
        self,
        ds,
        variable_dict={"u": 4, "v": 4, "rh": 2, "ta": 2, "p": 2},
        time_dimension="time",
        timestamp_frequency=4,
        fullness_threshold=0.8,
    ):
        """
        Calculates the profile coverage for a given set of variables, considering their sampling frequency.

        This function assumes that the time_dimension coordinates are spaced over 0.25 seconds,
        implying a timestamp_frequency of 4 hertz. This is applicable for ASPEN-processed QC and PQC files,
        specifically for RD41.

        For each variable in the variable_dict that is in self.qc_vars, the function calculates the fullness fraction. If the fullness
        fraction is less than the fullness_threshold, it sets the entry "profile_fullness_{variable}" in `self.qc_flag` to False.
        Otherwise, it sets this entry to True.

        For each variable in the variable_dict  that is in self.qc_vars, the function adds the fullness fraction to the qc_details dictionary

        Parameters
        ----------
        ds : dataset to run near_surface_coverage on
        variable_dict : dict, optional
            Dictionary containing the variables in `self.aspen_ds` and their respective sampling frequencies.
            The function will estimate the weighted profile-coverage for these variables.
            Default is {'u_wind':4,'v_wind':4,'rh':2,'tdry':2,'pres':2}.
        time_dimension : str, optional
            The independent dimension of the profile. Default is "time".
        timestamp_frequency : numeric, optional
            The sampling frequency of `time_dimension` in hertz. Default is 4.
        fullness_threshold : float or str, optional
            The threshold for the fullness fraction. If the calculated fullness fraction is less than this threshold,
            the profile is considered not full. Default is 0.8.


        """
        var_keys = set(variable_dict.keys())
        if set(var_keys) != set(self.qc_vars):
            var_keys = set(var_keys) & set(self.qc_vars)
            warnings.warn(
                f"variables for which frequency is given do not match the qc_variables. Continue for the intersection  {var_keys}"
            )
        for variable in var_keys:
            dataset = ds[variable]
            sampling_frequency = variable_dict[variable]
            weighed_time_size = len(dataset[time_dimension]) / (
                timestamp_frequency / sampling_frequency
            )
            fullness_fraction = np.sum(~np.isnan(dataset.values)) / weighed_time_size
            if fullness_fraction < fullness_threshold:
                self.qc_flags[f"{variable}_profile_fullness"] = False
            else:
                self.qc_flags[f"{variable}_profile_fullness"] = True
            self.qc_details[f"{variable}_profile_fullness_fraction"] = fullness_fraction

    def near_surface_coverage(
        self,
        ds,
        alt_bounds=[0, 1000],
        alt_dim="gpsalt",
        count_threshold=50,
    ):
        """
        Calculates the fraction of non-null values in specified variables near the surface.


        For each variable in self.qc_variable, the function calculates the near surface count. If the near surface count is less than
        the fullness_threshold, it sets the entry "near_surface_{variable}" in `self.qc_flag` to False.
        Otherwise, it sets this entry to True.

        For each variable in self.qc_vars, the function adds the near surface count to the qc_details dictionary


        Parameters
        ----------
        ds : dataset to run near_surface_coverage on
        alt_bounds : list, optional
            The lower and upper bounds of altitude in meters to consider for the calculation. Defaults to [0,1000].
        alt_dim : str, optional
            The name of the altitude dimension. Defaults to "alt". If the sonde is a floater, this will be set to "gpsalt" regardless of user-provided value.
        count_threshold : int, optional
            The minimum count of non-null values required for a variable to be considered as having near surface coverage. Defaults to 50.


        """

        count_threshold = int(count_threshold)

        if isinstance(alt_bounds, str):
            alt_bounds = alt_bounds.split(",")
            alt_bounds = [float(alt_bound) for alt_bound in alt_bounds]
        try:
            if self.is_floater and not (alt_dim == "gpsalt"):
                warnings.warn(
                    f"{ds.attrs['SondeId']} was detected as a floater but you did not chose gpsalt as altdim in the near surface coverage qc"
                )
        except KeyError:
            warnings.warn(
                f"{ds.attrs['SondeId']} has not been checked for being a floater. Please run is_floater first."
            )

        for variable in self.qc_vars:
            dataset = ds.where(
                (ds[alt_dim] > alt_bounds[0]) & (ds[alt_dim] < alt_bounds[1]), drop=True
            )
            near_surface_count = dataset[variable].count()
            if near_surface_count < count_threshold:
                self.qc_flags[f"{variable}_near_surface"] = False

            else:
                self.qc_flags[f"{variable}_near_surface"] = True
            self.qc_details[f"{variable}_near_surface_count"] = (
                near_surface_count.values
            )

    def alt_near_gpsalt(self, ds, diff_threshold=150):
        """
        Calculates the mean difference between msl altitude and gpsaltitude in the dataset

        For each variable in self.qc_variable, the function calculates the mean difference between altitude and gpsaltitude.
        If the difference is greater than the diff_threshold, it sets the entry "mean_alt_gpsalt_diff" in `self.qc_flag` to False.
        Otherwise, it sets this entry to True.

        For each variable in self.qc_vars, the function adds the mean alt to gpsalt difference to the qc_details dictionary


        Parameters
        ----------
        ds : dataset to run near_surface_coverage on
        diff_threshold : accepted difference between altitude and gpsaltitude. Default is 150m

        """

        dataset = ds[["alt", "gpsalt"]]
        if not self.qc_flags.get(f"{self.alt_dim}_values", True):
            return 0

        max_diff = np.abs((dataset.alt - dataset.gpsalt).max(skipna=True))
        if max_diff < diff_threshold:
            self.qc_flags["alt_near_gpsalt"] = True
        else:
            self.qc_flags["alt_near_gpsalt"] = False
        self.qc_details["alt_near_gpsalt_max_diff"] = max_diff.values

    def low_physics(self, ds, rh_min=0.3, ta_min=293.15, alt_dim="gpsalt"):
        """
        Checks that the temperature and relative humidity in the lowest 100m in a dataset
        are above a certain value


        Parameters
        ----------
        self : object
        The object containing the necessary attributes and methods.
        ds : xarray.Dataset
        The dataset to check for low physics conditions.
        alt_dim : str, optional
        The dimension name of the altitude coordinate (default is "gpsalt").

        Returns
        -------
        None
        """
        ds_check = ds.where(ds[alt_dim] < 100, drop=True)
        if ds_check.sizes["time"] == 0:
            self.qc_flags["low_physics"] = False
            self.qc_details["low_physics_rh_min"] = np.nan
            self.qc_details["low_physics_ta_min"] = np.nan
        else:
            self.qc_flags["low_physics"] = (ds_check.rh.min() > float(rh_min)) and (
                ds_check.ta.min() > float(ta_min)
            )
            self.qc_details["low_physics_rh_min"] = ds_check.rh.min().values
            self.qc_details["low_physics_ta_min"] = ds_check.ta.min().values

    def check_qc(self, used_flags=None, check_ugly=True):
        """
        check if any qc check has failed.
        If any has failed, return False, if not True

        Parameters:
        -----------
        used_flags: string or list
            list of qc flags to check
        """
        if used_flags is None:
            used_flags = []
        elif used_flags == "all":
            used_flags = list(self.qc_flags.keys()).copy()
        elif isinstance(used_flags, str):
            used_flags = used_flags.split(",")
            if (len(used_flags) == 1) and used_flags[0].startswith("all_except_"):
                all_flags = self.qc_flags.copy()
                all_flags.pop(used_flags[0].replace("all_except_", ""))
                used_flags = all_flags.copy()
            elif used_flags[0].startswith("all_except_"):
                raise ValueError(
                    "If 'all_except_<prefix>' is provided in filter_flags, it should be the only value."
                )
        if not all(flag in self.qc_flags for flag in used_flags):
            raise ValueError(
                "not all flags are in the qc dict. please check you ran all qc tests"
            )

        used_flags = {key: self.qc_flags[key] for key in used_flags}
        if check_ugly and all(used_flags.values()):
            return True
        elif (not check_ugly) and any(used_flags.values()):
            return True
        else:
            return False

    def get_qc_by_var(self):
        """
        Organizes quality control (QC) flags and details by variable.

        This method iterates over each variable in `self.qc_vars` and filters the
        `self.qc_flags` and `self.qc_details` dictionaries to include only the keys
        that are associated with the current variable. The keys are identified by
        checking if they contain the variable name as a prefix, followed by an
        underscore. The filtered dictionaries are then stored in `self.qc_flags`
        and `self.qc_details` under the corresponding variable name.

        Attributes:
            self.qc_vars (list): A list of variable names to filter QC data by.
            self.qc_flags (dict): A dictionary containing QC flags, which will be
                filtered and organized by variable.
            self.qc_details (dict): A dictionary containing QC details, which will
                be filtered and organized by variable.

        """
        for variable in self.qc_vars:
            self.qc_by_var[variable]["qc_flags"].update(
                {
                    key: self.qc_flags.get(key)
                    for key in list(self.qc_flags.keys())
                    if f"{variable}_" in key
                }
            )
            self.qc_by_var[variable]["qc_details"].update(
                {
                    key: self.qc_details.get(key)
                    for key in list(self.qc_details.keys())
                    if f"{variable}_" in key
                }
            )

    def get_byte_array(self, variable):
        """
        Generate a byte array and associated attributes for a given variable's quality control flags.

        This function checks if quality control flags for the specified variable are available.
        If not, it retrieves them. It then calculates a byte value representing the quality control
        status by iterating over the flags and their values. Additionally, it constructs a dictionary
        of attributes that describe the quality control flags.

        Parameters:
        - variable (str): The name of the variable for which to generate the byte array and attributes.

        Returns:
        - tuple: A tuple containing:
            - np.byte: The calculated byte value representing the quality control status.
            - dict: A dictionary of attributes with the following keys:
                - long_name (str): A descriptive name for the quality control of the variable.
                - standard_name (str): A standard name indicating the type of flag.
                - flag_masks (str): A comma-separated string of binary masks for each flag.
                - flag_meanings (str): A comma-separated string of the meanings of each flag.
        """
        if not self.qc_by_var.get(variable, {}).get("qc_flags"):
            self.get_qc_by_var()
        qc_val = np.byte(0)
        keys = []
        for i, (key, value) in enumerate(
            self.qc_by_var.get(variable).get("qc_flags").items()
        ):
            qc_val = qc_val + (2**i) * np.byte(not value)
            keys.append(key.split("_", 1)[1])
        if qc_val == 0:
            qc_status = "GOOD"
        elif qc_val == 2 ** (i + 1) - 1:
            qc_status = "BAD"
        else:
            qc_status = "UGLY"
        attrs = dict(
            long_name=f"qc for {variable}",
            standard_name="status_flag",
            flag_masks=", ".join([f"{2**x}b" for x in range(i + 1)]),
            flag_meanings=", ".join(keys),
            description="if non-zero, this sonde should be used with care.",
            qc_status=qc_status,
        )
        return np.byte(qc_val), attrs

    def get_details(self, variable):
        """
        Retrieve quality control details and attributes for a specified variable.

        This method checks if the quality control (QC) details for the given variable are available. If not, it invokes the `get_qc_by_var` method to populate them. It then constructs a dictionary of attributes for each QC key associated with the variable, providing a descriptive long name and units.

        Parameters:
            variable (str): The name of the variable for which QC details are to be retrieved.

        Returns:
            tuple: A tuple containing:
                - dict: The QC details for the specified variable.
                - dict: A dictionary of attributes for each QC key, including:
                    - long_name (str): A descriptive name for the QC key.
                    - units (str): The units for the QC key, defaulted to "1".
        """
        if self.qc_by_var.get(variable, {}).get("qc_details") is not None:
            self.get_qc_by_var()
        attrs = {}
        for key in list(self.qc_by_var.get(variable).get("qc_details").keys()):
            name = key.split("_", 1)[1]
            attrs.update(
                {
                    key: dict(
                        long_name=f"value for qc  {variable} " + name.replace("_", " "),
                        units="1",
                    )
                }
            )
        return self.qc_by_var.get(variable).get("qc_details"), attrs

    def add_variable_flags_to_ds(self, ds, variable, details=True):
        name = f"{variable}_qc"
        value, attrs = self.get_byte_array(variable)
        ds = ds.assign({name: value})
        ds[name].attrs.update(attrs)
        ds = hx.add_ancillary_var(ds, variable, name)
        # get detail
        if details:
            qc_dict, attrs = self.get_details(variable)
            for key in list(qc_dict.keys()):
                ds = ds.assign({key: qc_dict.get(key)})
                ds[key].attrs.update(attrs.get(key))
                ds = hx.add_ancillary_var(ds, variable, key)

        return ds

    def add_alt_near_gpsalt_to_ds(self, ds):
        if self.qc_flags.get("alt_near_gpsalt") is not None:
            ds = ds.assign(
                {"alt_near_gpsalt": np.byte(not self.qc_flags.get("alt_near_gpsalt"))}
            )
            ds["alt_near_gpsalt"].attrs.update(
                dict(
                    long_name="maximal difference between alt and gpsalt",
                    flag_values="0 1 ",
                    flag_meaning="GOOD BAD",
                )
            )

            ds = ds.assign(
                {
                    "alt_near_gpsalt_max_diff": self.qc_details.get(
                        "alt_near_gpsalt_max_diff"
                    )
                }
            )
            ds["alt_near_gpsalt_max_diff"].attrs.update(
                dict(
                    long_name="maximal difference between alt and gpsalt",
                    units="m",
                )
            )

            ds = hx.add_ancillary_var(
                ds, "alt", "alt_near_gpsalt alt_near_gpsalt_max_diff"
            )
        return ds

    def add_low_physic_flags_to_ds(self, ds):
        if self.qc_flags.get("low_physics") is not None:
            ds = ds.assign(
                {"low_physics": np.byte(not self.qc_flags.get("low_physics"))}
            )
            ds["low_physics"].attrs.update(
                dict(
                    long_name="low physics",
                    flag_values="0 1 ",
                    flag_meaning="GOOD BAD",
                )
            )

            ds = ds.assign(
                {"low_physics_rh_min": self.qc_details.get("low_physics_rh_min")}
            )
            ds["low_physics_rh_min"].attrs.update(
                dict(
                    long_name="minimal relative humidity below 100m",
                    units="%",
                )
            )

            ds = ds.assign(
                {"low_physics_ta_min": self.qc_details.get("low_physics_ta_min")}
            )
            ds["low_physics_ta_min"].attrs.update(
                dict(
                    long_name="minimal temperature below 100m",
                    units="degreeC",
                )
            )

            ds = hx.add_ancillary_var(
                ds,
                "sonde_id",
                "low_physics",
            )
            ds = hx.add_ancillary_var(
                ds,
                "rh",
                "low_physics_rh_min",
            )
            ds = hx.add_ancillary_var(
                ds,
                "ta",
                "low_physics_ta_min",
            )
        return ds

    def replace_alt_var(self, ds, alt_var):
        """
        Replace the altitude variable in a dataset with its counterpart.

        This method swaps the values of the specified altitude variable with its counterpart
        in the dataset. If `alt_var` is "alt", it will be replaced with "gpsalt", and vice versa.
        If `alt_var` is neither "alt" nor "gpsalt", a ValueError is raised.

        Parameters:
        - ds: The dataset containing the altitude variables.
        - alt_var: A string specifying the altitude variable to be replaced.
                It must be either "alt" or "gpsalt".

        Returns:
        - A new dataset with the specified altitude variable replaced by its counterpart.

        Raises:
        - ValueError: If `alt_var` is not "alt" or "gpsalt".
        """
        if alt_var == "alt":
            replace_var = "gpsalt"
        elif alt_var == "gpsalt":
            replace_var = "alt"
        else:
            raise ValueError(f"{alt_var} is no known altitude variable.")

        ds_out = ds.assign({alt_var: ds[replace_var]})
        self.qc_flags.update({f"{alt_var}_values": False})

        return ds_out

    def add_replace_alt_var_to_ds(self, ds):
        """
        Adds  an ancillary variable in the dataset for the altitude dimension.

        This function assigns a new variable to the dataset `ds` with a name based on
        the `alt_dim` attribute of the class. The new variable indicates whether values
        for the specified dimension are present in the raw data, using quality control
        flags. It updates the attributes of the new variable to include a long name,
        flag values, and flag meanings.

        Parameters:
        - ds: The dataset to which the ancillary variable will be added or replaced.

        Returns:
        - The updated dataset with the ancillary variable added or replaced.
        """
        ds = ds.assign(
            {
                f"{self.alt_dim}_values": np.byte(
                    (not self.qc_flags.get(f"{self.alt_dim}_values", False))
                )
            }
        )
        ds[f"{self.alt_dim}_values"].attrs.update(
            dict(
                long_name=f"Values for {self.alt_dim} are present in raw data",
                flag_values="0 1 ",
                flag_meaning="GOOD BAD",
            )
        )

        ds = hx.add_ancillary_var(ds, self.alt_dim, f"{self.alt_dim}_values")
        return ds

    def add_non_var_qc_to_ds(self, ds):
        """
        Adds non-variable quality control (QC) data to the given dataset.

        This method performs the following operations on the input dataset `ds`:
        1. Adds altitude near GPS altitude to the dataset using the `add_alt_near_gpsalt_to_ds` method.
        2. Replaces altitude variable in the dataset using the `add_replace_alt_var_to_ds` method.

        Parameters:
        - ds: The input dataset to which non-variable QC data will be added.

        Returns:
        - ds_out: The output dataset with added non-variable QC data.
        """
        ds_out = self.add_alt_near_gpsalt_to_ds(ds)
        ds_out = self.add_replace_alt_var_to_ds(ds_out)
        ds_out = self.add_low_physic_flags_to_ds(ds_out)

        return ds_out

    def add_sonde_flag_to_ds(self, ds, qc_name):
        if all(self.qc_flags.values()):
            qc_val = 0
        elif any(self.qc_flags.values()) and (
            self.qc_flags.get(f"{self.alt_dim}_values", True)
        ):
            qc_val = 2
        else:
            qc_val = 1

        ds = ds.assign({qc_name: qc_val})
        ds[qc_name].attrs.update(
            dict(
                standard_name="aggregate_quality_flag",
                long_name="aggregated quality flag for sonde",
                flag_values="0 1 2",
                flag_meaning="GOOD BAD UGLY",
                description="if not 0 some quality control has not been passed. Handle with care.",
            )
        )
        ds = hx.add_ancillary_var(ds, "sonde_id", qc_name)

        return ds
