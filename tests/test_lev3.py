import pytest
import xarray as xr
import numpy as np
from pydropsonde.processor import Sonde

s_id = "test_this_id"
flight_id = "test_this_flight"
platform_id = "test_this_platform"
launch_time = "2020-02-02 20:22:02"


@pytest.mark.parametrize(
    "test_input,expected",
    [
        # normal binning
        (
            dict(
                time=np.array(
                    [
                        np.datetime64("2024-01-01", "ns"),
                        np.datetime64("2024-01-02", "ns"),
                        np.datetime64("2024-01-04", "ns"),
                        np.datetime64("2024-01-06", "ns"),
                    ]
                ),
                rh=np.array([0.8, 0.7, 0.8, 0.7]),
                alt=np.array([30.0, 20.0, 15.0, 10.0]),
                p=np.array([1.0, 10.0, 100.0, 1000.0]),
            ),
            dict(
                time=np.array(
                    [
                        np.nan,
                        np.datetime64("2024-01-05", "ns"),
                        np.datetime64("2024-01-02", "ns"),
                        np.datetime64("2024-01-01", "ns"),
                    ]
                ),
                rh=np.array([np.nan, 0.75, 0.7, 0.8]),
                alt=np.array([0.0, 10.0, 20.0, 30.0]),
                p=np.array([np.nan, 550.0, 10.0, 1.0]),
                Nrh=[0, 2, 1, 1],
                mrh=[0, 2, 2, 2],
            ),
        ),
        # interpolation
        (
            dict(
                time=np.array(
                    [
                        np.datetime64("2024-01-01", "ns"),
                        np.datetime64("2024-01-02", "ns"),
                        np.datetime64("NaT"),
                        np.datetime64("2024-01-06", "ns"),
                    ]
                ),
                rh=np.array([0.8, 0.7, np.nan, 0.8]),
                alt=np.array([30.0, 20.0, 10.0, 1.0]),
                p=np.array([1.0, 1e1, np.nan, 1e3]),
            ),
            dict(
                time=np.array(
                    [
                        np.datetime64("2024-01-06", "ns"),
                        np.datetime64("2024-01-04", "ns"),
                        np.datetime64("2024-01-02", "ns"),
                        np.datetime64("2024-01-01", "ns"),
                    ]
                ),
                rh=np.array([0.8, 0.75, 0.7, 0.8]),
                alt=np.array([0.0, 10.0, 20.0, 30.0]),
                p=np.array([1e3, 1e2, 1e1, 1]),
                Nrh=[1, 0, 1, 1],
                mrh=[2, 1, 2, 2],
            ),
        ),
        # gap to big to fill
        (
            dict(
                time=np.array(
                    [
                        np.datetime64("2024-01-01", "ns"),
                        np.datetime64("NaT"),
                        np.datetime64("NaT"),
                        np.datetime64("2024-01-06", "ns"),
                    ]
                ),
                rh=np.array([0.8, np.nan, np.nan, 0.8]),
                alt=np.array([30.0, 20.0, 10.0, 1.0]),
                p=np.array([1.0, 10.0, 100.0, 1000.0]),
            ),
            dict(
                time=np.array(
                    [
                        np.datetime64("2024-01-06", "ns"),
                        np.datetime64("NaT"),
                        np.datetime64("NaT"),
                        np.datetime64("2024-01-01", "ns"),
                    ]
                ),
                rh=np.array([0.8, np.nan, np.nan, 0.8]),
                alt=np.array([0.0, 10.0, 20.0, 30.0]),
                p=np.array([1000, 100.0, 10.0, 1.0]),
                Nrh=[1, 0, 0, 1],
                mrh=[2, 0, 0, 2],
            ),
        ),
    ],
)
class TestGroup:
    @pytest.fixture(autouse=True)
    def sonde(self):
        sonde = Sonde(serial_id=s_id, launch_time=launch_time)
        sonde.add_flight_id(flight_id)
        sonde.add_platform_id(platform_id)
        self.sonde = sonde

    @pytest.fixture
    def sonde_interp(self, test_input, expected):
        data_dict = {
            "coords": {"time": {"dims": ("time"), "data": test_input["time"]}},
            "data_vars": {
                "rh": {"dims": ("time"), "data": test_input["rh"]},
                "p": {"dims": ("time"), "data": test_input["p"]},
                "alt": {"dims": ("time"), "data": test_input["alt"]},
            },
        }

        ds = xr.Dataset.from_dict(data_dict)

        object.__setattr__(self.sonde, "_prep_l3_ds", ds)

        new_sonde = self.sonde.interpolate_alt(
            alt_var="alt",
            interp_start=-5,
            interp_stop=36,
            interp_step=10,
            max_gap_fill=int(20),
            p_log=True,
            method="bin",
        )

        res_dict = {
            "coords": {"alt": {"dims": ("alt"), "data": expected["alt"]}},
            "data_vars": {
                "rh": {"dims": ("alt"), "data": expected["rh"]},
                "p": {"dims": ("alt"), "data": expected["p"]},
                "interp_time": {"dims": ("alt"), "data": expected["time"]},
            },
        }
        result_ds = xr.Dataset.from_dict(res_dict)

        print(result_ds)
        print(new_sonde._prep_l3_ds)

        assert not np.any(np.abs(result_ds.p - new_sonde._prep_l3_ds.p) > 1e-6)
        assert result_ds.drop_vars("p").equals(new_sonde._prep_l3_ds.drop_vars("p"))
        self.interp_sonde = new_sonde

    def test_N_m(self, sonde_interp, test_input, expected):
        new_sonde = self.interp_sonde.get_N_m_values(alt_var="alt")
        print(new_sonde._prep_l3_ds)
        assert np.all(new_sonde._prep_l3_ds["Nrh"].values == expected["Nrh"])
        assert np.all(new_sonde._prep_l3_ds["mrh"].values == expected["mrh"])