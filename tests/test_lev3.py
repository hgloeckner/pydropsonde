import pytest
import xarray as xr
import numpy as np
from pydropsonde.processor import Sonde

s_id = "test_this_id"
flight_id = "test_this_flight"
platform_id = "test_this_platform"
launch_time = "2020-02-02 20:22:02"


@pytest.mark.parametrize(
    "test_input,expected,expected_linear",
    [
        # normal binning
        (
            dict(
                time=np.array(
                    [
                        np.datetime64("2024-01-01", "us"),
                        np.datetime64("2024-01-02", "us"),
                        np.datetime64("2024-01-04", "us"),
                        np.datetime64("2024-01-06", "us"),
                    ]
                ),
                q=np.array([0.8, 0.7, 0.8, 0.7]),
                alt=np.array([30.0, 20.0, 15.0, 10.0]),
                p=np.array([1.0, 10.0, 100.0, 1000.0]),
            ),
            dict(
                time=np.array(
                    [
                        np.nan,
                        np.datetime64("2024-01-06", "us"),
                        np.datetime64("2024-01-03", "us"),
                        np.datetime64("2024-01-01", "us"),
                    ]
                ),
                q=np.array([np.nan, 0.7, 0.75, 0.8]),
                alt=np.array([0.0, 10.0, 20.0, 30.0]),
                p=np.array(
                    [np.nan, 1000.0, np.exp((np.log(10) + np.log(100)) / 2), 1.0]
                ),
                Nq=[0, 1, 2, 1],
                mq=[0, 2, 2, 2],
            ),
            dict(
                time=np.array(
                    [
                        np.nan,
                        np.datetime64("2024-01-06", "us"),
                        np.datetime64("2024-01-02", "us"),
                        np.datetime64("2024-01-01", "us"),
                    ],
                ),
                q=np.array([np.nan, 0.7, 0.7, 0.8]),
                alt=np.array([0.0, 10.0, 20.0, 30.0]),
                p=np.array([np.nan, 1000.0, 10.0, 1.0]),
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
                q=np.array([0.8, 0.7, np.nan, 0.8]),
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
                q=np.array([0.8, 0.75, 0.7, 0.8]),
                alt=np.array([0.0, 10.0, 20.0, 30.0]),
                p=np.array([1e3, 1e2, 1e1, 1]),
                Nq=[1, 0, 1, 1],
                mq=[2, 1, 2, 2],
            ),
            dict(
                time=np.array(
                    [
                        np.datetime64("2024-01-06 05:00", "ns"),
                        np.datetime64("NaT", "ns"),
                        np.datetime64("2024-01-02", "ns"),
                        np.datetime64("2024-01-01", "ns"),
                    ]
                ),
                q=np.array([0.8053, np.nan, 0.7, 0.8]),
                alt=np.array([0.0, 10.0, 20.0, 30.0]),
                p=np.array([1274, np.nan, 1e1, 1]),
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
                q=np.array([0.8, np.nan, np.nan, 0.8]),
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
                q=np.array([0.8, np.nan, np.nan, 0.8]),
                alt=np.array([0.0, 10.0, 20.0, 30.0]),
                p=np.array([1000, 100.0, 10.0, 1.0]),
                Nq=[1, 0, 0, 1],
                mq=[2, 0, 0, 2],
            ),
            dict(
                time=np.array(
                    [
                        np.datetime64("2024-01-06 04:00", "ns"),
                        np.datetime64("NaT"),
                        np.datetime64("NaT"),
                        np.datetime64("2024-01-01", "ns"),
                    ]
                ),
                q=np.array([0.8, np.nan, np.nan, 0.8]),
                alt=np.array([0.0, 10.0, 20.0, 30.0]),
                p=np.array([1291.5, 100.0, 10.0, 1.0]),
            ),
        ),
    ],
)
class TestGroup:
    @pytest.fixture(autouse=True)
    def sonde(self):
        sonde = Sonde(_serial_id=s_id, _launch_time=launch_time)
        sonde.add_flight_id(flight_id)
        sonde.add_platform_id(platform_id)
        sonde.set_alt_dim("alt")
        self.sonde = sonde

    @pytest.fixture
    def sonde_interp(self, test_input, expected, expected_linear):
        data_dict = {
            "coords": {"time": {"dims": ("time"), "data": test_input["time"]}},
            "data_vars": {
                "q": {"dims": ("time"), "data": test_input["q"]},
                "p": {"dims": ("time"), "data": test_input["p"]},
                "alt": {"dims": ("time"), "data": test_input["alt"]},
            },
        }

        ds = xr.Dataset.from_dict(data_dict)
        self.sonde.interim_l3_ds = ds
        self.sonde.swap_alt_dimension()

        new_sonde = self.sonde.interpolate_variables_to_common_grid(
            interp_start=-5,
            interp_stop=36,
            interp_step=10,
            max_gap_fill=int(20),
            p_log=True,
            interpolate=True,
            method="bin",
        )

        res_dict = {
            "coords": {"alt": {"dims": ("alt"), "data": expected["alt"]}},
            "data_vars": {
                "q": {"dims": ("alt"), "data": expected["q"]},
                "p": {"dims": ("alt"), "data": expected["p"]},
                "bin_average_time": {"dims": ("alt"), "data": expected["time"]},
            },
        }
        result_ds = xr.Dataset.from_dict(res_dict)

        print(result_ds)
        print(new_sonde.interim_l3_ds)

        assert not np.any(np.abs(result_ds.p - new_sonde.interim_l3_ds.p) > 1e-6)
        assert result_ds.drop_vars("p").equals(
            new_sonde.interim_l3_ds.drop_vars(
                ["p", "q_N_qc", "p_N_qc", "q_m_qc", "p_m_qc"]
            )
        )
        self.interp_sonde = new_sonde

    def test_N_m(self, sonde_interp, test_input, expected, expected_linear):
        new_sonde = self.interp_sonde
        print(new_sonde.interim_l3_ds)
        print(expected["Nq"])
        assert np.all(new_sonde.interim_l3_ds["q_N_qc"].values == expected["Nq"])
        assert np.all(new_sonde.interim_l3_ds["q_m_qc"].values == expected["mq"])

    def test_sonde_linear(self, test_input, expected, expected_linear):
        data_dict = {
            "coords": {"time": {"dims": ("time"), "data": test_input["time"]}},
            "data_vars": {
                "q": {"dims": ("time"), "data": test_input["q"]},
                "p": {"dims": ("time"), "data": test_input["p"]},
                "alt": {"dims": ("time"), "data": test_input["alt"]},
            },
        }

        ds = xr.Dataset.from_dict(data_dict)
        self.sonde.interim_l3_ds = ds
        self.sonde.swap_alt_dimension()
        self.sonde.interim_l3_ds = self.sonde.interim_l3_ds.reset_coords()
        new_sonde = self.sonde.interpolate_variables_to_common_grid(
            interp_start=-5,
            interp_stop=36,
            interp_step=10,
            p_log=True,
            method="linear_interpolate",
        )

        res_dict = {
            "coords": {"alt": {"dims": ("alt"), "data": expected_linear["alt"]}},
            "data_vars": {
                "q": {"dims": ("alt"), "data": expected_linear["q"]},
                "p": {"dims": ("alt"), "data": expected_linear["p"]},
                "interpolated_time": {
                    "dims": ("alt"),
                    "data": expected_linear["time"],
                },
            },
        }
        result_ds = xr.Dataset.from_dict(res_dict)
        # print(ds)

        print(new_sonde.interim_l3_ds.interpolated_time.astype("datetime64[ns]").values)
        print(result_ds.interpolated_time.astype("datetime64[ns]").values)

        assert not np.any(np.abs(result_ds.p - new_sonde.interim_l3_ds.p) > 1)
        assert not np.any(
            np.abs(
                result_ds.interpolated_time.astype("datetime64[ns]")
                - new_sonde.interim_l3_ds.interpolated_time.astype("datetime64[ns]")
            )
            > np.timedelta64(1, "h")
        )
        assert not np.any(np.abs(result_ds.q - new_sonde.interim_l3_ds.q) > 1e-3)
