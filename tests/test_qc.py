import pytest
import xarray as xr
import numpy as np
from pydropsonde.helper.quality import QualityControl


data_dict = {
    "coords": {
        "time": {
            "dims": ("time"),
            "data": np.array(
                [
                    np.datetime64("2024-01-01T12:00:00.0", "ns"),
                    np.datetime64("2024-01-01T12:00:00.5", "ns"),
                    np.datetime64("2024-01-01T12:00:01.0", "ns"),
                    np.datetime64("2024-01-01T12:00:01.5", "ns"),
                ]
            ),
        }
    },
    "data_vars": {
        "q": {"dims": ("time"), "data": np.array([np.nan, np.nan, np.nan, 0.7])},
        "p": {"dims": ("time"), "data": np.array([1.0, 10.0, 100.0, 1000.0])},
        "alt": {"dims": ("time"), "data": np.array([30.0, 20.0, 15.0, 10.0])},
        "rh": {"dims": ("time"), "data": np.array([30.0, np.nan, np.nan, 10.0])},
    },
}


ds = xr.Dataset.from_dict(data_dict)
ds = ds.assign_attrs(SondeId="test")
ds = ds.assign({"sonde_id": "test"})


@pytest.fixture
def qc():
    qc = QualityControl()
    qc.is_floater = False
    return qc


@pytest.fixture
def qc_vars(qc):
    qc.set_qc_variables({"q": "m s-1", "p": "Pa", "rh": "1"})
    qc.set_qc_ds(ds)
    qc.alt_dim = "alt"

    assert qc.qc_vars == {"q": "m s-1", "p": "Pa", "rh": "1"}
    return qc


def test_profile_sparsity(qc_vars):
    qc_vars.profile_sparsity(variable_dict={"q": 2, "p": 4, "rh": 4})
    assert qc_vars.qc_flags["p_profile_sparsity"]
    assert qc_vars.qc_flags["q_profile_sparsity"]
    assert not qc_vars.qc_flags["rh_profile_sparsity"]
    assert qc_vars.qc_details["p_profile_sparsity_fraction"] >= 0
    assert qc_vars.qc_details["rh_profile_sparsity_fraction"] >= 0


def test_near_surface(qc_vars):
    qc_vars.near_surface_coverage(alt_bounds=[0, 18], count_threshold=2)

    assert qc_vars.qc_flags["p_near_surface"]
    assert not qc_vars.qc_flags["q_near_surface"]
    assert qc_vars.qc_details["p_near_surface_count"] == 2
    assert qc_vars.qc_details["q_near_surface_count"] == 1


@pytest.mark.parametrize(
    "qc_flag,output",
    [
        ("p_profile_sparsity", True),
        ("rh_profile_sparsity", False),
        ("p_profile_sparsity,p_near_surface", True),
        ("p_profile_sparsity,q_near_surface", False),
        (None, True),
        ("all", False),
        ("all_except_rh_profile_sparsity", False),
        ("p_profile_sparsity,ta_profile_sparsity", ValueError),
    ],
)
def test_check_qc(qc_vars, qc_flag, output):
    qc_vars.profile_sparsity(variable_dict={"q": 2, "p": 2, "rh": 4})
    qc_vars.near_surface_coverage(alt_bounds=[0, 18], count_threshold=2)
    if (type(output) is type) and issubclass(output, Exception):
        with pytest.raises(output):
            res = qc_vars.check_qc(used_flags=qc_flag)
    else:
        res = qc_vars.check_qc(used_flags=qc_flag)
        assert res == output


@pytest.mark.parametrize(
    "varname,output",
    [
        ("p", "GOOD"),
        ("rh", "BAD"),
        ("q", "UGLY"),
    ],
)
def test_add_variable_flag_to_ds(qc_vars, varname, output):
    qc_vars.profile_sparsity(variable_dict={"q": 2, "p": 2, "rh": 4})
    qc_vars.near_surface_coverage(alt_bounds=[0, 18], count_threshold=2)
    ds_out = qc_vars.add_variable_flags_to_ds(ds, varname)
    assert ds_out[f"{varname}_qc"].qc_status == output
    assert (
        ds_out[varname].attrs["ancillary_variables"]
        == f"{varname}_qc {varname}_profile_sparsity_fraction {varname}_near_surface_count"
    )


@pytest.mark.parametrize(
    "variables,output",
    [
        ({"p": "Pa"}, 0),  # GOOD
        ({"q": "1", "p": "Pa"}, 2),  # UGLY
        ({"rh": "1"}, 1),  # BAD
        ({"q": "1"}, 2),  # UGLY
        ({"rh": "1", "q": "1"}, 2),  # UGLY
        ({"rh": "1", "p": "1"}, 2),  # UGLY
    ],
)
def test_sonde_qc(qc_vars, variables, output):
    varname = "sonde_qc"
    qc_vars.qc_vars = variables
    qc_vars.profile_sparsity(variable_dict={"q": 2, "p": 2, "rh": 4})
    qc_vars.near_surface_coverage(alt_bounds=[0, 18], count_threshold=2)

    ds_out = qc_vars.add_sonde_flag_to_ds(ds, varname)
    assert ds_out[varname] == output
