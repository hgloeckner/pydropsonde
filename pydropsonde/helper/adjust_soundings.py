def radio2drop_format(ds):
    sonde_id = ds.sounding.values[0]
    launch_time = ds.launch_time.values[0]
    ds = (
        ds.squeeze("sounding").reset_coords("sounding").rename({"sounding": "sonde_id"})
    )
    return sonde_id, launch_time, ds
