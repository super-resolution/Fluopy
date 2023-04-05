def frac_int_time(pandas_series, fraction):
    """
    Returns the relative time at which the specified fraction of the total collected photons is reached.

    Parameters
    ----------
    pandas_series : pd.Series
        Contains the time step (of a frame) in seconds as index and the number of events (photons) as values.
    fraction : float
        Number between 0 and 1.

    Returns
    -------
    arrival_time_rel : float
        The relative time at which the fraction of the total collected photons is reached.
    """
    end_time = pandas_series.index[-1]

    cumsum = pandas_series.cumsum()
    cumsum_norm = cumsum.multiply(1 / cumsum.max())
    arrival_time = cumsum_norm.gt(fraction).idxmax()
    arrival_time_rel = arrival_time / end_time

    return arrival_time_rel
