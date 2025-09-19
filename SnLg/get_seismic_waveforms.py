from pathlib import Path
import logging

from obspy import Stream, read, UTCDateTime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Private helpers
# -----------------------------------------------------------------------------

def _generate_date_strings(starttime, endtime, formatter):
    """Return *all* date strings between *starttime* and *endtime* (inclusive)."""
    dates = set()
    cur = starttime
    one_day = 86400  # seconds
    while cur <= endtime:
        dates.add(formatter(cur))
        cur += one_day
    return dates

def _load_merge_trim(files, starttime, endtime, merge_method=-1, merge_fill_value=None, buffer=1.0):
    """Read, merge, and *fully* trim traces using a small ``buffer`` margin."""
    st = Stream()
    for fp in files:
        st += read(str(fp))

    st.merge(method=merge_method, fill_value=merge_fill_value)
    st.trim(starttime, endtime, pad=True, fill_value=merge_fill_value)

    return Stream(
        tr for tr in st if tr.stats.starttime <= starttime + buffer and tr.stats.endtime >= endtime - buffer
    )

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def datefmt_mseed(t):
    """Return *YYYYMMDD* string used in ESDC MiniSEED archives."""
    return t.strftime("%Y%m%d")

def datefmt_sac(t):
    """Return *YYYY.JJJ.00.00.00.0000* string used in ESDC SAC archives."""
    return f"{t.year}.{t.julday:03d}.00.00.00.0000"

def load_waveform_from_directory(starttime, endtime, station_code, data_files, date_formatter,
                                  merge_method=-1, merge_fill_value=None, buffer=1.0, **kwargs):
    """Generic loader that powers the MiniSEED & SAC helpers below."""
    paths = [Path(p) for p in data_files]
    dates_needed = _generate_date_strings(starttime, endtime, date_formatter)

    selected = [
        p for p in paths if station_code in p.name and any(ds in p.name for ds in dates_needed)
    ]

    if not selected:
        logger.warning("No files found for %s between %s and %s", station_code, starttime, endtime)
        return None

    try:
        stream = _load_merge_trim(
            selected,
            starttime,
            endtime,
            merge_method=merge_method,
            merge_fill_value=merge_fill_value,
            buffer=buffer,
        )
    except Exception as exc:
        logger.error("Failed to read/merge/trim for %s: %s", station_code, exc)
        return None

    if len(stream) < 3:
        logger.warning("Expected at least 3 components, got %d for %s", len(stream), station_code)
        return None

    return stream

def get_esdc_mseed_sac_waveform(starttime, endtime, station, mseed_file_list, sac_file_list, **kwargs):
    """
    Try loading MiniSEED waveform data first; if that fails, fall back to SAC.

    Parameters:
        starttime, endtime : UTCDateTime
            Time window for waveform extraction.
        station : object
            Station object with `.code` attribute.
        mseed_file_list : list
            List of MiniSEED file paths.
        sac_file_list : list
            List of SAC file paths.
        **kwargs : dict
            Passed through to `load_waveform_from_directory`.

    Returns:
        Stream or None: A valid ObsPy Stream or None if both fail.
    """
    stream = load_waveform_from_directory(
        starttime,
        endtime,
        station.code,
        mseed_file_list,
        datefmt_mseed,
        **kwargs,
    )

    if not stream:
        logger.info("MiniSEED data not found for %s, falling back to SAC.", station.code)
        stream = load_waveform_from_directory(
            starttime,
            endtime,
            station.code,
            sac_file_list,
            datefmt_sac,
            **kwargs,
        )

    return stream

def get_esdc_mseed_waveform(starttime, endtime, station, data_file_list, **kwargs):
    """Return ESDC MiniSEED data for *station* in [*starttime*, *endtime*]."""
    return load_waveform_from_directory(
        starttime,
        endtime,
        station.code,
        data_file_list,
        datefmt_mseed,
        **kwargs,
    )

def get_esdc_sac_waveform(starttime, endtime, station, data_file_list, **kwargs):
    """Return ESDC SAC data for *station* in [*starttime*, *endtime*]."""
    return load_waveform_from_directory(
        starttime,
        endtime,
        station.code,
        data_file_list,
        datefmt_sac,
        **kwargs,
    )

def get_waveform_from_client(starttime, endtime, station, client, network_code,
                              location_requested="*", channel_requested="BH*", attach_response=True):
    """Thin wrapper around :py:meth:`obspy.clients.base.Client.get_waveforms`."""
    try:
        return client.get_waveforms(
            network=network_code,
            station=station.code,
            location=location_requested,
            channel=channel_requested,
            starttime=starttime,
            endtime=endtime,
            attach_response=attach_response,
        )
    except Exception as exc:
        logger.warning(
            "Error fetching %s.%s.%s.%s between %s–%s from %s: %s",
            network_code,
            station.code,
            location_requested,
            channel_requested,
            starttime,
            endtime,
            getattr(client, "base_url", "<unknown client>"),
            exc,
        )
        return None
