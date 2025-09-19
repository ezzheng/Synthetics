import multiprocessing as mp
from multiprocessing import Queue
from queue import Empty
from snlg_analyses import SnLg_Analyses
from get_seismic_waveforms import get_waveform_from_client
from obspy.clients.fdsn import Client
from obspy import UTCDateTime
from obspy.geodetics import kilometers2degrees
from copy import deepcopy
import pandas as pd
import numpy as np
import time
import os
import logging

END_SIGNAL = ("END", None)

def init_main_logger(log_path="./test/snlg_parallel_log.txt"):
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode='a')
        formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def select_stations_by_azimuth_distance(
    inventory, evla, evlo, max_station=20, n_azimuth_bins=8, n_distance_bins=8
):
    """
    Select representative stations using 2D binning by azimuth and distance.
    """
    from obspy.geodetics.base import gps2dist_azimuth
    from obspy.core.inventory import Inventory, Network

    station_info = []
    for network in inventory:
        for station in network:
            stla = station.latitude
            stlo = station.longitude
            dist_m, az, _ = gps2dist_azimuth(evla, evlo, stla, stlo)
            dist_km = dist_m / 1000
            dist_deg = kilometers2degrees(dist_km)
            station_info.append({
                'network': network.code,
                'station': station.code,
                'latitude': stla,
                'longitude': stlo,
                'distance_deg': dist_deg,
                'distance_km': dist_km,
                'azimuth': az,
                'station_obj': station
            })

    df = pd.DataFrame(station_info)

    # Bin azimuth [0, 360)
    df['az_bin'] = pd.cut(df['azimuth'], bins=np.linspace(0, 360, n_azimuth_bins + 1), labels=False, include_lowest=True)

    # Bin distance (km)
    dist_edges = np.linspace(df['distance_km'].min(), df['distance_km'].max(), n_distance_bins + 1)
    df['dist_bin'] = pd.cut(df['distance_km'], bins=dist_edges, labels=False, include_lowest=True)

    # Select top 1 station per (az_bin, dist_bin), or more depending on max_station
    selected_stations = []
    for _, group in df.groupby(['az_bin', 'dist_bin']):
        group_sorted = group.sort_values(by='distance_km')
        selected_stations.extend(group_sorted.head(1).to_dict('records')) # head(N) -> the first N stations

    # Limit to max_station
    selected_stations = selected_stations[:max_station]

    # Reconstruct a trimmed Inventory
    new_networks = {}
    for s in selected_stations:
        net_code = s['network']
        if net_code not in new_networks:
            new_networks[net_code] = Network(code=net_code, stations=[])
        new_networks[net_code].stations.append(s['station_obj'])

    return Inventory(networks=list(new_networks.values()), source=inventory.source)

def run_worker_instance(worker_id, task_queue, analyzer_config,
                        catalog, client, maximum_station):
    cfg = deepcopy(analyzer_config)

    logger0 = init_main_logger(log_path=os.path.join(cfg["base_directory"],"snlg_parallel_log.txt"))
    logger0.info(f"[Worker {worker_id}] start.")

    # Update configurations
    base_directory = os.path.join(cfg["base_directory"], f"worker_{worker_id}")
    cfg["base_directory"] = base_directory
    cfg["raw_directory"] = os.path.join(base_directory,"raw")
    cfg["corrected_directory"] = os.path.join(base_directory,"corrected")
    cfg["selected_directory"] = os.path.join(base_directory,"selected")
    cfg["log_fname"] = cfg["log_fname"] + f"_w{worker_id}"

    analyzer = SnLg_Analyses(**cfg)

    last_task_time = time.time()
    timeout_seconds = 60

    events = list(catalog.itertuples())
    min_dis_deg = kilometers2degrees(cfg["min_epi_distance"])
    max_dis_deg = kilometers2degrees(cfg["max_epi_distance"])

    while True:
        try:
            kind, payload = task_queue.get(timeout=5)
        except Empty:
            if time.time() - last_task_time > timeout_seconds:
                logger0.info(f"[Worker {worker_id}] idle timeout, exiting.")
                break
            continue

        last_task_time = time.time()

        if kind == "byevent":
            logger0.info(f"[Worker {worker_id}] Working on No. {payload + 1}/{len(events)} earthquake")
            evt = events[payload]
            evla = evt.event_lat
            evlo = evt.event_lon
            starttime = UTCDateTime(evt.event_time)
            endtime   = starttime + 3600

            for attempt in range(2):
                try:
                    custom_inventory = client.get_stations(
                    level='channel',    # level='channel' is nessary for rotating BH[12] -> BH[NE]
                    starttime=starttime, endtime=endtime,
                    channel='BH*', network="*", location="*", station="*",
                    latitude=evla, longitude=evlo,
                    minradius=min_dis_deg, maxradius=max_dis_deg
                )
                    break
                except Exception as e:
                    logger0.info(f"[Worker {worker_id}] Attempt {attempt+1} failed: {e}")
                    time.sleep(1)
            else:
                continue  # skip this event

            # Check if we need to reduce the number of stations
            if sum(len(net.stations) for net in custom_inventory) > maximum_station:
                custom_inventory = select_stations_by_azimuth_distance(
                    inventory=custom_inventory,
                    evla=evla,
                    evlo=evlo,
                    max_station=maximum_station,
                    n_azimuth_bins=8,
                    n_distance_bins=8
                )
            # proceed the pairs
            for net in custom_inventory:
                for stn in net:
                    analyzer.process_snlg_bypair(
                        task=(evt, net.code, stn),
                        inventory=custom_inventory,
                        get_seismic_trace_func=get_waveform_from_client,
                        client=client,
                        channel_requested='BH*',
                        location_requested="*")
        elif kind == "END":
            break

    analyzer.output_final_snlg_list()
    logger0.info(f"[Worker {worker_id}] done.")

def run_parallel_snlg(catalog, analyzer_config,
                     client, maximum_station, num_workers=None):
    num_workers = num_workers or (os.cpu_count() or 1)
    q = Queue()

    # put in all tasks
    for idx in range(len(catalog)):
        q.put(("byevent", idx))
    # put in end signals
    for _ in range(num_workers):
        q.put(END_SIGNAL)

    procs = []
    for wid in range(num_workers):
        p = mp.Process(
            target=run_worker_instance,
            args=(wid, q, analyzer_config,
                  catalog, client, maximum_station)
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

if __name__ == "__main__":

    n_workers = round(os.cpu_count() * 5)
    # n_workers = 100
    MAXIMUM_STATION = 40

    estimated_dump_length = round(3000/n_workers)

    # Define the directory paths
    BASE_DIRECTORY = './Test_Parallel/'
    RAW_DIRECTORY = BASE_DIRECTORY + 'raw/'
    CORRECTED_DIRECTORY = BASE_DIRECTORY + 'corrected/'
    SELECTED_DIRECTORY = BASE_DIRECTORY + 'selected/'
    PROJECT_NAME = 'Test_Parallel'

    os.makedirs(BASE_DIRECTORY, exist_ok=True)
    logger0 = init_main_logger(log_path=os.path.join(BASE_DIRECTORY,"snlg_parallel_log.txt"))
    logger0.info(f"estimated_dump_length: {estimated_dump_length}")

    # Initialize class configuration
    CONFIG = {
        "model": 'iasp91',   # Make sure the file exists when reading model from user-defined file. Default: 'iasp91'
        "project_name": PROJECT_NAME,
        "base_directory": BASE_DIRECTORY,
        "raw_directory": RAW_DIRECTORY,
        "corrected_directory": CORRECTED_DIRECTORY,
        "selected_directory": SELECTED_DIRECTORY,
        "enable_write_sac": False,   # Whether to output SAC files. Please use SnLg_Analyses.prepare_directory to prepare directories if this is true.
        "enable_archive_waveform": True, # Whether to achive raw ZNE waveforms and response-removed T component waveform
        "enable_remove_resp": True,  # Whether to remove instrument response
        "enable_sanity_check": True, # Whether to apply sanity check (i.e., horizontal components are available)
        "enable_trim_edge": True,    # Whether to trim edge - avoid some artificial signals at margins
        "edge_trim_length": 10,      # Edge length to be trimmed
        "prefilt_window": (0.05,0.1,10,20), # Prefilter used in removing instrument response
        "sn_filt":(1.0, 4.0),        # Sn filter 
        "lg_filt":(0.5, 4.0),
        "enable_zero_phase_filter": True,
        "min_epi_distance": 350,
        "max_epi_distance": 2000,
        "seconds_before_P": 60,
        "seconds_after_P": 400,
        "noise_windowlen": 20,
        "noise_offset": 5,
        "vsm": 4.7,
        "vsc": 3.7,
        "moho_snlg": 30,
        "snlg_SNR_threshold": 3,
        "enable_overwrite_log": True,
        "enable_new_log"  : False,
        "log_fname": BASE_DIRECTORY + "log_iris",
        "enable_console_output": False,
        "snlglist_filename": "snlg_list_iris",
        "enable_chunk_list": True,
        "max_records_before_dump": estimated_dump_length
    }

    # Read Inventory and Catalog
    cata_df = pd.read_csv('./Inputs/test.csv')
    # Get client
    client = Client("IRIS", timeout=60)

    start_time = time.time()
    run_parallel_snlg(num_workers=n_workers, analyzer_config=CONFIG,
                    catalog=cata_df, client = client, maximum_station = MAXIMUM_STATION)
    end_time = time.time()
    logger0.info(f"Executing time: {round((end_time-start_time)/60,1)} min")