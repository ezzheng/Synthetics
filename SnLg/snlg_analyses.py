import pathlib
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from obspy.clients.fdsn import Client
from obspy.taup import TauPyModel
from obspy.core.inventory import Inventory, Network, Station, Channel, Site
from obspy import read_inventory, UTCDateTime, Trace, Stream
from obspy.geodetics import gps2dist_azimuth, kilometer2degrees
from obspy.io.sac.sactrace import SACTrace
from scipy.optimize import brentq
# from pympler import asizeof
import os
import glob
import logging
import pickle
import random

# Module-level logger
logger = logging.getLogger(__name__)

@dataclass
class SnLg_OneEventStationPair:
    """Container for event-station pair metadata and calculated values"""
    ## Required metadata
    event: object
    station: Station
    ## Optional fields (populated later)
    # Waveform information
    zne_comp_waveform_raw: Optional[Stream] = None     # For skynet input
    t_comp_waveform: Optional[Trace] = None
    t_comp_waveform_for_sn: Optional[Trace] = None
    t_comp_waveform_for_lg: Optional[Trace] = None
    pre_filt: Optional[tuple] = None
    sn_filt: Optional[tuple] = None
    lg_filt: Optional[tuple] = None
    # Event information:
    evla: Optional[float] = None
    evlo: Optional[float] = None
    evdp: Optional[float] = None
    mag: Optional[float] = None
    evdp_err: Optional[float] = None
    moho: Optional[List[float]] = None
    origin_time: Optional[UTCDateTime] = None
    evcomment: Optional[str] = None
    # Station information
    stla: Optional[float] = None
    stlo: Optional[float] = None
    stel: Optional[float] = None
    kstnm: Optional[str] = None
    knetwk: Optional[str] = None
    # Event-Station pair information
    gcarc: Optional[float] = None
    dist: Optional[float] = None
    baz: Optional[float] = None
    p_arrival: Optional[UTCDateTime] = None
    a_sn: Optional[float] = None
    a_lg: Optional[float] = None
    sn_window: Optional[tuple] = None
    lg_window: Optional[tuple] = None
    noise_window: Optional[tuple] = None
    # Sn/Lg results
    amp_sn: Optional[float] = None
    amp_lg: Optional[float] = None
    amp_noise_sn: Optional[float] = None
    amp_noise_lg: Optional[float] = None
    SNR: Optional[float] = None
    SNR_Sn: Optional[float] = None
    SNR_Lg: Optional[float] = None
    SnRLg_raw: Optional[float] = None
    SnRLg_cor: Optional[float] = None
    SnRLg_err: Optional[float] = None
    SnRLg_raw_ratioed: Optional[float] = None
    max_sn: Optional[float] = None
    max_lg: Optional[float] = None
    max_SnRLg_raw: Optional[float] = None
    # Other information
    vsm: Optional[float] = None
    vsc: Optional[float] = None
    is_rmresp: Optional[bool] = None
    moho_snlg: Optional[float] = None
    # Skynet
    # pick_pn: Optional[list] = None
    # pick_pg: Optional[list] = None
    # pick_sn: Optional[list] = None
    # pick_lg: Optional[list] = None


def _get_available_log_name(base_name="log", ext="log"):
    """
    If 'log.log' exists, returns 'log_1.log', 'log_2.log', etc., choosing
    the first available filename.
    """
    candidate = f"{base_name}.{ext}"
    idx = 1
    # Keep bumping the suffix until we find a name that doesn't exist
    while os.path.exists(candidate):
        candidate = f"{base_name}_{idx}.{ext}"
        idx += 1
    return candidate

class SnLg_Analyses:
    def __init__(self, **kwargs):
        # 0） Build and override defaults
        defaults = {
            "model": 'iasp91',  
            "project_name": 'Z_SnLg',
            "base_directory": './Z_SnLg/',
            "raw_directory": './Z_SnLg/raw/',
            "corrected_directory": './Z_SnLg/corrected/',
            "selected_directory": './Z_SnLg/selected/',
            "enable_write_sac": False,
            "enable_archive_waveform": True,
            "enable_remove_resp": True,
            "enable_sanity_check": True,
            "enable_trim_edge": True,
            "edge_trim_length": 0,
            "prefilt_window": (0.05, 0.1, 10, 20),
            "sn_filt":(0.1, 0.5), # originally 1-4
            "lg_filt":(0.1, 0.5), # originally 0.5-4
            "enable_zero_phase_filter": False,
            "min_epi_distance": 350,
            "max_epi_distance": 2000,
            "seconds_before_P": 60,
            "seconds_after_P": 400,
            "noise_windowlen": 20,  # From Axel & Klemperer (2021)
            "noise_offset": 5,      # From Axel & Klemperer (2021)
            "vsm": 4.7,
            "vsc": 3.7,
            "moho_snlg": 70,
            "snlg_onepair":None,
            "snlg_list": [],
            "snlg_SNR_threshold": 3,
            "enable_overwrite_log": False,
            "enable_new_log": False,
            "log_fname": "log",
            "enable_console_output": False,
            "snlglist_filename": "snlg_list",
            "enable_chunk_list": False,
            "max_records_before_dump": 3000
        }
        
        defaults.update(kwargs)

        self.taupy_model      = TauPyModel(model=defaults["model"])
        self.project_name     = defaults["project_name"]
        self.base_directory   = defaults["base_directory"]
        self.raw_directory    = defaults["raw_directory"]
        self.corrected_directory = defaults["corrected_directory"]
        self.selected_directory  = defaults["selected_directory"]
        self.enable_write_sac    = defaults["enable_write_sac"]
        self.enable_archive_waveform = defaults["enable_archive_waveform"]
        self.enable_remove_resp  = defaults["enable_remove_resp"]
        self.enable_sanity_check = defaults["enable_sanity_check"]
        self.enable_trim_edge    = defaults["enable_trim_edge"]
        self.edge_trim_length = defaults["edge_trim_length"]
        self.prefilt_window   = defaults["prefilt_window"]
        self.sn_filt          = defaults["sn_filt"]
        self.lg_filt          = defaults["lg_filt"]
        self.enable_zero_phase_filter = defaults["enable_zero_phase_filter"]
        self.min_epi_distance = defaults["min_epi_distance"]
        self.max_epi_distance = defaults["max_epi_distance"]
        self.seconds_before_P = defaults["seconds_before_P"]
        self.seconds_after_P  = defaults["seconds_after_P"]
        self.noise_windowlen  = defaults["noise_windowlen"]
        self.noise_offset     = defaults["noise_offset"]
        self.vsm = defaults["vsm"]
        self.vsc = defaults["vsc"]
        self.moho_snlg     = defaults["moho_snlg"]
        self.snlg_onepair  = defaults["snlg_onepair"]
        self.snlg_list     = defaults["snlg_list"]
        self.snlg_SNR_threshold = defaults["snlg_SNR_threshold"]
        self.snlglist_filename  = defaults["snlglist_filename"]
        self.enable_chunk_list  = defaults["enable_chunk_list"]
        self.max_records_before_dump    = defaults["max_records_before_dump"] # Convert to bytes

        # 1） Create base directory
        # os.makedirs(self.base_directory, exist_ok = True)

        # 2) Decide log filename
        if defaults["enable_new_log"]:
            log_file = _get_available_log_name(base_name=defaults["log_fname"], ext="log")
        else:
            log_file = defaults["log_fname"]+".log"

        # 3） Configure logging
        handler_file  = logging.FileHandler(log_file,mode='a')
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        handler_file.setFormatter(formatter)
        # Get the root logger and add the handler
        root_logger = logging.getLogger()
        root_logger.addHandler(handler_file)
        root_logger.setLevel(logging.INFO)  # Set the root logger level

        # 4） Possibly wipe log file
        if defaults["enable_overwrite_log"]:
            open(log_file, 'w').close()
            # logger.info(f"Log file {log_file} has been cleared.")

        # 5） Set console output
        if defaults["enable_console_output"]:
            handler_console = logging.StreamHandler()  # Add StreamHandler for console output
            handler_console.setFormatter(formatter)
            root_logger.addHandler(handler_console) 

        # 6） Check saved list
        if self.enable_chunk_list and glob.glob(os.path.join(self.base_directory,self.snlglist_filename)+"_[0-9].pkl"):
            raise Exception(
                f"File already exist with pattern {os.path.join(self.base_directory,self.snlglist_filename)}_[0-9].pkl"
                f"Please check the directory {os.path.join(self.base_directory)} to avoid mixing of data."
            )
            # print(f"########### WARNING ###########"
            #     f"File already exist with pattern {os.path.join(self.base_directory,self.snlglist_filename)}_[0-9].pkl"
            #     f"Please check the directory {os.path.join(self.base_directory)} to avoid mixing of data."
            #     f"#################################"
            # )

    def clear_snlg_list(self):
        """
        Clear the snlg_list attribute to an empty list.
        """
        self.snlg_list.clear()
        logger.info(f"snlg_list is cleared.")

    def output_snlg_list(self,filename=None, overwrite=False):
        """
        Save the current snlg_list to a pickle file.

        Args:
            filename (str, optional): The path to the file where the list will be saved.
                                    Defaults to '<base_directory>/snlg_list.pkl'.
        """
        if filename is None:
            filename = os.path.join(self.base_directory, "snlg_list.pkl")
        else:
            filename = os.path.join(self.base_directory, filename)
        if not overwrite and os.path.exists(filename):
            print(f"Stop output snlg_list as {filename}, because the file already exists")
            logger.info(f"Stop output snlg_list as {filename}, because the file already exists")
            return False
        try:
            if os.path.exists(filename):
                print(f"Overwriting existing file: {filename}")
                logger.info(f"Overwriting existing file: {filename}")
            with open(filename,"wb") as f:
                pickle.dump(self.snlg_list,f)
            logger.info(f"snlg_list is outputed as {filename}.")
            return True
        except Exception as e:
            logger.exception(f"Error saving snlg_list to {filename}: {e}")
            return False
    
    def output_final_snlg_list(self,clean_list=False):
        if self.enable_chunk_list:
            fname = self._get_chunk_filename(base_name=self.snlglist_filename, ext="pkl")
        else:
            fname = self.snlglist_filename+".pkl"
        self.output_snlg_list(filename = fname, overwrite = False)
        logger.info(f"[Output] Final round outputting snlg_list to {fname}")
        if clean_list:
            self.clear_snlg_list()
    
    def read_snlg_list(self,filename=None):
        """
        Read the snlg_list from a pickle file and update self.snlg_list.

        Args:
            filename (str, optional): The path to the file to read from.
                                    Defaults to '<base_directory>/snlg_list.pkl'.
        """
        if filename is None:
            filename = os.path.join(self.base_directory, "snlg_list.pkl")
        try:
            with open(filename, "rb") as f:
                self.snlg_list = pickle.load(f)
                logger.info(f"snlg_list successfully loaded from {filename}.")
        except Exception as e:
            logger.error(f"Error in reading snlg_list from {filename}.")

    def _get_chunk_filename(self,base_name="snlg_list", ext="pkl"):
        """
        If 'snlg_list_0.pkl' exists, returns 'snlg_list_1.pkl', 'snlg_list_2.pkl', etc., 
        choosing the first available filename.
        """
        idx = 0
        # Keep bumping the suffix until we find a name that doesn't exist
        while True:
            candidate = f"{base_name}_{idx}.{ext}"
            # Files are outputted under base_directory by default
            if os.path.exists(os.path.join(self.base_directory,candidate)):
                idx += 1
            else:
                break
        return candidate

    @staticmethod
    def prepare_directory(path):
        """
        Checks if a directory exists and creates it if not. If it exists, deletes all `.SAC` files inside.
        """
        try:
            logger.info(f"Preparing directory: {path}")
            path_obj = pathlib.Path(path)
            if not path_obj.exists():
                path_obj.mkdir(parents=True, exist_ok=True)
                logger.info(f"Directory {path} did not exist and is created now.")
            else:
                for sac_file in path_obj.glob('*.SAC'):
                    sac_file.unlink()
                    logger.info(f"Deleted file: {sac_file}")
                logger.info(f"Directory {path} already exists. All .SAC files under this directory have been removed.")
        except Exception as e:
            logger.error(f"Error preparing the directory {path}: {e}")

    @staticmethod
    def write_sac_trace(trace,output_dir,sac_headers):
        """
        Write a single SAC trace to a file with the given sac headers.
        """
        time_str = sac_headers['reftime'].strftime("%Y-%j-%H-%M-%S")
        loc = sac_headers.get('loc','00')
        sac = SACTrace.from_obspy_trace(trace)
        for key, value in sac_headers.items():
            if hasattr(sac, key):
                setattr(sac, key, value)
            elif key == 'loc':
                continue  # loc is not a SAC header, but it exists in trace.stats.location
            else:
                logger.warning(f"Warning: SAC header {key} not recognized")
        sacnm = os.path.join(
            output_dir,
            f"{sac_headers['knetwk']}.{sac_headers['kstnm']}.{loc or '00'}.{time_str}.{sac_headers['kcmpnm']}.SAC")
        sac.write(sacnm)

    @staticmethod
    def snlg(d, H, g, vsm, vsc):
        """
        Calculate travel times for Sn and Lg phases based on input parameters.
        
        Parameters:
            d (float): Event depth.
            H (float): Reference Moho depth.
            g (float): Epicentral distance in km.
            vsm (float): Velocity of the S-wave in the mantle.
            vsc (float): Velocity of the S-wave in the crust.
        
        Returns:
            np.array: [tsn, tlg] travel time estimates.
        """
        # Convert inputs to floats
        d, H, g, vsm, vsc = map(float, [d, H, g, vsm, vsc])

        if d > H:
            # Handle deep earthquakes
            def equation(x):
                """Equation governing Sn travel time derivation"""
                term1 = -(g - x) / (np.sqrt((g - x)**2 + H**2) * vsc)
                term2 = x / (np.sqrt(x**2 + (d - H)**2) * vsm)
                return term1 + term2

            try:
                # Physically bounded solution space: 0 < x < g
                # Find the root using Brent’s method.
                xn = brentq(equation, 0, g, xtol=1e-4, maxiter=100)
            except ValueError as e:
                raise ValueError(
                    f"No valid Sn solution for d={d}km, H={H}km, g={g}km, "
                    f"vsm={vsm}km/s, vsc={vsc}km/s. Error: {str(e)}"
                ) from e

            # Calculate final travel times
            # Equation (S1.1d) in Song & Klemperer (2024)
            tsn = (np.sqrt((g - xn)**2 + H**2) / vsc +
                np.sqrt(xn**2 + (d - H)**2) / vsm)
            # Equation (S1.1b) in Song & Klemperer (2024)
            tlg = ((d - H) / vsm) + (np.sqrt(H**2 + g**2) / vsc)
        else:
            # Handle shallow earthquakes (d <= H)
            velocity_term = np.sqrt(vsm**2 - vsc**2) / (vsm * vsc)
            # Equation (S1.1c) in Song & Klemperer (2024)
            tsn = (g / vsm) + (2 * H - d) * velocity_term
            # Equation (S1.1a) in Song & Klemperer (2024)
            tlg = np.sqrt(d**2 + g**2) / vsc

        return np.array([tsn, tlg])

    def _write_sac_stream(self,stream,output_dir):
        """
        Loop over each trace in a stream and write it to a SAC file with constructed headers.
        """
        # Build sac header
        for tr in stream:
            sac_headers = {
                # Event parameters
                'evlo': self.snlg_onepair.evlo,
                'evla': self.snlg_onepair.evla,
                'evdp': self.snlg_onepair.evdp,
                'mag': self.snlg_onepair.mag,
                
                # Station parameters
                'stlo': self.snlg_onepair.stlo,
                'stla': self.snlg_onepair.stla,
                'stel': self.snlg_onepair.stel,
                'kstnm': self.snlg_onepair.kstnm,
                'knetwk': self.snlg_onepair.knetwk,
                
                # Timing parameters ###### How to write reftime, o, and a? ######
                'reftime': self.snlg_onepair.origin_time,
                'o': self.snlg_onepair.origin_time,
                'a': self.snlg_onepair.p_arrival,
                'ka': 'P',
                
                # Distance parameters
                'gcarc': self.snlg_onepair.gcarc,
                'dist': self.snlg_onepair.dist,
                'baz': self.snlg_onepair.baz,
                
                # other parameters
                'kcmpnm': tr.stats.channel,
                'loc':    tr.stats.location,
                'user0': getattr(self.snlg_onepair, 'evdp_err', -9999.0),
                'kuser0': getattr(self.snlg_onepair, 'evcomment', None)
            }
            self.write_sac_trace(trace=tr, output_dir=output_dir, sac_headers=sac_headers)


    def detrend_and_remove_response(self, stream, prefilt_window=None):
        """
        Detrend and remove instrument response from the seismic stream.

        Args:
            stream (Stream): Seismic data stream
            prefilt_window (tuple): Frequency limits for pre-filtering.
        
        Returns:
            stream (Stream): Detrended and response-corrected seismic data
            is_rmresp (bool): True if response removal is successful, False otherwise
        """
        if prefilt_window is None:
            prefilt_window = self.prefilt_window

        is_rmresp = True
        for tr in stream:
            if self.enable_trim_edge:
                # Be careful with the use of tr.stats.starttime and tr.stats.endtime
                # Pre-processing trimming: trim possible high amplitude signals at the edge in raw data
                tr.trim(tr.stats.starttime + self.edge_trim_length / 2, 
                        tr.stats.endtime   - self.edge_trim_length / 2)
            
            try:
                tr.detrend(type='constant')
                tr.detrend(type='linear')
            except Exception as e:
                logger.error(f"Error in detrend. Please check trace {tr.id}: {e}")
            if self.enable_remove_resp:
                try:
                    tr.remove_response(output='VEL',taper = True, taper_fraction = 0.05,
                                        pre_filt = prefilt_window, water_level = 60.0)
                    logger.info(f"Success in removing response for trace {tr.id}")
                except Exception as e:
                    logger.warning(f"Error removing response for trace {tr.id}: {e}")
                    is_rmresp = False
            else:
                is_rmresp = False
        return stream, is_rmresp
    
    def process_snlg_bulk(self, catalog_dataframe, inventory, 
                            get_seismic_trace_func, model=None,
                            min_epi_distance=None, max_epi_distance=None,
                            seconds_before_P=None, seconds_after_P=None,
                            pre_filter=None, **kwargs):
        """
        Process a bulk of events and stations, fetching seismic data and performing Sn/Lg analysis.
        The outer loop is determined by which collection (events or stations) is smaller.
        """
        # update default parameters
        model = model or self.taupy_model
        min_epi_distance = min_epi_distance or self.min_epi_distance
        max_epi_distance = max_epi_distance or self.max_epi_distance
        seconds_before_P = seconds_before_P or self.seconds_before_P
        seconds_after_P = seconds_after_P or self.seconds_after_P
        pre_filter = pre_filter or self.prefilt_window

        # Flatten inventory to list of (network, station)
        station_list = [(net.code, stn) for net in inventory for stn in net]
        events = list(catalog_dataframe.itertuples())
        num_events = len(events)
        num_stations = len(station_list)
        logger.info(f"Found {num_events} events and {num_stations} stations.")

        # choose iteration order
        if num_events <= num_stations:
            logger.info("Looping over events first.")
            tasks = (
                (evt, net_code, stn) for evt in events for (net_code, stn) in station_list
            )
        else:
            logger.info("Looping over stations first.")
            tasks = (
                (evt, net_code, stn) for (net_code, stn) in station_list for evt in events
            )
        
        # process each pair
        for event, net_code, stn in tasks:
            try:
                self._process_pair(
                    event=event,
                    stn=stn,
                    network_code=net_code,
                    model=model,
                    min_epi_distance=min_epi_distance,
                    max_epi_distance=max_epi_distance,
                    seconds_before_P=seconds_before_P,
                    seconds_after_P=seconds_after_P,
                    pre_filter=pre_filter,
                    inventory=inventory,
                    get_seismic_trace_func=get_seismic_trace_func,
                    **kwargs
                )
            except Exception as e:
                logger.error(f"Error processing event {getattr(event, 'event_time', '')} "
                             f"and station {net_code}.{stn.code}: {e}")
                continue

    def process_snlg_bypair(self, task, get_seismic_trace_func, 
                            inventory, model=None,
                            min_epi_distance=None, max_epi_distance=None,
                            seconds_before_P=None, seconds_after_P=None,
                            pre_filter=None, **kwargs):
        """
        A parallel interface for self._process_pair
        """
        # unwrap the task
        event, net_code, stn = task
        # update default parameters
        model = model or self.taupy_model
        min_epi_distance = min_epi_distance or self.min_epi_distance
        max_epi_distance = max_epi_distance or self.max_epi_distance
        seconds_before_P = seconds_before_P or self.seconds_before_P
        seconds_after_P  = seconds_after_P or self.seconds_after_P
        pre_filter       = pre_filter or self.prefilt_window
        # call pair processing function
        try:
            self._process_pair(
                event=event,
                stn=stn,
                network_code=net_code,
                model=model,
                min_epi_distance=min_epi_distance,
                max_epi_distance=max_epi_distance,
                seconds_before_P=seconds_before_P,
                seconds_after_P=seconds_after_P,
                pre_filter=pre_filter,
                inventory=inventory,
                get_seismic_trace_func=get_seismic_trace_func,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error processing event {getattr(event, 'event_time', '')} "
                            f"and station {net_code}.{stn.code}: {e}")


    def _process_pair(self, event, stn, network_code, model,
                    min_epi_distance, max_epi_distance,
                    seconds_before_P, seconds_after_P,
                    pre_filter, inventory, get_seismic_trace_func, **kwargs):
        """
        Process a single event–station pair.
        
        Parameters:
            event: Information about the Earthquake.
            stn: Obspy Station (from the inventory).
            network_code: Network Code.
            model: Obspy TauPyModel.
            min_epi_distance, max_epi_distance: Distance thresholds.
            seconds_before_P, seconds_after_P: Time windows around P arrival.
            pre_filter: Pre-filter parameters.
            inventory: The inventory object.
            get_seismic_trace_func: External function handle to fetch seismic data.
            kwargs: Any additional parameters.
        """
        # Extract event parameters.
        evlo = event.event_lon
        evla = event.event_lat
        evdp = event.event_dep
        origin_time = UTCDateTime(event.event_time)
        mag = event.event_mag
        # moho = getattr(event, "Moho", None)
        moho = [getattr(event, attr) for attr in event._fields if attr.startswith("Moho")]
        evdp_err = getattr(event, "event_depErr", None)
        evcomment = getattr(event, "event_comment", None)
        
        # Extract station parameters.
        stla = stn.latitude
        stlo = stn.longitude
        epi_dist, baz, _ = gps2dist_azimuth(stla, stlo, evla, evlo)
        epi_dist_km = epi_dist / 1000.0  # convert to km
        gcarc = kilometer2degrees(epi_dist_km)  # convert distance to degrees
        
        logger.info(f"-----------------------------------------------------------")
        logger.info(f"[Process Start] Event ({origin_time}) at station {stn.code}")
        
        # Check if distance is within acceptable range.
        if not (min_epi_distance <= epi_dist_km <= max_epi_distance):
            logger.info(f"Event ({origin_time}) station {stn.code}: Distance {epi_dist_km:.1f} km out of range")
            return
        
        # Calculate travel times for P phase.
        p_arrivals = model.get_travel_times(source_depth_in_km=evdp,
                                            distance_in_degree=gcarc,
                                            phase_list=['p', 'P'])
        if not p_arrivals:
            logger.warning(f"No travel time found for event ({origin_time}) at station {stn.code}")
            return
        
        p_arrival = origin_time + p_arrivals[0].time
        if self.enable_trim_edge:
            starttime = p_arrival - seconds_before_P - self.edge_trim_length
            endtime = p_arrival + seconds_after_P + self.edge_trim_length
        else:
            starttime = p_arrival - seconds_before_P
            endtime = p_arrival + seconds_after_P

        # Create a dataclass for the event-station pair.
        # Convert the event (pandas namedtuples) to a dict. Pickle may not able to save pandas namedtuples directly.
        self.snlg_onepair = SnLg_OneEventStationPair(event=event._asdict(), station=stn)
        self.snlg_onepair.pre_filt = pre_filter
        self.snlg_onepair.sn_filt = self.sn_filt
        self.snlg_onepair.lg_filt = self.lg_filt
        self.snlg_onepair.evla = evla
        self.snlg_onepair.evlo = evlo
        self.snlg_onepair.evdp = evdp
        self.snlg_onepair.mag = mag
        self.snlg_onepair.evdp_err = evdp_err
        self.snlg_onepair.evcomment = evcomment
        self.snlg_onepair.moho = moho
        self.snlg_onepair.origin_time = origin_time
        self.snlg_onepair.stla = stla
        self.snlg_onepair.stlo = stlo
        self.snlg_onepair.stel = stn.elevation
        self.snlg_onepair.kstnm = stn.code
        self.snlg_onepair.knetwk = network_code
        self.snlg_onepair.gcarc = gcarc
        self.snlg_onepair.dist = epi_dist_km
        self.snlg_onepair.baz = baz
        self.snlg_onepair.p_arrival = p_arrival
        self.snlg_onepair.vsm = self.vsm
        self.snlg_onepair.vsc = self.vsc
        self.snlg_onepair.moho_snlg = self.moho_snlg  # Can be modified as needed.
    
        # Fetch seismic data.
        try:
            stream = get_seismic_trace_func(starttime=starttime, endtime=endtime, station=stn, network_code=network_code, **kwargs)
        except Exception as e:
            logger.error(f"[Missing Data] Error fetching data for event ({origin_time}) at station {stn.code}: {e}")
            return
        if stream is None:
            logger.warning(f"[Missing Data] Event ({origin_time}) at station {stn.code}: Stream is None")
            return
        
        # Merge fragmented traces
        stream.merge()
        logger.info(f"Success fetched data for event ({origin_time}) at station {stn.code}")
        if not hasattr(stream[0].stats,"response"):
            stream.attach_response(inventory)
        
        # Ensure the stream is in the ZNE coordinate system.
        if not all(tr.stats.channel.endswith(("E", "N", "Z")) for tr in stream):
            logger.info(f"Channels for station {stn.code} are not in ZNE, rotating to ZNE.")
            logger.info(f"Current Channels are: {[tr.stats.channel for tr in stream]}")
            stream.rotate(method="->ZNE", inventory=inventory)

        # Perform sanity check.
        if self.enable_sanity_check:
            channels = [tr.stats.channel for tr in stream]
            has_E = any(ch.endswith("E") for ch in channels)
            has_N = any(ch.endswith("N") for ch in channels)

            if not (has_E and has_N):
                logger.warning(
                    f"[Missing Data] Event ({origin_time}) at station {stn.code}: "
                    f"Missing horizontal components. Current components: {channels}"
                )
                return

        
        # Successfully obtained a seismic data stream for this event-station pair. Next, process the stream for Sn/Lg analysis.
        if self.enable_archive_waveform:
            self.snlg_onepair.zne_comp_waveform_raw = stream.copy() # stream containing raw ZNE components. Use copy to avoid any further processing.
        self.process_snlg_one_stream(stream=stream)
        # Append the pair to list
        self.snlg_list.append(self.snlg_onepair)
        # Check if the size of self.snlg_list exceeds the threshold
        if self.enable_chunk_list:
            ############### Method 1: chunk by the size of self.snlg_list, but this consumes much time ###############
            # if len(self.snlg_list) % 1500 == 0:  # reduce the amount of time to check, because check would causes a lot of time
            #     sample_size = 200
            #     fsize_sample = asizeof.asizeof(random.sample(self.snlg_list,sample_size)) 
            #     fsize = fsize_sample * len(self.snlg_list) / sample_size
            #     if fsize >= self.chunk_threshold:
            #         fname = self._get_chunk_filename(base_name=self.snlglist_filename, ext="pkl")
            #         self.output_snlg_list(filename = fname, overwrite = False)
            #         logger.info(f"[Output] Estimated size of snlg_list ({fsize} bytes) exceeds the threshold, output as {fname}")
            #         self.clear_snlg_list()
            ############### Method 2: chunk by the length of self.snlg_list ###################
            if len(self.snlg_list) >= self.max_records_before_dump:
                fname = self._get_chunk_filename(base_name=self.snlglist_filename, ext="pkl")
                self.output_snlg_list(filename = fname, overwrite = False)
                logger.info(f"[Output] Chunk snlg list and output as {fname}")
                self.clear_snlg_list()
        logger.info(f"[Process End]")

    def process_snlg_bypair_synthetic(self, stream,
                            epi_dist_km,
                            event=None, origin_time=None, evdp=None, baz=None, model=None,
                            min_epi_distance=None, max_epi_distance=None,
                            seconds_before_P=None, seconds_after_P=None,
                            pre_filter=None, **kwargs):
        # update default parameters
        model = model or self.taupy_model
        min_epi_distance = min_epi_distance or self.min_epi_distance
        max_epi_distance = max_epi_distance or self.max_epi_distance
        seconds_before_P = seconds_before_P or self.seconds_before_P
        seconds_after_P  = seconds_after_P or self.seconds_after_P
        pre_filter       = pre_filter or self.prefilt_window
        # call pair processing function
        try:
            self._process_pair_synthetic(
                stream = stream,
                epi_dist_km = epi_dist_km,
                model=model,
                min_epi_distance=min_epi_distance,
                max_epi_distance=max_epi_distance,
                seconds_before_P=seconds_before_P,
                seconds_after_P=seconds_after_P,
                pre_filter=pre_filter,
                event=event,
                origin_time=origin_time,
                evdp = evdp,
                baz=baz,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Error processing synthetic event {origin_time}")

    def _process_pair_synthetic(self, stream, epi_dist_km, model,
                min_epi_distance, max_epi_distance,
                seconds_before_P, seconds_after_P,
                pre_filter, event=None, origin_time=None, evdp=None, baz=None, **kwargs):
        if event:
            # Get Event information if it is provided
            evlo = event.event_lon
            evla = event.event_lat
            evdp = event.event_dep or evdp
            origin_time = UTCDateTime(event.event_time) if getattr(event, "event_time", None) else origin_time
            mag = event.event_mag
            moho = [getattr(event, attr) for attr in event._fields if attr.startswith("Moho")]
            evdp_err = getattr(event, "event_depErr", None)
            evcomment = getattr(event, "event_comment", None)
        else:
            # No Information
            evlo = evla = mag = evdp_err = evcomment = None
            moho = []
        # Check if origin time and event depth exist.
        if origin_time is None:
            logger.error(f"Event origin time is not provided for the synthetic event!")
            return
        if evdp is None:
            logger.error(f"Event depth is not provided for the synthetic event!")
            return

        # Synthetics don't have station information. We will use epicentral distance, event depth directly.
        logger.info(f"-----------------------------------------------------------")
        logger.info(f"[Process Start] Synthetic Event ({origin_time})")

        # Check if distance is within acceptable range.
        if not (min_epi_distance <= epi_dist_km <= max_epi_distance):
            logger.info(f"Skip synthetic event ({origin_time}): Distance {epi_dist_km:.1f} km out of range; min_epi_distance: {min_epi_distance}, max_epi_distance: {max_epi_distance}")
            return

        # Calculate travel times for P phase.
        gcarc = kilometer2degrees(epi_dist_km)
        p_arrivals = model.get_travel_times(source_depth_in_km=evdp,
                                            distance_in_degree=gcarc,
                                            phase_list=['p', 'P'])

        if not p_arrivals:
            logger.warning(f"No travel time found for synthetic event ({origin_time})")
            return
        
        p_arrival = origin_time + p_arrivals[0].time
        if self.enable_trim_edge:
            starttime = p_arrival - seconds_before_P - self.edge_trim_length
            endtime = p_arrival + seconds_after_P + self.edge_trim_length
        else:
            starttime = p_arrival - seconds_before_P
            endtime = p_arrival + seconds_after_P

        # Create a dataclass for the event-station pair.
        # Convert the event (pandas namedtuples) to a dict. Pickle may not able to save pandas namedtuples directly.
        if event:
            self.snlg_onepair = SnLg_OneEventStationPair(event=event._asdict(), station=None)
            self.snlg_onepair.evla = evla
            self.snlg_onepair.evlo = evlo
            self.snlg_onepair.mag = mag
            self.snlg_onepair.evdp_err = evdp_err
            self.snlg_onepair.evcomment = evcomment
            self.snlg_onepair.moho = moho
        else:
            self.snlg_onepair = SnLg_OneEventStationPair(event=None, station=None)
        self.snlg_onepair.pre_filt = pre_filter
        self.snlg_onepair.sn_filt = self.sn_filt
        self.snlg_onepair.lg_filt = self.lg_filt
        self.snlg_onepair.origin_time = origin_time
        self.snlg_onepair.gcarc = gcarc
        self.snlg_onepair.dist = epi_dist_km
        self.snlg_onepair.evdp = evdp
        self.snlg_onepair.baz = baz
        self.snlg_onepair.p_arrival = p_arrival
        self.snlg_onepair.vsm = self.vsm
        self.snlg_onepair.vsc = self.vsc
        self.snlg_onepair.moho_snlg = self.moho_snlg  # Can be modified as needed.
    
        # Process Seismic Data
        stream.trim(starttime, endtime, pad=True, fill_value=0.0) # Check the fill_value!
        logger.info(f"Success clipped data for synthetic event ({origin_time})")

        # Perform sanity check.
        if self.enable_sanity_check:
            channels = [tr.stats.channel for tr in stream]
            has_E = any(ch.endswith("E") for ch in channels)
            has_N = any(ch.endswith("N") for ch in channels)
            logger.info(f"Synthetic Event ({origin_time}) has channels: {channels}")

            if not (has_E and has_N):
                logger.warning(
                    f"[Missing Data] Synthetic Event ({origin_time}) "
                    f"Missing horizontal components. Current components: {channels}"
                )
                return
        

        # Successfully obtained a seismic data stream for this event-station pair. Next, process the stream for Sn/Lg analysis.
        if self.enable_archive_waveform:
            self.snlg_onepair.zne_comp_waveform_raw = stream.copy() # stream containing raw ZNE components. Use copy to avoid any further processing.
        self.process_snlg_one_stream(stream=stream)
        # Append the pair to list
        self.snlg_list.append(self.snlg_onepair)
        # Check if the size of self.snlg_list exceeds the threshold
        if self.enable_chunk_list:
            ############### Method 1: chunk by the size of self.snlg_list, but this consumes much time ###############
            # if len(self.snlg_list) % 1500 == 0:  # reduce the amount of time to check, because check would causes a lot of time
            #     sample_size = 200
            #     fsize_sample = asizeof.asizeof(random.sample(self.snlg_list,sample_size)) 
            #     fsize = fsize_sample * len(self.snlg_list) / sample_size
            #     if fsize >= self.chunk_threshold:
            #         fname = self._get_chunk_filename(base_name=self.snlglist_filename, ext="pkl")
            #         self.output_snlg_list(filename = fname, overwrite = False)
            #         logger.info(f"[Output] Estimated size of snlg_list ({fsize} bytes) exceeds the threshold, output as {fname}")
            #         self.clear_snlg_list()
            ############### Method 2: chunk by the length of self.snlg_list ###################
            if len(self.snlg_list) >= self.max_records_before_dump:
                fname = self._get_chunk_filename(base_name=self.snlglist_filename, ext="pkl")
                self.output_snlg_list(filename = fname, overwrite = False)
                logger.info(f"[Output] Chunk snlg list and output as {fname}")
                self.clear_snlg_list()
        logger.info(f"[Process End]")

    def process_snlg_one_stream(self,stream):
        """
        Process a single seismic stream (one evnet-station pair) for Sn/Lg analysis:
          1. Rotate to RTZ coordinate system and write raw SAC files.
          2. Remove response and write corrected SAC files.
          3. Extract the T component, perform amplitude measurements for Sn, Lg, and noise, and write selected (passes SNR threshold) SAC files.
        """
        # -> write raw data (uncorrected ENZ traces)
        if self.enable_write_sac:
            self._write_sac_stream(stream=stream,output_dir=self.raw_directory)
        # Step1: Remove response
        corrected_stream, is_rmresp = self.detrend_and_remove_response(stream=stream,prefilt_window=self.snlg_onepair.pre_filt)
        self.snlg_onepair.is_rmresp = is_rmresp
        # Step2: Rotate to RTZ coordinate system
        corrected_stream.rotate(method = 'NE->RT', back_azimuth=self.snlg_onepair.baz)
        logger.info(f"Success rotated to RTZ coordinate system for event ({self.snlg_onepair.origin_time})")
        # -> write corrected data (corrected RTZ traces)
        if self.enable_write_sac:
            self._write_sac_stream(stream=corrected_stream,output_dir=self.corrected_directory)
        # Step3: Calculate SNR and sn/lg
        try:
            cor_stream_T = corrected_stream.select(channel="??T")
            ###### Further Check: More than 1 traces? ######
            if len(cor_stream_T) > 1:
                logger.warning(f"{len(cor_stream_T)} T traces are found at {self.snlg_onepair.kstnm} for event {self.snlg_onepair.origin_time}")
            ###### Further Check: More than 1 traces? ######
            cor_trace_T = cor_stream_T[0]
            if self.enable_archive_waveform:
                self.snlg_onepair.t_comp_waveform = cor_trace_T.copy() # Archive t_comp_waveform. Use copy to avoid any further processing.
        except Exception as e:
            logger.error(f"Error in extracting T component from corrected stream: {e}")
            return
        # Get Sn and Lg window
        try:
            sn_lg_arrivals = self.snlg(d=float(self.snlg_onepair.evdp), 
                                    H=float(self.snlg_onepair.moho_snlg),   # Check this: Use fixed moho or varying moho?
                                    g=float(self.snlg_onepair.dist), 
                                    vsm=float(self.snlg_onepair.vsm), 
                                    vsc=float(self.snlg_onepair.vsc))
            logger.info(f"Success in calculating Sn and Lg arrivals for event ({self.snlg_onepair.origin_time})")
        except Exception as e:
            logger.error(f"Error in calculating Sn and Lg arrivals (event {self.snlg_onepair.origin_time}, station {self.snlg_onepair.kstnm}): {e}")
            return
        a_sn, a_lg = sn_lg_arrivals
        
        # MODIFY 
        sn_window = 4  * self.snlg_onepair.gcarc # modified from 3.5
        lg_window = 5  * self.snlg_onepair.gcarc
        lg_start  = a_lg - 0.1*lg_window # modified from .1
        lg_end    = a_lg + 0.9*lg_window # modified from .9
        sn_start  = a_sn - 0.2*sn_window # modified from .2
        sn_end    = min(a_sn + 0.8*sn_window, a_lg - 0.05*lg_window) # modified from .8
        if sn_end == a_lg - 0.05*lg_window:
            lg_start = a_lg - 0.05*lg_window

        self.snlg_onepair.a_sn = a_sn
        self.snlg_onepair.a_lg = a_lg
        self.snlg_onepair.sn_window = (sn_start,sn_end)
        self.snlg_onepair.lg_window = (lg_start,lg_end)
        # Note that: When the earthquake is very shallow and epicentral distance is very close,
        #            lg_start can be earlier than sn_start, given the definition above.
        #            To avoid this situation, it is not recommend to use cases with epicentral distance < 350 km (or even 400 km)
        if lg_start <= sn_start:
            logger.warning(f"Lg start {lg_start:.2f} is earlier than Sn start {sn_start:.2f}. Sn/Lg will NOT be calculated.")
        else:
            # Get RMS measurements of noise, Sn, and Lg measurement
            logger.info(f"Proceeding to calculate Sn/Lg for event ({self.snlg_onepair.origin_time})")
            self.get_SnRLg(trace=cor_trace_T)
            if self.enable_write_sac and self.snlg_onepair.SNR >= self.snlg_SNR_threshold:  # Modify the selection criteria if needed
                self._write_sac_stream(stream=cor_stream_T,output_dir=self.selected_directory)
    
    def get_SnRLg(self,trace):
        """
        Compute the RMS amplitudes for Sn and Lg signals and their noise, then calculate:
          - The maximum SNR,
          - The natural logarithm of the Sn/Lg amplitude ratio,
          - An error estimate for Sn/Lg ratio,
          - A geometric spreading corrected Sn/Lg ratio.
        """

        sn_start,sn_end = self.snlg_onepair.sn_window
        lg_start,lg_end = self.snlg_onepair.lg_window
        p_arrival = self.snlg_onepair.p_arrival
        origin_time = self.snlg_onepair.origin_time
        logger.info(f"Success in beginning formal Sn/Lg computation for event ({self.snlg_onepair.origin_time})")

        # Get traces for Sn and Lg analyses (use zerophase filter)
        # Use copy to avoid reduplicative filtering
        try:
            t_comp_waveform_for_sn = trace.copy().filter(
                "bandpass",
                freqmin=self.snlg_onepair.sn_filt[0],
                freqmax=self.snlg_onepair.sn_filt[1],
                corners=4,
                zerophase=self.enable_zero_phase_filter
            )
        except Exception as e:
            logger.error(f"Error in bandpass filter for Sn: {e}")
            raise
        try:
            t_comp_waveform_for_lg = trace.copy().filter(
                "bandpass",
                freqmin=self.snlg_onepair.lg_filt[0],
                freqmax=self.snlg_onepair.lg_filt[1],
                corners=4,
                zerophase=self.enable_zero_phase_filter
            )
        except Exception as e:
            logger.error(f"Error in bandpass filter for Lg: {e}")
            raise
        if self.enable_trim_edge:
            # Post-processing trimming: trim possible high amplitude signals at the edge resulting the previous processing
            # Be careful with the use of tr.stats.starttime and tr.stats.endtime
            t_comp_waveform_for_sn.trim(t_comp_waveform_for_sn.stats.starttime + self.edge_trim_length/2, 
                                        t_comp_waveform_for_sn.stats.endtime   - self.edge_trim_length/2)
            t_comp_waveform_for_lg.trim(t_comp_waveform_for_lg.stats.starttime + self.edge_trim_length/2, 
                                        t_comp_waveform_for_lg.stats.endtime   - self.edge_trim_length/2)
        
        self.snlg_onepair.t_comp_waveform_for_sn = t_comp_waveform_for_sn.copy() # Use copy to avoid any further processing.
        self.snlg_onepair.t_comp_waveform_for_lg = t_comp_waveform_for_lg.copy() # Use copy to avoid any further processing.
        logger.info(f"Success in obtaining Sn and Lg traces for event ({self.snlg_onepair.origin_time})")
        
        # Define noise window relative to the p_arrival
        noise_start = p_arrival - self.noise_windowlen - self.noise_offset
        noise_end   = p_arrival - self.noise_offset
        self.snlg_onepair.noise_window = (noise_start,noise_end)
        try:
            # Compute RMS for noise slices using np.std
            # Use copy to avoid further processing on original data
            noise_sn = self.snlg_onepair.t_comp_waveform_for_sn.copy().trim(
                starttime=noise_start,
                endtime=noise_end
            )

            noise_lg = self.snlg_onepair.t_comp_waveform_for_lg.copy().trim(
                starttime=noise_start,
                endtime=noise_end
            )

            # Compute RMS for signal slices for Sn and Lg
            # Note that: Sn and Lg windows are defined relative to origin time, as shown in Section S1 of Song & Klemperer (2024)
            signal_sn = self.snlg_onepair.t_comp_waveform_for_sn.copy().trim(
                starttime = origin_time + sn_start,
                endtime   = origin_time + sn_end
            )

            signal_lg = self.snlg_onepair.t_comp_waveform_for_lg.copy().trim(
                starttime = origin_time + lg_start,
                endtime   = origin_time + lg_end
            )

            ###### Choose a method to calculate RMS amplitude ######
            # The differences between Method 1 and Method 2 should be negligible if data have near-zero mean
            # Method 1
            # self.snlg_onepair.amp_noise_sn = np.std(noise_sn.data)
            # self.snlg_onepair.amp_noise_lg = np.std(noise_lg.data)
            # self.snlg_onepair.amp_sn = np.std(signal_sn.data)
            # self.snlg_onepair.amp_lg = np.std(signal_lg.data)
            # Method 2
            self.snlg_onepair.amp_noise_sn = np.sqrt(np.mean(noise_sn.data ** 2))
            self.snlg_onepair.amp_noise_lg = np.sqrt(np.mean(noise_lg.data ** 2))
            self.snlg_onepair.amp_sn = np.sqrt(np.mean(signal_sn.data ** 2))
            self.snlg_onepair.amp_lg = np.sqrt(np.mean(signal_lg.data ** 2))
            logger.info(f"Success in calculating RMS amplitude of noise, Sn, and Lg.")
        except Exception as e:
            logger.error(f"Error in trimming Sn {self.snlg_onepair.sn_window} and Lg {self.snlg_onepair.lg_window}: {e}")
            return
        
        # Check invalid values for amp_sn and amp_lg
        if (self.snlg_onepair.amp_sn is None or self.snlg_onepair.amp_lg is None or
            not np.isfinite(self.snlg_onepair.amp_sn) or not np.isfinite(self.snlg_onepair.amp_lg) or
            self.snlg_onepair.amp_lg == 0):
            logger.warning(f"[Invalid Value] Sn {self.snlg_onepair.amp_sn} and Lg {self.snlg_onepair.amp_lg}. Sn/Lg calculation terminates.")
            return
        
        # Compute the natural logarithm of the Sn/Lg amplitude ratio
        self.snlg_onepair.SnRLg_raw = np.log(self.snlg_onepair.amp_sn / self.snlg_onepair.amp_lg)
        self.snlg_onepair.SnRLg_raw_ratioed = np.log(
            (self.snlg_onepair.amp_sn / np.max(np.abs(self.snlg_onepair.t_comp_waveform_for_sn.data))) /
            (self.snlg_onepair.amp_lg / np.max(np.abs(self.snlg_onepair.t_comp_waveform_for_lg.data)))
        )

        # Other Sn, Lg Measurements
        # Maximum Amplitude - Is it nessary to apply a normalization by maximum value of the whole trace?
        self.snlg_onepair.max_sn = np.max(np.abs(signal_sn.data)) / np.max(np.abs(self.snlg_onepair.t_comp_waveform_for_sn.data))
        self.snlg_onepair.max_lg = np.max(np.abs(signal_lg.data)) / np.max(np.abs(self.snlg_onepair.t_comp_waveform_for_lg.data))
        self.snlg_onepair.max_SnRLg_raw = self.snlg_onepair.max_sn / self.snlg_onepair.max_lg

        # Calculate the signal-to-noise ratios for Sn and Lg and take the maximum
        sn_snr = self.snlg_onepair.amp_sn / self.snlg_onepair.amp_noise_sn
        lg_snr = self.snlg_onepair.amp_lg / self.snlg_onepair.amp_noise_lg
        self.snlg_onepair.SNR = max(sn_snr, lg_snr)
        self.snlg_onepair.SNR_Sn = sn_snr
        self.snlg_onepair.SNR_Lg = lg_snr

        # Estimate the error in the Sn/Lg ratio using error propagation
        # Equation S4.1 - S4.3 in Song & Klemperer (2024), after some simplification
        self.snlg_onepair.SnRLg_err = np.sqrt(
            (self.snlg_onepair.amp_noise_sn / self.snlg_onepair.amp_sn) ** 2 +
            (self.snlg_onepair.amp_noise_lg / self.snlg_onepair.amp_lg) ** 2
        )

        # Apply the geometric spreading correction to the Sn/Lg ratio
        # DEPRECATED
        self.snlg_onepair.SnRLg_cor = (
            self.snlg_onepair.SnRLg_raw +
            self.geometric_spreading_SnRLg()
        )
        # # NEW
        # K_Sn, K_Lg = self.geometric_spreading_SnRLg()
        # amp_sn_cor = self.snlg_onepair.amp_sn * K_Sn
        # amp_lg_cor = self.snlg_onepair.amp_lg * K_Lg
        # self.snlg_onepair.SnRLg_cor = np.log(amp_sn_cor / amp_lg_cor)
        
    def geometric_spreading_SnRLg(self):
        """
        Compute the geometric spreading correction for the Sn/Lg ratio.
        
        Parameters:
            gcarc: The great circle arc distance (in degrees).
        
        Returns:
            The geometric spreading correction factor for natural log .
        """
        # DEPRECATED 
        # Ref: S5. Geometric Spreading detrend in Song & Klemperer (2024)
        dist  = self.snlg_onepair.dist

        K_Lg = (dist - 500) * 0.00025
        K_Sn = max(0, (dist - 800)) * (-0.0006)

        return (K_Sn - K_Lg) * np.log(10)

        # # NEW: Following the logic from the notebook
        # dist = self.snlg_onepair.dist
        # K_Lg = 10**((dist-500)*0.00025)
        # if dist > 800:
        #     K_Sn = -10**((dist-800)*0.0006)
        # else:
        #     K_Sn = 1
        # return K_Sn, K_Lg

