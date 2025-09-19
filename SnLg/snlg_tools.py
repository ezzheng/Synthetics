######## A list of useful tools ########
# group_by_event: group the snlg pairs by event
# group_by_station: group the snlg pairs by network and station name
# select_pairs_match_evst_from_filelist: match pairs by event time and network/station name (input: list of pkl files)
# select_pairs_match_evst: match pairs by event time and network/station name (input: list of pairs)
# select_pairs_match_evtime: match pairs with close event_time
# merge_pkl_list: merge multiple pkl list
# read_one_pkl: read one pkl file
# export_snlg_to_excel: export pkl to csv file


from collections import defaultdict
from obspy import UTCDateTime
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pickle
from pathlib import Path

@dataclass
class MetaEntry:
    origin_time: Optional[UTCDateTime]
    kstnm: Optional[str]
    knetwk: Optional[str]
    file_path: str
    index_in_file: int

def generate_meta_file(file_paths, output_meta_file):
    """
    Generate a metadata file indexing event-station pairs for faster access.

    Parameters:
        file_paths (str): list of .pkl files locations.
        output_meta_file (str): Path to output metadata index pickle file.

    Returns:
        List[MetaEntry]: List of metadata entries extracted.
    """
    meta_list = []

    print(f"[START] Generating metadata from files...")

    for file_path in file_paths:
        try:
            with open(file_path, 'rb') as f:
                data_list = pickle.load(f)

            for idx, obj in enumerate(data_list):
                entry = MetaEntry(
                    origin_time=getattr(obj, 'origin_time', None),
                    kstnm=getattr(obj, 'kstnm', None),
                    knetwk=getattr(obj, 'knetwk', None),
                    file_path=file_path,
                    index_in_file=idx
                )
                meta_list.append(entry)

        except Exception as e:
            print(f"[ERROR] Failed to process {file_path}: {e}")

    # Save to output file
    if output_meta_file:
        with open(output_meta_file, "wb") as f:
            pickle.dump(meta_list, f)
            print(f"[OUTPUT] Saved {len(meta_list)} entries to {output_meta_file}.")

    print(f"[DONE] metadata generated")

    return meta_list

def group_by_event(pairs):
    """
    Groups pairs by origin_time.
    Returns a dict: { origin_time: [pair, pair, …], … }
    """
    dd = defaultdict(list)
    for p in pairs:
        if p.SnRLg_raw is None:
            continue
        # key = p.origin_time.isoformat()  # e.g. "2025-04-15T12:34:56.789000"
        key = p.origin_time.strftime("%Y%m%d_%H%M%S") # e.g. "20050415_123456"
        dd[key].append(p)
    return dd

def group_by_station(pairs):
    """
    Groups pairs by knetwk.kstnm (network name and station name).
    Returns a dict: { knwtwk.kstnm: [pair, pair, …], … }
    """
    dd = defaultdict(list)
    for p in pairs:
        if p.SnRLg_raw is None:
            continue
        # Use network code + station code to prevent misgrouping cases with the same station code but different network code
        dd[f"{p.knetwk}.{p.kstnm}"].append(p)
    return dd

def select_pairs_match_evst_from_filelist(filelist, evtime, net_code=None, stnm=None, max_diff=0.1):
    """
    Select the event-station pair(s) by event time and network/station name from a list of files.

    Parameters:
        filelist (list): List of file paths for pickled data.
        evtime (UTCDateTime): Event time to match.
        net_code (str): Network code to filter.
        stnm (str, optional): Station name to filter.
        max_diff (float): Maximum allowed time difference in seconds.

    Returns:
        list: Matching pair(s), or None if not found.
    """
    for file in filelist:
        try:
            pairs = read_one_pkl(file)
            if not isinstance(pairs, list):
                raise ValueError(f"Expected a list in {file!r}, got {type(pairs).__name__}") 
            print(f"Successfully read file: {file}")

            pair_selected = select_pairs_match_evst(
                pairs=pairs, evtime=evtime, net_code=net_code, stnm=stnm, max_diff=max_diff
            )

            if pair_selected:  # Found a match, return immediately
                return pair_selected

        except Exception as e:
            print(f"Error processing file {file}: {e}")

    print("No matching pair found.")
    return None


def select_pairs_match_evst(pairs, evtime, net_code=None, stnm=None, max_diff=0.1):
    """
    Select the event-station pair(s) by event time and network/station name from a list of SnLg_OneEventStationPair.
    """

    candidate_pairs = select_pairs_match_evtime(pairs=pairs, evtime=evtime, max_diff=max_diff, return_all=True)
    pair_return = []
    for pair in candidate_pairs:
        if pair.kstnm == stnm:
            if net_code:
                if pair.knetwk == net_code:
                    pair_return.append(pair)
            else:
                pair_return.append(pair)
    if not pair_return:
        print(f"No match for network code {net_code} and station code {stnm}")
    else:
        print(f"{len(pair_return)} match for network code {net_code} and station code {stnm}")
    return pair_return


def select_pairs_match_evtime(pairs, evtime, max_diff = 1.0, return_all = False):
    """
    Select 
    (1) the event-station pair whose origin_time is closest to a given event time.
    or (2) a list of event-station pairs whose origin_time are within max_diff to a given event time

    Parameters:
        pairs (list): List of SnLg_OneEventStationPair dataclass.
        evtime (UTCDateTime): Reference event time to match.
        max_diff (float, optional): Maximum allowed difference in seconds. Defaults to 1.0.
        return_all (bool, optional): If True, returns all pairs within max_diff; 
                                     otherwise returns the single best match.

    Returns:
        If return_all is False:
            The single pair with the smallest time difference to evtime.
        Else:
            List of all pairs with time difference <= max_diff.
    """
    if not isinstance(evtime, UTCDateTime):
        raise TypeError(f"Expected 'evtime' to be UTCDateTime, got {type(evtime)}")

    # Filter out pairs with valid origin_time and compute differences
    diffs = [
        (idx, abs(p.origin_time - evtime)) 
        for idx, p in enumerate(pairs) 
        if p.origin_time is not None
    ]

    if not diffs:
        raise ValueError("No valid 'origin_time' found in any pair.")

    # Sort by time difference
    diffs.sort(key=lambda x: x[1])

    if return_all:
        close_idxs = [idx for idx, diff in diffs if diff <= max_diff]
        if not close_idxs:
            raise ValueError(f"No pairs found within {max_diff} seconds of {evtime}.")
        print(f"Return all match for {evtime} within {max_diff} s difference: {len(close_idxs)} matches")
        return [pairs[i] for i in close_idxs]

    # Return the closest match
    best_idx, best_diff = diffs[0]
    if best_diff > max_diff:
        raise ValueError(
            f"Closest match exceeds max_diff: {best_diff:.2f}s > {max_diff}s. "
            f"Origin time: {pairs[best_idx].origin_time}"
        )

    print(f"Return the closest match for {evtime}")
    return pairs[best_idx]


def merge_pkl_list(file_paths, output_path=None, drop_waveforms=True):
    """
    Merge multiple pickled lists into a single list.
    """
    merged = []
    for fp in file_paths:
        data = read_one_pkl(fp)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list in {fp!r}, got {type(data).__name__}") 
        if drop_waveforms:
            [pair.__setattr__('zne_comp_waveform_raw', None) or
                pair.__setattr__('t_comp_waveform', None)
                for pair in data]
        merged.extend(data)
        print(f"{fp} done.")

    if output_path:
        try:
            # ensure parent directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump(merged, f)  
        except Exception as e:
            raise RuntimeError(f"Failed to write merged pickle to {output_path!r}: {e}")

    return merged

def read_one_pkl(file_path):
    """
    Load and return the contents of a single pickle file.
    """
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)  
    except Exception as e:
        raise RuntimeError(f"Error reading {file_path}: {e}")

def export_snlg_to_excel(pairs, export_fields, path):
    # Convert to dicts and filter
    rows = []
    count = 0
    for inst in pairs:
        d = asdict(inst)
        row = { k: d.get(k) for k in export_fields }
        rows.append(row)
        count += 1
        if count % 100 == 0:
            print(count)

    df = pd.DataFrame(rows, columns=export_fields)
    df.to_excel(path, index=False)
    print(f"Done: wrote {len(df)} records → {path}")