"""Pitch related classes and functions."""

import amfm_decompy.pYAAPT as YAAPT   # noqa: N817
import amfm_decompy.basic_tools as basic
import numpy as np
from typing import Union, BinaryIO
import scipy.io.wavfile
import logging
import sys


logger = logging.getLogger(__name__)

import numpy as np
import scipy.io.wavfile
import pyworld as pw
from typing import Union, BinaryIO

def extract(
    wav_data: Union[BinaryIO, str],
    min_f0: int = 60,
    max_f0: int = 600,
    frame_length: int = 25,
    frame_shift: int = 10,
) -> np.array:
    """Extract pitch using pyworld.

    Parameters
    ----------
    wav_data : BinaryIO or str
        BinaryIO A buffered binary stream
        str path to wav file
    min_f0 : int
        Minimum pitch searched (default: 60 Hz)
    max_f0 : int
        Maximum pitch searched (default: 600 Hz)
    frame_length : int
        Length of each analysis frame in ms (default: 25 ms)
    frame_shift : int
        Spacing between analysis frames in ms (default: 10 ms)
    
    Returns
    -------
    numpy array
        Single-precision floating point number array
    """
    fs, data = scipy.io.wavfile.read(wav_data)
    data = data.astype(np.float64)  # Ensure compatibility with pyworld
    
    # Convert ms to sample frames
    frame_period = frame_shift  # pyworld expects frame_period in ms
    
    # Extract fundamental frequency (f0)
    f0, _time = pw.dio(data, fs, f0_floor=min_f0, f0_ceil=max_f0, frame_period=frame_period)
    f0 = pw.stonemask(data, f0, _time, fs)  # Refine f0 estimation
    
    return np.asarray(f0, dtype=np.dtype("<f4", 1))

def extractOLD(
    wav_data: Union[BinaryIO, str],
    min_f0: int = 60,
    max_f0: int = 600,
    frame_length: int = 25,
    frame_shift: int = 10,
) -> np.array:
    """Extract pitch using pYaapt.

    Parameters
    ----------
    wav_data : BinaryIO or StrPath
        BinaryIO A buffered binary stream
        StrPath to wav file
    min_f0 : int
        minimum pitch searched (default: 60 Hz)
    max_f0 : int
        maximum pitch searched (default: 600 Hz)
    frame_length : int
        length of each analysis frame (default: 25 ms)
    frame_shift : int
        spacing between analysis frames (default: 10 ms)
    Returns
    -------
    numpy array
        single-precision floating point number array
        if process successful
    """
    logger.debug(wav_data)
    fs, data = scipy.io.wavfile.read(wav_data)
    signal = basic.SignalObj(data, fs)
    logger.debug(signal)
    pitch = YAAPT.yaapt(signal,
                        f0_min=min_f0,
                        f0_max=max_f0,
                        frame_length=frame_length,
                        frame_space=frame_shift)
    # select little-endian as datatype
    return np.asarray(pitch.samp_values,
                      dtype=np.dtype(("<f4", 1)))
