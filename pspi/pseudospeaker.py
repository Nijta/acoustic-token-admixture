"""Pseudospeaker generation related classes and functions."""
import glob
import os.path
import pathlib

import numpy as np
import random
from kaldiio import ReadHelper
import joblib
import json
import math
import sys
import logging
from typing import Any, List, Optional, Set, Union, Dict
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

CRITERION_CLUSTER_DENSE = 'cluster_dense'
CRITERION_CLUSTER_SPARSE = 'cluster_sparse'
GENDER_MALE = 'm'
GENDER_FEMALE = 'f'
GENDERS = [GENDER_FEMALE, GENDER_MALE]
DENSITY_BASED_CRITERIA = [CRITERION_CLUSTER_SPARSE, CRITERION_CLUSTER_DENSE]


class Pool:
    """Class to hold data related to anonymization pool.

    It stores data needed for selecting the PseudoSpeaker.
    Information related to speakers, their gender, xvectors,
    clustering information pre-computed offline, and the pitch
    sequences extracted using a pitch extraction module.

    Attributes
    -----------
    speakers: List[str]
            an ordered list of speakers in the pool
    xvectors: Dict[str, np.array]
            a dictionary mapping speakers to their xvectors
            averaged over all their utterances
    genders: Dict[str, str]
            a dictionary mapping speakers to their gender
    clustering_info: Dict[str, list]
            a dictionary containing three keys:
                'density_rank': rank of cluster centers in the increasing
                                order of density
                'gender': dominant gender of each cluster
                'labels': cluster identity label for each xvector
    pitch: Dict[str, np.array]
            a dictionary mapping speakers to all the pitch values extracted from their utterances
    """

    XVECTOR_DIR = 'xvectors'
    XVECTOR_SCP = 'spk_xvector.scp'
    XVECTOR_ARK = 'spk_xvector.ark'
    SPEAKER_TO_GENDER = 'spk2gender'
    CLUSTER_DIR = 'cluster'
    CLUSTER_CENTRE_INDEX = 'cc_idx.pkl'
    CLUSTER_LABELS = 'labels.pkl'
    CLUSTER_DENSITY_RANK = 'drank.pkl'

    def __init__(self,
                 speakers: List[str],
                 xvectors: Dict[str, np.array],
                 genders: Dict[str, str],
                 clustering_info: Dict[str, list],
                 pitch: Dict[str, np.array],
                 is_old_pitch: bool = False):
        """Initialize the anonymization pool."""
        self.speakers = speakers
        self.xvectors = xvectors
        self.genders = genders
        self.clustering_info = clustering_info
        self.pitch = pitch
        self.is_old_pitch = is_old_pitch

    def get_xvectors(self, speaker_ids: List['str']) -> np.array:
        """Return the xvectors of a list of speakers in a numpy array.

        Parameters
        -----------
        speaker_ids: List[str]
            List of the speakers ids to consider

        Returns
        --------
        np.array
            xvectors of the considered speakers in a numpy array respecting the same order
            of speakers than the speakers' id list
        """
        xvector_dimension = self.xvectors[speaker_ids[0]].shape[0]
        xvectors_matrix = np.zeros((len(speaker_ids), xvector_dimension), dtype='float64')
        for i, spk_id in enumerate(speaker_ids):
            xvectors_matrix[i, :] = self.xvectors[spk_id]
        return xvectors_matrix

    def get_speakers_of_gender(self, gender: Literal[GENDERS]) -> List[str]:
        """Return all the speakers of the prescribed gender.

        Parameters
        -----------
        gender: ['f', 'm']
            Requested gender

        Returns
        ---------
        List['str]
            The ids of all the speakers of this gender, in a list

        """
        return [spk_id for spk_id, spk_gender in self.genders.items() if spk_gender == gender]

    def choose_one_cluster_of_speakers(self,
                                       gender: Literal[GENDERS],
                                       criterion: Literal[DENSITY_BASED_CRITERIA],
                                       top_n: int) -> int:
        """Return a list of speakers from this pool based on gender and density criterion.

        Parameters
        ------------
        gender: ['f', 'm']
            Requested gender
        criterion: ['cluster_dense', 'cluster_sparse']
            Whether to choose the cluster among the densest or sparsest ones
        top_n: int
            Depending on the criterion, the returned cluster will be randomly chosen among the
            `top_n` densest or sparsest ones

        Returns
        --------
        int
            Returns the label of the selected cluster

        """
        # Filter the clusters by gender
        clusters = [c for c in self.clustering_info['density_rank']
                    if self.clustering_info['gender'][c] == gender]
        # Select the n more dense/sparse according to the criterion
        if criterion == CRITERION_CLUSTER_SPARSE:
            clusters = clusters[:top_n]
        elif criterion == CRITERION_CLUSTER_DENSE:
            clusters = clusters[-top_n:]
        else:
            raise ValueError(f"Criterion must be one among {DENSITY_BASED_CRITERIA}, "
                             f"got {criterion}.")
        # Choose one of them
        return random.choice(clusters)

    def get_speakers_from_cluster(self, cluster_id: int) -> List[str]:
        """Return the list of all the speakers in a cluster.

        Parameters
        ----------
        cluster_id: int
            Label of the considered cluster

        Returns
        -------
        List['str']
            the list of the ids of all the speakers forming this cluster
        """
        return [spk_id for spk_id, spk_cluster in
                zip(self.speakers, self.clustering_info['labels'])
                if spk_cluster == cluster_id]

    def get_pitch_statistics_of_speakers(self, speaker_ids: List[str]) -> Dict[str, float]:
        """Return statistics from the pitch of a given list of speakers in this pool.

        Parameters
        ----------
        speaker_ids: List[str]
            List of the speakers ids to consider

        Returns
        -------
        Dict[str, float]
            the mean and standard deviation of the log of all the non-zeros values extracted from
            the speakers pitch
        """
        if self.is_old_pitch:
            all_values = np.concatenate([self.pitch[spk_id] for spk_id in speaker_ids])
            log_non_zeros_values = np.log(all_values[all_values != 0])
            return {'mean': np.mean(log_non_zeros_values), 'std': np.std(log_non_zeros_values)}
        else:
            spk_stats = [self.pitch[spk_id] for spk_id in speaker_ids]
            total_weight = sum(W for M, S, W in spk_stats)
            total_mean = sum(W * M for M, S, W in spk_stats)/ total_weight
            total_variance = sum(W * (S ** 2 + (M - total_mean) ** 2) for M, S, W in spk_stats) / total_weight
            total_std = math.sqrt(total_variance)
            return {'mean': total_mean, 'std': total_std}

    def get_pseudospeaker(self, selected_speakers_ids: Set[str]):
        """Build a Pseudospeaker object from a list of speakers from this pool.

        Parameters
        ----------
        speaker_ids: List[str]
            Ids of the speakers in this pool to merge into a pseudospeaker

        Returns
        -------
        PseudoSpeaker
            The resulting pseudo-speaker
        """
        genders = [self.genders[s] for s in selected_speakers_ids]
        if genders.count(GENDER_FEMALE) >= genders.count(GENDER_MALE):
            gender = GENDER_FEMALE
        else:
            gender = GENDER_MALE
        pseudo_speaker_x_vector = np.mean(self.get_xvectors(selected_speakers_ids), axis=0)
        pitch_statistics = self.get_pitch_statistics_of_speakers(selected_speakers_ids)
        return PseudoSpeaker(gender, pseudo_speaker_x_vector,
                             pitch_statistics, selected_speakers_ids)

    @classmethod
    def load(cls, data_path: str, pitch_dir='yaapt_pitch', pitch_dtype: str = '<f4'):
        """Load pool data from a data directory.

        Static method to load pool data from a directory containing
        the following files:
        1. xvectors
           (data_path/xvectors/{spk_xvector.scp, spk_xvector.ark})
        2. speaker-to-gender mapping
           (data_path/spk2gender)
        3. Density clustering information obtained using
           Affinity Propagation algorithm
           (data_path/cluster/{cc_idx.pkl, drank.pkl, labels.pkl})
        4. Pitch sequences for speakers
           (data_path/yaapt_pitch/<pitch_for_each_speaker_id.f0>)
        """
        logging.debug("Reading pool spk2gender.")
        speakers, spk2gender = cls._load_spk2gender(data_path)

        logging.debug("Reading pool xvectors.")
        xvectors = cls._load_xvectors(data_path)

        logging.debug("Reading clustering info.")
        cluster_dict = cls._load_clustering_info(data_path, speakers, spk2gender)

        logging.debug("Reading speakers pitch.")
        pitch, is_old_pitch = cls._load_pitch(os.path.join(data_path, pitch_dir), pitch_dtype)

        return Pool(speakers, xvectors, spk2gender, cluster_dict, pitch)

    @classmethod
    def _load_spk2gender(cls, data_path: str):
        """Load speaker ids and gender information from spk2gender file."""
        # TODO: Since Python 3.7 dict are sorted in the order of insertion
        # So no need for a separate speakers list
        speakers = []  # to keep track of order of speakers
        spk2gender = {}
        with open(os.path.join(data_path, cls.SPEAKER_TO_GENDER)) as f:
            for line in f.read().splitlines():
                speaker_id, speaker_gender = line.split()
                speakers.append(speaker_id)
                spk2gender[speaker_id] = speaker_gender
        return speakers, spk2gender

    @classmethod
    def _load_xvectors(cls, data_path: str):
        """Load xvectors from spk_xvector.scp file which is in Kaldi format."""
        xvectors = {}
        dimensions = set()
        with ReadHelper('ark:' + os.path.join(data_path,
                                              cls.XVECTOR_DIR, cls.XVECTOR_ARK)) as reader:
            for key, xvector in reader:
                xvectors[key] = xvector
                dimensions.add(xvector.shape[0])
        logging.debug(f"Read {len(xvectors)} pool xvectors")
        if len(dimensions) > 1:
            raise ValueError(f"Not all xvectors of same dimension: {dimensions}")
        logging.debug(f"Found x-vector dimensions to be: {dimensions}")
        return xvectors

    @classmethod
    def _load_clustering_info(cls, data_path: str,
                              speakers: List[str],
                              spk2gender: Dict[str, str]):
        """Load clustering information from pre-computed pickle files."""
        path = os.path.join(data_path, cls.CLUSTER_DIR)
        centers = joblib.load(os.path.join(path, cls.CLUSTER_CENTRE_INDEX))
        labels = joblib.load(os.path.join(path, cls.CLUSTER_LABELS))
        density_rank = joblib.load(os.path.join(path, cls.CLUSTER_DENSITY_RANK))
        # Separate male and female cluster centroids and keep track whether
        # they are used or not
        cc_gender = [spk2gender[speakers[x]] for x in centers]
        logging.debug(f"Male clusters: {cc_gender.count(GENDER_MALE)}")
        logging.debug(f"Female clusters: {cc_gender.count(GENDER_FEMALE)}")
        return {
            'density_rank': density_rank,
            'gender': cc_gender,
            'labels': labels
        }

    @staticmethod
    def _load_pitch(data_path, pitch_dtype) -> Dict[str, Dict]:
        if os.path.exists(os.path.join(data_path, 'pitch.json')):
            with open(os.path.join(data_path, 'pitch.json')) as f:
                pitch_dict = json.load(f)
            return pitch_dict, False
        else:
            pitch = {}
            for f0_file in glob.glob(f'{data_path}/*.f0'):
                spk_id = pathlib.Path(f0_file).stem
                with open(f0_file) as f:
                    pitch[spk_id] = np.fromfile(f, dtype=pitch_dtype)
            return pitch, True


class PseudoSpeaker:
    """An imaginary sample in xvector speaker space.

    Attributes
    -----------
    gender: str
        gender assigned to the pseudospeaker
        according to the design choices
    xvector: np.array
        vector containing speaker information for the new target identity
    pitch_stats: dict
        pitch statistics (mean and standard deviation) used for pitch conversion
    selected_speakers: list (optional)
        List on the ids of the speakers which have been selected to generate this pseudospeaker
    """

    def __init__(self, gender: str, xvector: np.array, pitch_stats: Dict[str, float],
                 selected_speakers: Optional[List[str]] = None):
        """Initialize a pseudospeaker."""
        self.gender = gender
        self.xvector = xvector
        self.pitch_stats = pitch_stats
        self.selected_speakers = selected_speakers

    def __eq__(self, other):
        """Check if another PseudoSpeaker is equal to this one."""
        if isinstance(other, PseudoSpeaker):
            return self.gender == other.gender and self.pitch_stats == other.pitch_stats \
                and (self.xvector == other.xvector).all()
        return False

    def convert_pitch(self, src_f0: np.array, flatten=False):
        """Convert a pitch series, so it matches the pitch statistics of this pseudospeaker.

        Normalize the non zeros values of the given pitch so the mean and standard
        deviation of the resulting pitch match the ones of this pseudospeaker.
        Zero values are preserved.

        Parameters
        ----------
        src_f0: np.array
            Original pitch series to convert
        flatten: boolean
            Whether the new pitch should be flatten (so the result sounds robotic)
            or respect the original one (and be more natural). Deafult: false

        Returns
        --------
        np.array
            The results of the conversion, of the same shape as the input

        """
        target_mean, target_std = self.pitch_stats['mean'], self.pitch_stats['std']

        new_f0 = np.zeros(src_f0.shape[0])
        src_f0_non_zeros = np.log(src_f0[src_f0 != 0])
        if flatten:
            new_f0[src_f0.nonzero()] = np.exp(target_mean)
        else:
            # Compute mean and standard deviation from the original pitch
            src_mean, src_std = np.mean(src_f0_non_zeros), np.std(src_f0_non_zeros)

            # Normalize the non zero values
            new_f0_non_zeros = ((src_f0_non_zeros - src_mean) / src_std) * target_std + target_mean

            # Create a new series with them + the zeros from the original pitch
            new_f0[src_f0.nonzero()] = np.exp(new_f0_non_zeros)
        return new_f0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Build a Pseudospeaker object from a dictionary."""
        return cls(data["gender"],
                   np.array(data["xvector"], dtype=data["xvector_dtype"]),
                   data["pitch_stats"], data["selected_speakers"])

    def to_dict(self):
        """Build a dictionary representation of the pseudospeaker object."""
        speakers_id_string = ':'.join(self.selected_speakers) if self.selected_speakers else ''
        return {
            "gender": self.gender,
            "xvector": self.xvector.tolist(),
            "xvector_dtype": self.xvector.dtype.name,
            "pitch_stats": self.pitch_stats,
            "selected_speakers": speakers_id_string
        }


def generate_pseudospeaker(pool: Pool, n_speakers: Union[int, float] = 0.5,
                           gender: Literal[GENDERS] = None,
                           criterion: Literal[DENSITY_BASED_CRITERIA] = None,
                           top_n_clusters: int = 10, seed: Optional[int] = None) -> PseudoSpeaker:
    """Generate a pseudospeaker based on the specified design choices.

    A flexible pseudo-speaker selection technique based on https://hal.inria.fr/hal-02610447v2.
    Design choices include:
    - the target gender
    - the possibility to filter a cluster: the criterion for the region of x-vector
    space where the speakers

    Finally, `n_speakers` are selected to be averaged in a pseudospeaker

    Parameters
    ----------
    pool: Pool
        A pool of data from where speakers are picked
    n_speakers: int/float
        proportion (or number) of speakers to select among the candidate speakers to create a
        pseudospeaker (default: 0.5)
    gender: str
        target gender for pseudospeaker among m/f, if None is provided random
        gender is selected among m/f (default: None)
    criterion: cluster_sparse/cluster_dense
        Optional criterion to select the speakers from a cluster among the most sparse/dense ones.
        If None is provided then clustering information is not used and speakers are selected
        at random from the full list of speakers of the given gender (default: None)
    top_n_clusters: int
        if criterion is among cluster_sparse/cluster_dense, then the cluster will be
        randomly picked among the `top_n_clusters` most dense/sparse clusters (default: 10).
    seed: int
        integer seed for pseudo-random number generator (default: None)

    Returns
    -------
    A PseudoSpeaker object
    """
    # Check validity of parameters
    if criterion and criterion not in DENSITY_BASED_CRITERIA:
        raise ValueError(f"Criterion must be one among {DENSITY_BASED_CRITERIA}, got {criterion}.")

    if gender and gender not in GENDERS:
        raise ValueError(f"Gender must be one among {GENDERS}, got {gender}.")

    if seed is not None:
        random.seed(seed)

    gender = gender if gender else random.choice(GENDERS)
    if criterion:
        # Select candidate speakers according to density and gender criteria
        cluster_to_use = pool.choose_one_cluster_of_speakers(gender, criterion, top_n_clusters)
        candidate_speakers = pool.get_speakers_from_cluster(cluster_to_use)
        n_candidate_speakers = len(candidate_speakers)
        logging.debug(f"Using cluster {cluster_to_use} which has {n_candidate_speakers} members.")
    else:
        # Select candidate speakers according to gender criterion
        candidate_speakers = pool.get_speakers_of_gender(gender)
        n_candidate_speakers = len(candidate_speakers)

    # Select the requested number of speakers from this shortlist
    if n_speakers < 1:
        n_speakers = max(int(n_candidate_speakers * n_speakers), 1) #to avoid n_speaker == 0
    else:
        n_speakers = min(n_candidate_speakers, n_speakers)
    selected_speakers_ids = random.sample(candidate_speakers, n_speakers)

    return pool.get_pseudospeaker(selected_speakers_ids)
