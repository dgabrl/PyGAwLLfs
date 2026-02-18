"""
Variable Configuration Module.

This module manages the mapping between dataset names and their respective
feature names. It is essential for XAI (eXplainable AI) metrics and for
mapping chromosome bits back to human-readable variables.
"""

from typing import List, Dict


def get_variables_names(dataset_name: str) -> List[str]:
    """
    Retrieves the list of feature names for a given dataset.

    This function acts as a central repository for feature metadata.
    It ensures that each bit in the Genetic Algorithm's chromosome
    can be mapped back to a specific variable.

    Args:
        dataset_name (str): The identifier of the dataset (e.g., 'boson', 'covidx').

    Returns:
        List[str]: A list containing the feature names in the order they
            appear in the dataset. Returns an empty list if the dataset
            name is not found.
    """

    configs: Dict[str, List[str]] = {
        'boson': [
            'lepton_pT', 'lepton_eta', 'lepton_phi', 'missing_energy_magnitude',
            'missing_energy_phi', 'jet1pt', 'jet1eta', 'jet1phi', 'jet1b-tag',
            'jet2pt', 'jet2eta', 'jet2phi', 'jet2b-tag', 'jet3pt', 'jet3eta',
            'jet3phi', 'jet3b-tag', 'jet4pt', 'jet4eta', 'jet4phi', 'jet4b-tag',
            'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb'
        ],
        'covidx': [
            "fo_10Percentile", "fo_90Percentile", "fo_Energy", "fo_Entropy",
            "fo_IQR", "fo_Kurtosis", "fo_Max", "fo_MAD", "fo_Mean", "fo_Median",
            "fo_Min", "fo_Range", "fo_RMAD", "fo_RMS", "fo_Skewness",
            "fo_TotalEnergy", "fo_Uniformity", "fo_Variance", "glcm_Autocorr",
            "glcm_ClusProm", "glcm_ClusShade", "glcm_ClusTend", "glcm_Contrast",
            "glcm_Corr", "glcm_DiffAvg", "glcm_DiffEnt", "glcm_DiffVar",
            "glcm_Id", "glcm_Idm", "glcm_Idmn", "glcm_Idn", "glcm_Imc1",
            "glcm_Imc2", "glcm_InvVar", "glcm_JointAvg", "glcm_JointEnergy",
            "glcm_JointEnt", "glcm_MCC", "glcm_MaxProb", "glcm_SumAvg",
            "glcm_SumEnt", "glcm_SumSq", "gldm_DepEnt", "gldm_DepNU",
            "gldm_DepNUN", "gldm_DepVar", "gldm_GLNU", "gldm_GLVar",
            "gldm_HGLGEmp", "gldm_LDEmp", "gldm_LDEHGEmp", "gldm_LDELGEmp",
            "gldm_LGLGEmp", "gldm_SDEmp", "gldm_SDEHGEmp", "gldm_SDELGEmp",
            "glrlm_GLNU", "glrlm_GLNUN", "glrlm_GLVar", "glrlm_HGLREmp",
            "glrlm_LREmp", "glrlm_LRHGLEmp", "glrlm_LRLGLEmp", "glrlm_LGLREmp",
            "glrlm_RunEnt", "glrlm_RunNU", "glrlm_RunNUN", "glrlm_RunPerc",
            "glrlm_RunVar", "glrlm_SREmp", "glrlm_SRHGLEmp", "glrlm_SRLGLEmp",
            "glszm_GLNU", "glszm_GLNUN", "glszm_GLVar", "glszm_HGLZEmp",
            "glszm_LAEmp", "glszm_LAHGLEmp", "glszm_LALGLEmp", "glszm_LGLZEmp",
            "glszm_SZNU", "glszm_SZNUN", "glszm_SAEmp", "glszm_SAHGLEmp",
            "glszm_SALGLEmp", "glszm_ZoneEnt", "glszm_ZonePerc", "glszm_ZoneVar",
            "ngtdm_Busyness", "ngtdm_Coarseness", "ngtdm_Complexity",
            "ngtdm_Contrast", "ngtdm_Strength"
        ]
    }

    return configs.get(dataset_name, []).copy()