from ..datasets import _BaseDataset
import pandas as pd
import torch
from typing import Union, Tuple
import numpy as np
import os

def log_transform(x):
    return np.log(x + 1)


class RNADataset(_BaseDataset):
    """
    Simple dataset for RNA data.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        preprocessor: "sklearn.pipeline.Pipeline",
        return_mask: bool = False,
        base_path: str = None
    ) -> None:
        super().__init__(dataframe, return_mask, base_path)
        self.preprocessor = preprocessor

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, bool], torch.Tensor]:
        sample = self.dataframe.iloc[idx]

        if not pd.isna(sample.RNA):
            file_path = None
            if self.base_path is not None:
                file_path = os.path.join(
                    self.base_path, 
                    sample.RNA.split(os.sep)[-2], #patient
                    sample.RNA.split(os.sep)[-1] #file
                )
            else:
                file_path = sample.RNA
                
            out = torch.from_numpy(
                self.preprocessor.transform(
                    pd.read_csv(file_path)["fpkm_uq_unstranded"].values.reshape(1, -1)
                )
            ).float()
            mask = True
        else:
            out = torch.zeros(1, 16304).float()
            mask = False

        if self.return_mask:
            return out, mask
        else:
            return out
