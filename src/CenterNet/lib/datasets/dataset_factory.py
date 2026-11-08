
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ctdet import CTDetDataset
from .dataset.forensic import Forensic
from .dataset.forensic_micro import ForensicMicro
from .dataset.forensic_liquid import ForensicLiquid
from .dataset.forensic_box import ForensicBox
from .dataset.forensic_paper import ForensicPaper
from .dataset.forensic_3 import Forensic3
from .dataset.forensic_3_v2 import Forensic3V2
from .dataset.forensic_3_v3 import Forensic3V3
from .dataset.forensic_3_v4 import Forensic3V4

dataset_factory = {
  'forensic_liquid': ForensicLiquid,
  'forensic': Forensic,
  'forensic_micro': ForensicMicro,
  'forensic_box': ForensicBox,
  'forensic_paper': ForensicPaper,
  'forensic_3': Forensic3,
  'forensic_3_v2': Forensic3V2,
  'forensic_3_v3': Forensic3V3,
  'forensic_3_v4': Forensic3V4,
}

_sample_factory = {
  'ctdet': CTDetDataset
}

def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
