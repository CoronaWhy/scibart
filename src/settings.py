"""global settings"""

import torch


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
'Device type'

PERC_VALIDATION_SET = 0.1
'Percentage of training set to set away for model validation'