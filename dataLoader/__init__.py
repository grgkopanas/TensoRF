from .llff import LLFFDataset
from .blender import BlenderDataset
from .nsvf import NSVF
from .rtmv import RTMVDataset
from .tankstemple import TanksTempleDataset
from .your_own_data import YourOwnDataset

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'tankstemple': TanksTempleDataset,
                'nsvf': NSVF,
                'rtmv': RTMVDataset,
                'own_data': YourOwnDataset}
