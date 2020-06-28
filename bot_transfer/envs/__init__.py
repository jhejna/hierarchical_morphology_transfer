# Add The wrappers
from .hierarchical import LowLevelEnv as Low
from .hierarchical import HighLevelEnv as High
from .hierarchical import JointAC1Env as JointAC1
from .hierarchical import JointAC2Env as JointAC2
from .hierarchical import JointOPEnv as JointOP
from .hierarchical import DiscriminatorEnv as Discriminator
from .hierarchical import FullEnv as Full
from .hierarchical import LowFinetuneEnv as LowFinetune

# Add the environments
from .point_mass import *
from .maze import * # has extra functions, may need to change.
from .ant import *
from .humanoid import *
from .swimmer import *
from .quadruped import *
from .reacher import *
from .insert import *
