# FermiNet: A Deep Dive Into the Code, for compilers sake

> "FermiNet is a neural network for learning the ground state wavefunctions of atoms and molecules using a variational Monte Carlo approach."

FermiNet is the implementation of the (2020) paper ["Ab-Initio Solution of the Many-Electron Schroedinger Equation with Deep Neural Networks"](https://arxiv.org/pdf/1909.02487.pdf)

It is a neural network for learning the ground state [wavefunctions](https://phys.libretexts.org/Bookshelves/University_Physics/Book%3A_University_Physics_(OpenStax)/Book%3A_University_Physics_III_-_Optics_and_Modern_Physics_(OpenStax)/07%3A_Quantum_Mechanics/7.02%3A_Wavefunctions) of atoms and molecules plus [variational Monte Carlo](https://pubs.acs.org/doi/10.1021/acs.jctc.0c00147).

<img src="wave.gif" width="600" height="400" /> 

## [Setup.py](https://github.com/deepmind/ferminet/blob/jax/setup.py)
FermiNet's setup file has some key dependencies that give us a peek into what is going on: 
* [Kfac](https://github.com/deepmind/deepmind-research/tree/master/kfac_ferminet_alpha) specific to fermiNet which is different than [TensorFlow Kfac](https://github.com/tensorflow/kfac) -- [Kfac Paper](https://arxiv.org/pdf/1503.05671.pdf) :blue_book: 
* [Jax](https://github.com/google/jax)
* [ML Collections](https://github.com/google/ml_collections)
* [Optax](https://github.com/deepmind/optax)
* [PyScf](https://github.com/pyscf/pyscf/)


## [Base Config.py](https://github.com/deepmind/ferminet/blob/jax/ferminet/base_config.py)

The base_config.py file lets the user set the system and hyperparameters.

#### enum

The code begins: 
```python
import enum
import ml_collections
from ml_collections import config_dict
```

Simmilar to [enums in to C++ 11](https://en.cppreference.com/w/cpp/language/enum):
```c++
// color may be red (value 0), yellow (value 1), green (value 20), or blue (value 21)
enum color
{
    red,
    yellow,
    green = 20,
    blue
};
```

[Python enums](https://docs.python.org/3/library/enum.html) introduced in  >= python 3.4 are a set of symbolic names (members) bound to unique, constant values:

```python
class NBA_ORG_RANKING(Enum):
  CELTICS = 7
  JAZZ = 4
  KNICKS = 31
  NETS = 1
```

FermiNet instantiates classes using enum. The code continues:

```python
class SystemType(enum.IntEnum):
  MOLECULE = 0
```
[IntEnum](https://docs.python.org/3/library/enum.html#enum.IntEnum) creates enumerated constants that are also subclasses of the [int](https://docs.python.org/3/library/functions.html#int) class.

"Members of an IntEnum can be compared to integers; by extension, integer enumerations of different types can also be compared to each other...However, they can’t be compared to standard Enum enumerations:

```python
from enum import Enum
from enum import IntEnum

class Knicks(IntEnum):
  RANDLE = 1
  BARRETT = 2
  NTILIKINA = 3

class Nets(Enum):
  DURANT = 1
  IRVING = 2
  HARDEN = 3

print(Knicks.RANDLE == Nets.DURANT)
>> False
```
#### @classmethod decorator

The code continues: 
```python
class SystemType(enum.IntEnum):
  MOLECULE = 0

  @classmethod
  def has_value(cls, value):
    return any(value is item or value == item.value for item in cls)
```

The [@classmethod](https://stackoverflow.com/questions/12179271/meaning-of-classmethod-and-staticmethod-for-beginner) decorator:

> "...take a cls parameter that points to the class—and not the object instance—when the method is called." [source](https://realpython.com/instance-class-and-static-methods-demystified/)

> "Because the class method only has access to this cls argument, it can’t modify object instance state. That would require access to self. However, class methods can still modify class state that applies across all instances of the class." [source](https://realpython.com/instance-class-and-static-methods-demystified/)

What we have here is the [factorymethod](https://en.wikipedia.org/wiki/Factory_method_pattern#:~:text=From%20Wikipedia%2C%20the%20free%20encyclopedia,object%20that%20will%20be%20created.) design pattern.

![chaplin](factory.gif)

```python
class Nets:
  def __init__(self, skills):
    self.skills = skills

  def __repr__(self):
    return f'Nets({self.skills!r})'

  @classmethod
  def Joe_Harris(cls):
    return cls(['3Pts', 'drives'])

  @classmethod
  def Nic_Claxton(cls):
    return cls(['D', 'jams'])

print(Nets.Nic_Claxton())
print(Nets.Joe_Harris())

>> Nets(['D', 'jams'])
>> Nets(['3Pts', 'drives'])
```

In this example we can create new Players of the Nets class configured with the skills that we want them to have from a single __init__ but many constructors. Now the class (cls) is the first argument rather than the instance of the class (self). 

> "Another way to look at this use of class methods is that they allow you to define alternative constructors for your classes. Python only allows one __init__ method per class. Using class methods it’s possible to add as many alternative constructors as necessary." [source](https://realpython.com/instance-class-and-static-methods-demystified/)

### ml_collections ConfigDict

```python
def default() -> ml_collections.ConfigDict:
  """Create set of default parameters for running qmc.py.
  Note: placeholders (cfg.system.molecule and cfg.system.electrons) must be
  replaced with appropriate values.
  Returns:
    ml_collections.ConfigDict containing default settings.
  """
 ```
 
 Default returns a [mlcollections.ConfigDict]()
 
 > ConfigDict...is a "dict-like" data structure with dot access to nested elements...Supposed to be used as a main way of expressing configurations of experiments and models.


## Installation

`pip install -e .` will install all required dependencies. This is best done
inside a [virtual environment](https://docs.python-guide.org/dev/virtualenvs/).

```
virtualenv -p python3.7 ~/venv/ferminet
source ~/venv/ferminet/bin/activate
pip install -e .
```

If you have a GPU available (highly recommended for fast training), then use
`pip install -e '.[tensorflow-gpu]'` to install TensorFlow with GPU support.

We use python 3.7 (or earlier) because there is TensorFlow 1.15 wheel available
for it. TensorFlow 2 is not currently supported.

The tests are easiest run using pytest:

```
pip install pytest
python -m pytest
```

## Usage

```
ferminet --batch_size 1024 --pretrain_iterations 100
```

will train FermiNet to find the ground-state wavefunction of the LiH molecule
with a bond-length of 1.63999 angstroms using a batch size of 1024 MCMC
configurations ("walkers" in variational Monte Carlo language), and 100
iterations of pretraining (the default of 1000 is overkill for such a small
system). The system and hyperparameters can be controlled by flags. Run

```
ferminet --help
```

to see the available options. Several systems used in the FermiNet paper are
included by default. Other systems can easily be set up, by setting the
appropriate system flags to `ferminet`, modifying `ferminet.utils.system` or
writing a custom training script. For example, to run on the H2 molecule:

```
import sys

from absl import logging
from ferminet.utils import system
from ferminet import train

# Optional, for also printing training progress to STDOUT
logging.get_absl_handler().python_handler.stream = sys.stdout
logging.set_verbosity(logging.INFO)

# Define H2 molecule
molecule = [system.Atom('H', (0, 0, -1)), system.Atom('H', (0, 0, 1))]

train.train(
  molecule=molecule,
  spins=(1, 1),
  batch_size=256,
  pretrain_config=train.PretrainConfig(iterations=100),
  logging_config=train.LoggingConfig(result_path='H2'),
)
```

`train.train` is controlled by a several lightweight config objects. Only
non-default settings need to be explicitly supplied. Please see the docstrings
for `train.train` and associated `*Config` classes for details.

Note: to train on larger atoms and molecules with large batch sizes, multi-GPU
parallelisation is essential. This is supported via TensorFlow's
[MirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy)
and the `--multi_gpu` flag.

## Output

The results directory contains `pretrain_stats.csv`, which contains the
pretraining loss for each iteration, `train_stats.csv` which contains the local
energy and MCMC acceptance probability for each iteration, and the `checkpoints`
directory, which contains the checkpoints generated during training. If
requested, there is also an HDF5 file, `data.h5`, which contains the walker
configuration, per-walker local energies and per-walker wavefunction values for
each iteration. Warning: this quickly becomes very large!

## Giving Credit

If you use this code in your work, please cite the associated paper:

```
@article{ferminet,
  title={Ab-Initio Solution of the Many-Electron Schr{\"o}dinger Equation with Deep Neural Networks},
  author={D. Pfau and J.S. Spencer and A.G. de G. Matthews and W.M.C. Foulkes},
  journal={Phys. Rev. Research},
  year={2020},
  volume={2},
  issue = {3},
  pages={033429},
  doi = {10.1103/PhysRevResearch.2.033429},
  url = {https://link.aps.org/doi/10.1103/PhysRevResearch.2.033429}
}
```

