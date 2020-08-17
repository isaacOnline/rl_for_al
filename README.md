## Contents
The *agents* directory contains agents written by John or I. The change-point directory adds 
change point agents to your gym registry. The experiments directory contains code for using 
the agents on the environments contained in the change-point library. The base directory contains 
utilities that are referenced by more than one of the other directories.  

## Agents
There are three value iteration agents in this directory, as well as one MCTS agent.
The MCTS agent is `uniform_mcts_agent.py`. The value iteration agents are `uniform_agent.py`, `non_uniform_agent.py`,
and `recharging_agent.py`. The `value_iteratory.py` file contains the base class for the three
value iterators. You can see how the agents are used in the experiments directory. The gist is that you feed them 
environment paramaters, then they calculate a policy, which other code can apply to a gym environment.

## Change Point Environments
The change-point directory is a package that registers uniform, non uniform, and recharging
environments with Open AI Gym. The environments are based on a change_point base class, which does most of the heavy
lifting.

### Installing Environments
 After `gym` has been installed, the change-point package can be 
installed with `pip install -e change-point`. Instances of uniform, non-uniform, and recharging
environments can then be created with 
`gym.make('change_point:uniform-v0')`, `gym.make(change_point:non_uniform-v0)`, and 
`gym.make(change_point:non_uniform-v0)`, respectively.

## Experiments
The highest level code is contained in this directory, in the vi_vs_sb folder. There are three folders in this 
directory, one for each environment type. The scripts to call are called `uniform_comparison.py`, 
`non_uniform_comparison.py`, and `recharging_comparison.py`. Each one contains a ModelRunner, which calculates (or loads 
from disc) a value iteration policy, as well as trains a stable baselines model and extracts a policy from it. It then
scores both policies on the relevant environment, logs the results, prints them, and saves a visualization. I'd 
recommend stepping through `uniform_comparison.py` to get acquainted with this repo, as it is uses the simplest agent
from the agents directory along with the simplest environment.
