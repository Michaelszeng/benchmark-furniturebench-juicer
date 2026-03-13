First, let's define terminology. Let's consider a finite state machine (FSM), which has states we'll call "FSM states". Let's then call the state of the environment (i.e. robot EEF pose, object poses, etc.) the "environment state".

I want to make the scripted policy used in `src/data_collection/scripted.py` a Markovian policy; at every point, I want the action to be determined from the current environment state.

## Outline of the Current Implementation

As you can see now, `src/data_collection/scripted.py` creates a `DataCollector`, which calls `env.get_assembly_action()` in a loop. `get_assembly_action()` is implemented in `furniture-bench/furniture_bench/envs/furniture_sim_env.py`. `furniture-bench/furniture_bench/envs/furniture_sim_env.py` works by iterating through a `should_be_assembled` list one index at a time; this list contains tuples of pairs of parts that should be assembled together. For two given parts, the scripted policy then first call the `pre_assemble()` procedures on `part1`, then `part2`, then calls `part2`'s `fsm_step()` procedure. All of the `pre_assemble()` and `fsm_step()` procedures manage separate internal FSMs; these FSMs also step through a sequence of states one at a time.

As you may notice, there are multiple non-Markovian elements of the current implementation. Firstly, the determination of which assembly step we are on in `should_be_assembled` is non-Markovian, since this is stepped through one index at a time. Secondly, there are some hysteresis/counters to allow the simulator to stabilize before transitioning through FSM states. There may also be other non-Markovian elements of the current scripted policy that I haven't covered here.


## The Task

The first step to making this policy non-Markovian is to adapt each of the state machines to determine their FSM state through the environment state alone rather than maintaining a fixed schedule of FSM states. This should occur at the per-part `fsm_step()` and `pre_assemble()` level, as well as on the level of `env.get_assembly_action()`'s FSM.

The second step is to remove all instance of hysteresis and counters and replace this with environment state-based checks.


## Implementation Notes

Since you are a coding agent and aren't able to view the simulator GUI, you should add sufficient print debugging so that you can still get the gist of what happened in the simulation without actually viewing it.