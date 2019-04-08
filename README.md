# Rarity of Events
Code for the Rarity of Events method that rewards the agent based on the temporal rarity of events.
Pre-trained models are found in the models directory.

## Videos
[![A2C RoE in VizDoom](https://img.youtube.com/vi/v5NkHVuV8gs/0.jpg)](https://www.youtube.com/watch?v=YG-lf732a0U&list=PL3-IRrahTCWLSDCYij20BDn-uKdVGOiu9 "A2C+RoE in VizDoom")

## Packages to install
* pytorch
* scipy
* sdl2
* vizdoom
* pickle

## Training the agent
~~~~
# A2C baseline
python main.py --num-processes 16 --config-path scenarios/deathmatch.cfg --num-frames 75000000 --no-vis

# A2C+RoE
python main.py --num-processes 16 --config-path scenarios/deathmatch.cfg --num-frames 75000000 --no-vis --roe

# A2C+RoE+QD
python main.py --num-processes 16 --config-path scenarios/deathmatch.cfg --num-frames 75000000 --no-vis --roe --qd --agent-id 1
python main.py --num-processes 16 --config-path scenarios/deathmatch.cfg --num-frames 75000000 --no-vis --roe --qd --agent-id 2
~~~~

## Running the agent
~~~~
# A2C baseline
python enjoy.py --config-path scenario/deatmatch.cfg

# A2C+RoE
python enjoy.py --config-path scenario/deatmatch.cfg --roe
~~~~

## Acknowledgements
This repository is based on https://github.com/openai/baselines and https://github.com/p-kar/a2c-acktr-vizdoom.
