# Rarity of Events
Code for the Rarity of Events method that rewards the agent based on the temporal rarity of events.
Pre-trained models are found in the models directory.

# Packages to install
pytorch
scipy
sdl2
vizdoom

## Training the agent
~~~~
# A2C baseline
python main.py --num-processes 16 --config-path scenario/deathmatch.cfg --num-frames 75000000 --no-vis

# A2C+RoE
python main.py --num-processes 16 --config-path scenario/deathmatch.cfg --num-frames 75000000 --no-vis --roe
~~~~

## Running the agent
~~~~
# A2C baseline
python enjoy.py --config-path scenario/deatmatch.cfg

# A2C+RoE
python enjoy.py --config-path scenario/deatmatch.cfg --roe
~~~~
