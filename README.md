

# Video-Pre-Training
Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos


> :page_facing_up: [Read Paper](https://cdn.openai.com/vpt/Paper.pdf) \
  :mega: [Blog Post](https://openai.com/blog/vpt) \
  :space_invader: [MineRL Environment](https://github.com/minerllabs/minerl) (note version 1.0+ required) \
  :checkered_flag: [MineRL BASALT Competition](https://www.aicrowd.com/challenges/neurips-2022-minerl-basalt-competition)

# Training Inverse Dynamics Model (IDM)

Designed to run using 16 frames of data rather than 128 and being a lot... dumber, but its goal is to be a lightweight... approximation rather than high def labels
but, its also a test of whats required as when limiting the original IDM to 16 frames of data its guesses weren't terrible.

Known Limitations: does not apply virtual cursor to videos.

Original code by ViktorThink https://github.com/ViktorThink/Video-Pre-Training
Tweaks by Corianas

# Running Inverse Dynamics Model (IDM)

IDM aims to predict what actions player is taking in a video recording.

Setup:
* Install requirements: `pip install -r requirements.txt`
* Download the IDM model [.model :arrow_down:](https://openaipublic.blob.core.windows.net/minecraft-rl/idm/4x_idm.model) and [.weight :arrow_down:](https://openaipublic.blob.core.windows.net/minecraft-rl/idm/4x_idm.weights) files
* For demonstration purposes, you can use the contractor recordings shared below to. For this demo we use
  [this .mp4](https://openaipublic.blob.core.windows.net/minecraft-rl/data/10.0/cheeky-cornflower-setter-02e496ce4abb-20220421-092639.mp4)
  and [this associated actions file (.jsonl)](https://openaipublic.blob.core.windows.net/minecraft-rl/data/10.0/cheeky-cornflower-setter-02e496ce4abb-20220421-092639.jsonl).

To run the model with above files placed in the root directory of this code:
```
python run_inverse_dynamics_model.py -weights 4x_idm.weights --model 4x_idm.model --video-path cheeky-cornflower-setter-02e496ce4abb-20220421-092639.mp4 --jsonl-path cheeky-cornflower-setter-02e496ce4abb-20220421-092639.jsonl
```

A window should pop up which shows the video frame-by-frame, showing the predicted and true (recorded) actions side-by-side on the left.

Note that `run_inverse_dynamics_model.py` is designed to be a demo of the IDM, not code to put it into practice.

# Contribution
This was a large effort by a dedicated team at OpenAI:
[Bowen Baker](https://github.com/bowenbaker),
[Ilge Akkaya](https://github.com/ilge),
[Peter Zhokhov](https://github.com/pzhokhov),
[Joost Huizinga](https://github.com/JoostHuizinga),
[Jie Tang](https://github.com/jietang),
[Adrien Ecoffet](https://github.com/AdrienLE),
[Brandon Houghton](https://github.com/brandonhoughton),
[Raul Sampedro](https://github.com/samraul),
Jeff Clune
The code here represents a minimal version of our model code which was
prepared by [Anssi Kanervisto](https://github.com/miffyli) and others so that these models could be used as
part of the MineRL BASALT competition.
