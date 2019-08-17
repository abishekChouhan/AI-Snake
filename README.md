# AI-Snake
Reinforcement Learning Snake Game

![alt text](https://github.com/abishekChouhan/AI-Snake/blob/master/images/snake-input.png)

### Pre-Requirements:
* python3
* Google Chrome

### Install Python Dependencies
```bash
pip3 install -r requirements.txt
```

### Setup on Windows System:
* Download suitable chrome driver from: https://chromedriver.chromium.org/downloads
* Extract the .zip file. Copy the chromedriver.exe in the cloned directory folder (AI-snake/chromedriver.exe)
* Install Pillow(PIL): 
  ```bash
  pip3 install Pillow==2.2.1
  ```
* Open ``Deep_Reinforcement_Learning_Snake.py``, Comment out line 9 and 101 and un-comment line 8 and 100

### Setup on Linux System:
* Install chrome driver:
  ```bash
  sudo apt-get install chromium-chromedriver
  ```
* Install Pyscreenshot:
  ```bash
  sudo pip3 install pyscreenshot
  ```
* Open ``Deep_Reinforcement_Learning_Snake.py``, Comment out line 8 and 100 and un-comment line 9 and 101

### CNN Network Architecture
![alt text](https://github.com/abishekChouhan/AI-Snake/blob/master/images/network.png)

### Configuration:
* You can open ``Deep_Reinforcement_Learning_Snake.py`` and change the following parameters:
  ```
  img_rows, img_cols = 64,64
  SPEED = 360
  WIDTH = 8
  HEIGHT = 8
  ACTIONS = 4 
  GAMMA = 0.99 
  OBSERVATION = 65.
  EXPLORE = 70
  FINAL_EPSILON = 0.1
  INITIAL_EPSILON = 1
  REPLAY_MEMORY = 50000
  BATCH = 64
  FRAME_PER_ACTION = 1
  LEARNING_RATE = 1e-4
  img_channels = 4
  ```

### Start Training:
```bash
python3 Deep_Reinforcement_Learning_Snake.py
```
### Checking the scores:
* File ```data/scores_df.csv``` contains scores of each game played by our model.
* File ```data/actions_df.csv``` contains action at each step taken by the model.
* File ```data/loss_df.csv``` contains loss at each step taken by the model.
* File ```data/time_df.csv``` contains seconds played of each game.

### Reference:
https://github.com/yenchenlin/DeepLearningFlappyBird

https://medium.com/acing-ai/how-i-build-an-ai-to-play-dino-run-e37f37bdf153
