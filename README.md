[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

In this project, DQN agent is trained to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]


A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

### Installation and Requirments:

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the main folder, and unzip (or decompress) the file. 

### Challenge: Learning from Pixels

After you have successfully completed the project, if you're looking for an additional challenge, you have come to the right place!  In the project, your agent learned from information such as its velocity, along with ray-based perception of objects around its forward direction.  A more challenging task would be to learn directly from pixels!

To solve this harder task, you'll need to download a new Unity environment.  This environment is almost identical to the project environment, where the only difference is that the state is an 84 x 84 RGB image, corresponding to the agent's first-person view.  (**Note**: Udacity students should not submit a project with this new environment.)

You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/VisualBanana_Windows_x86_64.zip)

Then, place the file in the main folder in the DRLND GitHub repository, and unzip (or decompress) the file.  Next, open `Navigation_Pixels.ipynb` and follow the instructions to learn how to use the Python API to control the agent.

### Navigation Architecture

Through episodes, agent interacts with the environment by taking actions and observing the changes coming back from the environment in form of reward, next_state and done variables. Rewards get accumulated per episode to teach the agent what action to take in specific states for maximum reward. For learning from rewards and penalties of its actions, agent uses 2 separate deep neural networks with exact same architectures, local and target networks, as well as a replay memory for using the experiences to improve its performance.
 * Replay buffer or Memory: keeps a buffer of states, actions, next_states and rewards for the target network
 * Target network: this network is initialized randomly at first and gets updated after memory is large enough. It picks batches of random samples from memory and tunes the weights and biases based on the experiences to acheive best action for a given state
 * Local network: takes a state as input and outputs an action, the goal is to minimize the error from Target network

The goal is to get an average score of +13 over 100 consecutive episodes.

### Model Architecture:
 * 1- DQN agent is initialized with 
   * 2 FeedForward Networks as QN_local and QN_target
   * replay buffer 
   * starting epsilon 1.0 (e-greedy)
 * 2- agent observes its current state from the environment and calls act module 
   * it can randomly take an action or pass the state throught the QN_local network
   * at the beginning of the training agent has more chance to explore the environment becuase epsilon is closer to 1.0
   * but as epsilon gets smaller throught decaying process, there would be less chance for exploring and it'd use its experiences that gained through the QN_local network 
   * each episode decreases the epsilon value for making the process greedy until a minimum valu is reached
 * 3- action is passed to the environment and reward and the next state of the environment are returned
 * 4- agent sends the observations to step() module 
   * state, action, reward, next_state and done are added to replay buffer for the QN_target
   * if memory size has reached a threshold, agent collects a batch of random sample for learning for minimizing QN_local error and updating QN_target network
 * 5- here is the agent's learning process for the QN_target and minimizing QN_local error
    * sends samples to QN_target for target actions
    * sends samples to the QN_local for expected actions
    * minimizes the expected actions error from the target actions using an optimizer
    * updates QN_target
 * 6- agent continues episodes until it reaches the max episode
 * QNework consists of 4 feed forward layers:
 `  * fc1 : in:state_size, out:64
    * relu: activation for adding nonlinearity
    * fc2: in:64, out:512
    * relu: activation for adding nonlinearity
    * fc3: in:512, out:64
    * relu: activation for adding nonlinearity
    * fc4: in: 64, out: action_size
    * **Note**:I tried other combinations with different batch_sizes and learning_rates but this resulted in higher scores
    `
### Model Architecture (Pixel Challenge):

#### Experiment 1 

For this experiment, model is trained on Tesla k80 instance but cuda ran out of memory after 1129 episodes with avergae score of 6.15, which I beleive it could go higher if I could resolve gpu memory problem like adding elastic gpus or do parallel computing in pytorch.

Model Architecture:

    * For each iteration, 3 input image frames are stacked and resized to 32x32x3, color channels are kept to improve color detection. Grayscaling images would lose a lot of infromation from them image frames as yellow belnds in floor and background.
    * input_shape: 1x3x3x32x32 (used deque for stacking frames)
    * learning_rate = 0.001 , batch_size = 512
    * conv3d layer1: in_channels=3, out_channels=10, kernel_size=(1,5,5), stride=1
    * Relu layer
    * Maxpool3d: kernel_size=(1,2,2)
    * size calculation : 32x32x3 -> 28x28x10 -> 14x14x10
    * conv3d layer2: in_channels=10, out_channels=32, kernel_size=(1,5,5) , stride=1
    * Relu layer
    * Maxpool3d: kernel_size=(1,2,2)
    * size calculation : 14x14x10 -> 10x10x32 -> 5x5x32
    * fully connected layer: action_size
    
Result:

    * Episode: 100	Average Score: 0.01
    * Episode: 200	Average Score: 0.132
    * Episode: 300	Average Score: 0.90
    * Episode: 400	Average Score: 1.07
    * Episode: 500	Average Score: 1.83
    * Episode: 600	Average Score: 2.34
    * Episode: 700	Average Score: 3.25
    * Episode: 800	Average Score: 4.59
    * Episode: 900	Average Score: 5.35
    * Episode: 1000	Average Score: 5.63
    * Episode: 1100	Average Score: 6.32
    * Episode: 1129	Average Score: 6.15
    
Exception: RuntimeError: CUDA error: out of memory


