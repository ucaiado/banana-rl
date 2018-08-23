Banana RL
===========================

In this project, I will train an agent to navigate (and collect bananas!) in a vast, square world. I am going to use the [Unity Machine Learning Agents Toolkit](https://github.com/Unity-Technologies/ml-agents) to design, train, and evaluate my own deep reinforcement learning algorithms implementations.

The environment used for this project is the Udacity version of the Banana Collector environment, from [Unity](https://youtu.be/heVMs3t9qSk). The goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

<p align="center"><img src="https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif" alt="Example game of isolation" width="50%" style="middle"></p>

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. The state space has 37 dimensions and contains the agent’s velocity, along with a ray-based perception of objects around agent’s forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

- 0 - move forward.
- 1 - move backward.
- 2 - turn left.
- 3 - turn right.

The task is episodic, and to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes. This project is part of the [Deep Reinforcement Learning Nanodegree](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwigwuKwr4LdAhUMI5AKHTuBCz0QFjAAegQIDBAB&url=https%3A%2F%2Fwww.udacity.com%2Fcourse%2Fdeep-reinforcement-learning-nanodegree--nd893&usg=AOvVaw3OfEe4LlR9h_4vW3TZpE_o) program, from Udacity. You can check my report [here](reports/Report.pdf).

### Install
This project requires **Python 3.5** or higher, the Banana Collector Environment (follow the instructions to download [here](drlnd/README.md)) and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Torch](https://pytorch.org)
- [UnityAgents](https://github.com/Unity-Technologies/ml-agents)
- [OpenAI Gym](https://gym.openai.com)


### Run
In a terminal or command window, navigate to the top-level project directory `banana-rl/` (that contains this README) and run the following command:

```shell
$ python -m aind.foo
```


### References
1. Schaul, T., Quan, J., Antonoglou, I., & Silver, D. *Prioritized Experience Replay*. arXiv.org, 2015.
2. Van Hasselt, H., Guez, A., & AAAI, D. S. *Deep Reinforcement Learning with Double Q-Learning*. Aaai.org, 2016.
3. Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., et al. *Human-level control through deep reinforcement learning*. Nature, 52015.


### License
The contents of this repository are covered under the [MIT License](LICENSE).
