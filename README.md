# DQN (Q-learning) for CartPole reinforcement task

### Installation and run
```
$ git clone https://github.com/tocehka/DQN-CartPole
$ pip install -r requirements.txt
$ python main.py
```

### Realization
DQN approach with 1 Neural Networks: Quality estimator with 2 layers
The complexity of this realization - correct quality processing per current state for forming the train batches for NN

### Results
![Reward graph](/images/graph.png)

We can view a convergence as early as 1500 epochs

At the same time the model is sufficiently trained and can hold a stick