# Q-networks for binary vector actions
Matlab code and data used in Deep RL Workshop @ NIPS 2015

```
Yoshida, Naoto. "Q-networks for binary vector actions." arXiv preprint arXiv:1512.01332 (2015).
```

RL-glue for Matlab is assumed to run.
```
Tanner, Brian, and Adam White. "RL-Glue: Language-independent software for reinforcement-learning experiments." The Journal of Machine Learning Research 10 (2009): 2133-2136.
```

# The result of the grid world with population coding task
In this experiment, the action is represented by a 40-bit binary vector. And the moves of the agent are driven according to the type of population coding. Because the discrete action space exponentially grows according to the length of the binary action vector, the size of the corresponding action space is huge |A| = 2^40 > 10^12
. Therefore, efficient sampling of the action is also required in this domain.

![Mar-17-2022 12-19-05](https://user-images.githubusercontent.com/1684732/158730226-c18b46b7-cc27-4434-9855-0c3e4f704e39.gif)


# Workshop
https://rll.berkeley.edu/deeprlworkshop/
