import torch
import numpy as np
import random
from model import CupHeadNet
from collections import deque
import time



# Agent class

class CupHead(object):
    def __init__(
        self,
        state_dim, 
        action_dim, 
        logging = False,
        save_dir='trash',
        exploration_rate_decay=0.99999975,
        exploration_rate_min=0.1,
        batch_size=32,
        gamma=0.9,
        burnin=1e2,
        learn_every=3,
        learning_rate = 0.001,
        sync_every=1e2,
        device = "cuda",
        learn_during_episode = False,
        ):

        ## act()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        self.logging = logging

        # if device == "cuda":
        #     self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #     use_cuda = torch.cuda.is_available()
        #     print(f"Using CUDA: {use_cuda}")
        #     print()
       
        self.device = device

        # CupHead's DNN to predict the most optimal action
        self.net = CupHeadNet(self.state_dim, self.action_dim).float()
        self.net = self.net.to(device=self.device)

        self.exploration_rate = 1
        self.exploration_rate_decay = exploration_rate_decay
        self.exploration_rate_min = exploration_rate_min
        self.curr_step = 0
        self.curr_ep = 0

        self.save_every = 100  # no. of experiences between saving Cuphead Net

        ## cache() and recall()

        self.memory = deque(maxlen=10000)    # mémoire occupée : 100 000 * 80 bytes = 7,629394531 MBytes < 16 Mbytes
        self.batch_size = batch_size

        ## td_estimate and td_target()

        self.gamma = gamma

        ## update_Q_online() and sync_Q_online

        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.loss_fn = torch.nn.SmoothL1Loss()

        ## learn()

        self.learn_during_episode = learn_during_episode
        self.burnin = burnin  # min. steps before training
        self.learn_every = learn_every  # no. of steps between updates to Q_online
        self.sync_every = sync_every  # no. of steps between Q_target & Q_online sync



    def act(self, state):
        """
    Given a state, choose an epsilon-greedy action and update value of step.

    Inputs:
    state : A single observation of the current state, dimension is (state_dim)
    Outputs:
    action_idx (int): An integer representing which action CupHead will perform
    """
        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)

        # EXPLOIT
        else:
            state = state[0].__array__() if isinstance(state, tuple) else state.__array__()
            state = torch.tensor(state, device=self.device).unsqueeze(0)
            # state = torch.tensor(state, device="cpu").unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx
    
    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        # def first_if_tuple(x):
        #     return x[0] if isinstance(x, tuple) else x
        # state = first_if_tuple(state).__array__()
        # next_state = first_if_tuple(next_state).__array__()

        # state = torch.tensor(state, device=self.device)
        # next_state = torch.tensor(next_state, device=self.device)
        # action = torch.tensor([action], device=self.device)
        # reward = torch.tensor([reward], device=self.device)
        # done = torch.tensor([done], device=self.device)

        # start = time.time()

        state = torch.tensor(state, device="cpu")
        next_state = torch.tensor(next_state, device="cpu")
        action = torch.tensor([action], device="cpu")
        reward = torch.tensor([reward], device="cpu")
        done = torch.tensor([done], device="cpu")

        self.memory.append((state, next_state, action, reward, done,))

        # print(time.time()-start)

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
    
    def td_estimate(self, state, action):
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()
    
    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def save(self):
        if self.logging:
            save_path = (
                self.save_dir / f"cuphead_net_{int(self.curr_step // self.save_every)}.chkpt"
            )
            torch.save(
                dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
                save_path,
            )
            print(f"CupHeadNet saved to {save_path} at step {self.curr_step}")
    
    def learn(self):
        if self.learn_during_episode:
            if self.curr_step % self.sync_every == 0:
                self.sync_Q_target()

            # if self.curr_step % self.save_every == 0:
            #     self.save()

            if self.curr_step < self.burnin:
                return None, None

            if self.curr_step % self.learn_every != 0:
                return None, None
        else:
            if self.curr_ep % self.sync_every == 0:
                self.sync_Q_target()

            # if self.curr_ep % self.save_every == 0:
            #     self.save()
            # if self.curr_ep < self.burnin:
            #     return None, None
            # if self.curr_ep % self.learn_every != 0:
            #     return None, None

        # Sample from memory
        

        state, next_state, action, reward, done = self.recall()
        if self.device != "cpu" :
            # print(f"Transfer : memory to {self.device}")
            state=state.to(device=self.device)
            next_state=next_state.to(device=self.device)
            action=action.to(device=self.device)
            reward=reward.to(device=self.device)
            done=done.to(device=self.device)

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)