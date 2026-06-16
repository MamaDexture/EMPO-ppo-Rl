
import numpy as np
from collections import deque


class MemoryBuffer:

    def __init__(self, max_size=5000):
        self.memory = deque(maxlen=max_size)

    def add_experience(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        episode_reward=0
    ):

        experience = {
            "state": np.array(state),
            "action": int(action),
            "reward": float(reward),
            "next_state": np.array(next_state),
            "done": bool(done),
            "episode_reward": float(episode_reward)
        }

        self.memory.append(experience)

    def size(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()

    def get_all(self):
        return list(self.memory)

    def sample(self, batch_size=32):

        if len(self.memory) == 0:
            return []

        batch_size = min(batch_size, len(self.memory))

        idx = np.random.choice(
            len(self.memory),
            batch_size,
            replace=False
        )

        return [self.memory[i] for i in idx]

    def get_top_experiences(self, top_k=50):

        if len(self.memory) == 0:
            return []

        sorted_memory = sorted(
            self.memory,
            key=lambda x: x["episode_reward"],
            reverse=True
        )

        return sorted_memory[:top_k]

    def save(self, filepath):

        np.save(
            filepath,
            list(self.memory),
            allow_pickle=True
        )

    def load(self, filepath):

        loaded = np.load(
            filepath,
            allow_pickle=True
        )

        self.memory = deque(
            loaded.tolist(),
            maxlen=self.memory.maxlen
        )
