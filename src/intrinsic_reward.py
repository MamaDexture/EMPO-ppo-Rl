
import numpy as np


class IntrinsicReward:

    def __init__(
        self,
        threshold=0.95,
        bonus_reward=0.1
    ):

        self.threshold = threshold
        self.bonus_reward = bonus_reward

        self.visited_states = []

    def cosine_similarity(
        self,
        state1,
        state2
    ):

        state1 = np.array(state1).flatten()
        state2 = np.array(state2).flatten()

        norm1 = np.linalg.norm(state1)
        norm2 = np.linalg.norm(state2)

        if norm1 == 0 or norm2 == 0:
            return 0

        return np.dot(
            state1,
            state2
        ) / (norm1 * norm2)

    def is_novel(
        self,
        state
    ):

        if len(self.visited_states) == 0:
            return True

        similarities = []

        for visited in self.visited_states:

            similarity = self.cosine_similarity(
                state,
                visited
            )

            similarities.append(similarity)

        max_similarity = max(similarities)

        return max_similarity < self.threshold

    def compute_bonus(
        self,
        state
    ):

        if self.is_novel(state):

            self.visited_states.append(
                np.array(state)
            )

            return self.bonus_reward

        return 0.0

    def reset(self):

        self.visited_states = []
