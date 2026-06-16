
import numpy as np


class MemoryRetriever:

    def __init__(self, memory_buffer):

        self.memory_buffer = memory_buffer

    def cosine_similarity(
        self,
        state1,
        state2
    ):

        state1 = np.array(state1).flatten()
        state2 = np.array(state2).flatten()

        denominator = (
            np.linalg.norm(state1)
            * np.linalg.norm(state2)
        )

        if denominator == 0:
            return 0

        return np.dot(
            state1,
            state2
        ) / denominator

    def retrieve_best_match(
        self,
        current_state
    ):

        memories = self.memory_buffer.get_all()

        if len(memories) == 0:
            return None

        best_memory = None
        best_score = -1

        for memory in memories:

            similarity = self.cosine_similarity(
                current_state,
                memory["state"]
            )

            if similarity > best_score:

                best_score = similarity
                best_memory = memory

        return best_memory

    def retrieve_top_k(
        self,
        current_state,
        k=5
    ):

        memories = self.memory_buffer.get_all()

        if len(memories) == 0:
            return []

        scores = []

        for memory in memories:

            similarity = self.cosine_similarity(
                current_state,
                memory["state"]
            )

            scores.append(
                (
                    similarity,
                    memory
                )
            )

        scores.sort(
            reverse=True,
            key=lambda x: x[0]
        )

        return scores[:k]
