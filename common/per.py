import numpy as np
import random
from common.replay_buffer import ReplayBuffer
from common.segment_tree import SumSegmentTree, MinSegmentTree


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, args):
        super().__init__(args)
        self._alpha = 0.6
        self.per_indexes_priorities = {}
        self._max_priority = 1.0
        self.sample_times = args.sample_times

    def store_episode(self, episode_batch):
        batch_size = episode_batch['o'].shape[0]  # episode_number
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers['o'][idxs] = episode_batch['o']
            self.buffers['u'][idxs] = episode_batch['u']
            self.buffers['s'][idxs] = episode_batch['s']
            self.buffers['r'][idxs] = episode_batch['r']
            self.buffers['o_next'][idxs] = episode_batch['o_next']
            self.buffers['s_next'][idxs] = episode_batch['s_next']
            self.buffers['avail_u'][idxs] = episode_batch['avail_u']
            self.buffers['avail_u_next'][idxs] = episode_batch['avail_u_next']
            self.buffers['u_onehot'][idxs] = episode_batch['u_onehot']
            self.buffers['padded'][idxs] = episode_batch['padded']
            self.buffers['terminated'][idxs] = episode_batch['terminated']
            if self.args.use_ja:
                self.buffers['neighbor_idacts'][idxs] = episode_batch['neighbor_idacts']
                self.buffers['neighbor_ids'][idxs] = episode_batch['neighbor_ids']
                self.buffers['neighbor_mask'][idxs] = episode_batch['neighbor_mask']
                if self.args.use_per:
                    # print(sum(self.buffers['neighbor_mask'][idxs]))
                    iscoop = sum(self.buffers['neighbor_mask'][idxs]).any()
                    if iscoop and idxs not in self.per_indexes:
                        self.per_indexes.append(idxs)
                        self.per_indexes_priorities[idxs] = self._max_priority
                    elif not iscoop and idxs in self.per_indexes:
                        self.per_indexes.remove(idxs)
                        self.per_indexes_priorities.pop(idxs)
            # if self.args.use_dqloss:
            #     self.buffers['neighbor_ids'][idxs] = episode_batch['neighbor_ids']
            if self.args.alg == 'maven':
                self.buffers['z'][idxs] = episode_batch['z']

    def _sample_proportional(self, batch_size):
        add_up = 0
        p_total = sum(self.per_indexes_priorities.values())
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            for idx in self.per_indexes_priorities.keys():
                add_up += self.per_indexes_priorities[idx]
                if add_up > mass:
                    break
        return [idx]

    def sample(self, batch_size):
        temp_buffer = {}
        p = not (np.random.randint(0, self.sample_times))
        if self.args.use_per and len(self.per_indexes) > 0 and p:
            # idx = random.sample(self.per_indexes, batch_size)
            idx = self._sample_proportional(batch_size)
        else:
            idx = np.random.randint(0, self.current_size, batch_size)
        # print(idx, self.per_indexes)
        for key in self.buffers.keys():
            temp_buffer[key] = self.buffers[key][idx]
        return temp_buffer, idx

    # def _get_priorities(self, neighbor_mask, ):

    def update_priorities(self, idxes, priorities):
        i = idxes[0]
        if i in self.per_indexes_priorities.keys():
            self.per_indexes_priorities[i] = priorities ** self._alpha
            # self._it_min[idx] = priority ** self._alpha
            self._max_priority = max(self._max_priority, priorities)
