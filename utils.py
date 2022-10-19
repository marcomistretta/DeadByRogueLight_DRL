import numpy as np
import json
import colorama, textwrap
import random
from copy import deepcopy

colorama.init(convert=True)

def color_string(description, value=None):
    ret_string = colorama.Fore.LIGHTGREEN_EX + "{}".format(description)
    if value is None:
        ret_string += colorama.Fore.RESET
    else:
        value = textwrap.fill(str(value), width=80, subsequent_indent=(2 + len(description))*" ")
        ret_string += " " + colorama.Fore.LIGHTRED_EX + "[{}]".format(value) + colorama.Fore.RESET
    return ret_string

def product(xs, empty=1):
    result = None
    for x in xs:
        if result is None:
            result = x
        else:
            result *= x

    if result is None:
        result = empty

    return result

class RunningStat(object):
        def __init__(self, shape=()):
            self._n = 0
            self._M = np.zeros(shape)
            self._S = np.zeros(shape)

        def push(self, x):
            x = np.asarray(x)
            assert x.shape == self._M.shape
            self._n += 1
            if self._n == 1:
                self._M[...] = x
            else:
                oldM = self._M.copy()
                self._M[...] = oldM + (x - oldM)/self._n
                self._S[...] = self._S + (x - oldM)*(x - self._M)

        @property
        def n(self):
            return self._n

        @property
        def mean(self):
            return self._M

        @property
        def var(self):
            if self._n >= 2:
                return self._S/(self._n - 1)
            else:
                return np.square(self._M)

        @property
        def std(self):
            return np.sqrt(self.var)

        @property
        def shape(self):

            return self._M.shape

class LimitedRunningStat(object):
    def __init__(self, len=1000):
        self.values = np.array(np.zeros(len))
        self.n_values = 0
        self.i = 0
        self.len = len

    def push(self, x):
        self.values[self.i] = x
        self.i = (self.i + 1) % len(self.values)
        if self.n_values < len(self.values):
            self.n_values += 1

    @property
    def n(self):
        return self.n_values

    @property
    def mean(self):
        return np.mean(self.values[:self.n_values])

    @property
    def var(self):
        return np.var(self.values[:self.n_values])

    @property
    def std(self):
        return np.std(self.values[:self.n_values])

class DynamicRunningStat(object):

    def __init__(self):
        self.current_rewards = list()
        self.next_rewards = list()

    def push(self, x):
        self.next_rewards.append(x)

    def reset(self):
        self.current_rewards = self.next_rewards
        self.next_rewards = list()

    @property
    def n(self):
        return len(self.current_rewards)

    @property
    def mean(self):
        return np.mean(np.asarray(self.current_rewards))

    @property
    def std(self):
        return np.std(np.asarray(self.current_rewards))


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def shape_list(x):
    '''
        deal with dynamic shape in tensorflow cleanly
    '''
    ps = x.get_shape().as_list()
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]

def stable_masked_softmax(logits, mask):

    #  Subtract a big number from the masked logits so they don't interfere with computing the max value
    if mask is not None:
        mask = tf.expand_dims(mask, 2)
        logits -= (1.0 - mask) * 1e10

    #  Subtract the max logit from everything so we don't overflow
    logits -= tf.reduce_max(logits, axis=-1, keepdims=True)
    unnormalized_p = tf.exp(logits)

    #  Mask the unnormalized probibilities and then normalize and remask
    if mask is not None:
        unnormalized_p *= mask
    normalized_p = unnormalized_p / (tf.reduce_sum(unnormalized_p, axis=-1, keepdims=True) + 1e-10)
    if mask is not None:
        normalized_p *= mask
    return normalized_p

def entity_avg_pooling_masked(x, mask):
    '''
        Masks and pools x along the second to last dimension. Arguments have dimensions:
            x:    batch x time x n_entities x n_features
            mask: batch x time x n_entities
    '''
    mask = tf.expand_dims(mask, -1)
    masked = x * mask
    summed = tf.reduce_sum(masked, -2)
    denom = tf.reduce_sum(mask, -2) + 1e-5
    return summed / denom

def entity_max_pooling_masked(x, mask):
    '''
        Masks and pools x along the second to last dimension. Arguments have dimensions:
            x:    batch x time x n_entities x n_features
            mask: batch x time x n_entities
    '''
    mask = tf.expand_dims(mask, -1)
    has_unmasked_entities = tf.sign(tf.reduce_sum(mask, axis=-2, keepdims=True))
    offset = (mask - 1) * 1e9
    masked = (x + offset) * has_unmasked_entities
    return tf.reduce_max(masked, -2)

# Boltzmann transformation to probability distribution
def boltzmann(probs, temperature = 1.):
    sum = np.sum(np.power(probs, 1/temperature))
    new_probs = []
    for p in probs:
        new_probs.append(np.power(p, 1/temperature) / sum)

    return np.asarray(new_probs)

# Very fast np.random.choice
def multidimensional_shifting(num_samples, sample_size, elements, probabilities):
    # replicate probabilities as many times as `num_samples`
    replicated_probabilities = np.tile(probabilities, (num_samples, 1))
    # get random shifting numbers & scale them correctly
    random_shifts = np.random.random(replicated_probabilities.shape)
    random_shifts /= random_shifts.sum(axis=1)[:, np.newaxis]
    # shift by numbers & find largest (by finding the smallest of the negative)
    shifted_probabilities = random_shifts - replicated_probabilities
    return np.argpartition(shifted_probabilities, sample_size, axis=1)[:, :sample_size]

def tf_normalize(value, tmin, tmax, rmin=-1, rmax=1):
    return (((value - rmin) / (rmax - rmin))*(tmax - tmin)) + tmin

# This class will create the dataset for the Decision Transformer algorithm
class DTDemDataset:
    def __init__(self, trajectories, context_len, rtg_scale, state_norm=False, gamma=.99):

        # Trajectories must be a list of *episodes dict*, each of one should be a dict:
        # states => list of states of the trajectory_i
        # actions => list of actions of the trajectory_i
        # rewards => list of actions of the rewards_i
        self.trajectories = trajectories
        self.context_len = context_len

        # Compute the minimum length of the trajectories, and if state_norm=True normalize the *states*
        # In the original paper they scale the running reward, we set scale=1
        min_len = 10 ** 6
        states = []
        for traj in self.trajectories:
            traj_len = len(traj['states'])
            min_len = min(min_len, traj_len)
            states.append(traj['states'])
            traj['returns_to_go'] = self.discount_cumsum(traj['rewards'], gamma) / rtg_scale

        if state_norm:
            states = np.concatenate(states, axis=0)
            self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

            # normalize states
            for traj in self.trajectories:
                traj['states'] = (traj['states'] - self.state_mean) / self.state_std

    def get_state_stats(self):
        return self.state_mean, self.state_std

    def __len__(self):
        return len(self.trajectories)

    # Compute discount cumulative reward. In the original paper, gamma=1
    def discount_cumsum(self, x, gamma):
        disc_cumsum = np.zeros_like(x)
        disc_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            disc_cumsum[t] = x[t] + gamma * disc_cumsum[t + 1]
        return disc_cumsum

    # This will do the dataset magic. It creates sequence of context_len transitions to create the conext.
    # The sequences are created sampling a random index to slice the trajectories (if this is > context_len)
    def get_item(self, idx):
        traj = self.trajectories[idx]
        traj_len = len(traj['states'])

        # If the trajectory is less than context_len, we need to pad it and create a *mask*
        # (we can do it as we are using transformers)

        if traj_len >= self.context_len:
            si = random.randint(0, traj_len - self.context_len)

            states = traj['states'][si: si + self.context_len]
            actions = traj['actions'][si: si + self.context_len]
            returns_to_go = traj['returns_to_go'][si: si + self.context_len]
            timesteps = np.arange(si, si + self.context_len)

            traj_mask = np.ones(self.context_len)
        else:
            padding_len = self.context_len - traj_len

            states = deepcopy(traj['states'])
            # TODO: this works only if you have global_in. For now it is okay but correct this
            for p in range(padding_len):
                states.append(dict(global_in=np.zeros(np.shape(states[0]['global_in']))))

            actions = traj['actions']
            actions = np.concatenate([actions, np.zeros(([padding_len] + list(actions.shape[1:])))], axis=0)

            returns_to_go = traj['returns_to_go']
            returns_to_go = np.concatenate([returns_to_go, np.zeros(([padding_len] + list(returns_to_go.shape[1:])))],
                                      axis=0)

            timesteps = np.arange(0, self.context_len)
            traj_mask = np.concatenate([np.ones(traj_len), np.zeros(padding_len)], axis=0)

        return timesteps, states, actions, returns_to_go, traj_mask

    def get_minibatch(self, batch_size):
        random_indices = np.random.choice(len(self.trajectories), batch_size, replace=False)
        timesteps = []
        states = []
        actions = []
        returns_to_go = []
        traj_mask = []

        for idx in random_indices:
            i_timesteps, i_states, i_actions, i_rtg, i_tmask = self.get_item(idx)
            timesteps.append(i_timesteps)
            states.append(i_states)
            actions.append(i_actions)
            returns_to_go.append(i_rtg)
            traj_mask.append(i_tmask)

        return timesteps, states, actions, returns_to_go, traj_mask
