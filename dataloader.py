import numpy as np
import time


class GenDataLoader(object):
    def __init__(self, batch_size, source_index, source_len, target_idx, target_len,
                 max_len, source_label=None, memory=None):
        assert len(source_index) == len(target_idx)
        self.batch_size = batch_size
        self.source_idx = source_index
        self.source_len = source_len
        self.target_idx = target_idx
        self.target_len = target_len
        self.max_len = max_len
        self.has_label = False
        if source_label is not None:
            self.has_label = True
            self.source_label = source_label
        if memory is not None:
            self.has_mem = True
            self.memory = memory
        self.num_batch = len(source_index) // batch_size

    def create_batch(self):
        self.si_batch = np.split(self.source_idx[:self.num_batch * self.batch_size], self.num_batch)
        self.sl_batch = np.split(self.source_len[:self.num_batch * self.batch_size], self.num_batch)
        self.tl_batch = np.split(self.target_len[:self.num_batch * self.batch_size], self.num_batch)
        self.ti_batch = np.split(self.target_idx[:self.num_batch * self.batch_size], self.num_batch)
        if self.has_label:
            self.slbl = np.split(self.source_label[:self.num_batch * self.batch_size], self.num_batch)
        if self.has_mem:
            self.smem = np.split(self.memory[:self.num_batch * self.batch_size], self.num_batch)

        self.g_pointer = 0

    def next_batch(self):
        generator_batch = [self.si_batch[self.g_pointer],
                           self.sl_batch[self.g_pointer],
                           self.ti_batch[self.g_pointer],
                           self.tl_batch[self.g_pointer],
                           ]
        if self.has_label:
            generator_batch.append(self.slbl[self.g_pointer])
        if self.has_mem:
            generator_batch.append(self.smem[self.g_pointer])
        self.g_pointer = (self.g_pointer + 1) % self.num_batch
        return generator_batch

    def reset_pointer(self):
        self.g_pointer = 0


class DisDataLoader(object):
    def __init__(self, sess, generator, batch_size, max_len, num_class, topic_input, topic_len, topic_label, target_idx,
                 memory):
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_class = num_class
        self.G = generator
        self.sess = sess
        self.topic_input = topic_input
        self.topic_len = topic_len
        self.topic_label = topic_label
        self.target_idx = target_idx
        self.memory = memory

    def prepare_data(self, generate_batches):
        generate_num = generate_batches * self.G.batch_size
        if generate_num > len(self.topic_input):
            generate_batches = len(self.topic_input) // self.G.batch_size
            generate_num = generate_batches * self.G.batch_size

        # shuffle
        ti, tl, tlbl, tidx, tmem = shuffle_data(generate_num, self.topic_input, self.topic_len,
                                                self.topic_label, self.target_idx, self.memory)

        # print(tidx.shape) # batch_size x dynamic_length
        # print("shuffle time cost: %.3f" % (time.time() - t0))
        fake_idx = []
        t1 = time.time()
        for gb in range(generate_batches):
            # print(gb)
            # print(ti[gb * self.G.batch_size:(gb+1) * self.G.batch_size])
            fake_idx.extend(self.G.generate_essay(self.sess,
                                                  ti[gb * self.G.batch_size:(gb + 1) * self.G.batch_size],
                                                  tl[gb * self.G.batch_size:(gb + 1) * self.G.batch_size],
                                                  memory=tmem[gb * self.G.batch_size: (gb + 1) * self.G.batch_size]))
        print("generate time cost: %.4f" % (time.time() - t1))

        # print(tidx.shape)
        # print(fake_idx.shape)

        fake_label = np.zeros([1, self.num_class], dtype=int)
        fake_label[0, self.num_class - 1] += 1  # one-hot at last dimension
        fake_labels = np.repeat(fake_label, len(fake_idx), axis=0)
        padded_fake = self._padding(fake_idx, self.max_len)
        padded_true = self._pad_numpy(tidx, self.max_len)
        self.idx = np.concatenate([padded_fake, padded_true], axis=0)
        # print(fake_labels.shape)
        # print(tlbl.shape)
        self.labels = np.concatenate([fake_labels, tlbl], axis=0)

        # split batches
        self.num_batch = int(len(self.labels) / self.batch_size)
        self.idx = self.idx[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.idx_batches = np.split(self.idx, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)
        self.pointer = 0

    def prepare_data_no_fake(self):

        tlbl, tidx = shuffle_data(len(self.topic_input),
                                  self.topic_label, self.target_idx)

        self.idx = self._pad_numpy(tidx, self.max_len)
        self.labels = tlbl

        self.num_batch = int(len(self.labels) / self.batch_size)
        self.idx = self.idx[:self.num_batch * self.batch_size]
        self.labels = self.labels[:self.num_batch * self.batch_size]

        self.idx_batches = np.split(self.idx, self.num_batch, 0)
        self.labels_batches = np.split(self.labels, self.num_batch, 0)
        self.pointer = 0

    def next_batch(self):
        batch_idx, batch_label = self.idx_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return batch_idx, batch_label

    def _pad_numpy(self, index, max_len):
        batch_size = len(index)
        padded = np.zeros([batch_size, max_len], dtype=int)
        for i in range(batch_size):
            true_len = min(max_len, len(index[i]))
            for j in range(true_len):
                padded[i, j] = index[i][j]
        return padded

    def _padding(self, inputs, max_sequence_length):
        batch_size = len(inputs)
        inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)  # == PAD
        for i, seq in enumerate(inputs):
            for j, element in enumerate(seq):
                inputs_batch_major[i, j] = element
        return inputs_batch_major

    def reset(self):
        self.pointer = 0


def shuffle_data(num, *data):
    size = len(data[0])
    permutation = np.random.permutation(size)
    ret = []
    for d in data:
        d = d[permutation]
        ret.append(d[:num])
    return ret


def padding(index, max_len):
    batch_size = len(index)
    padded = np.zeros([batch_size, max_len])
    for i, seq in enumerate(index):
        for j, element in enumerate(seq):
            padded[i, j] = element
    return padded


def get_weights(lengths, max_len):
    x_len = len(lengths)
    ans = np.zeros((x_len, max_len))
    for ll in range(x_len):
        kk = lengths[ll] - 1
        for jj in range(kk):
            # print(ll)
            # print(jj)
            ans[ll][jj] = 1 / float(kk)
    return ans


def prepare_data(test_ratio, *data):
    length = len(data[0])
    test_size = int(length * test_ratio)
    print(test_size)
    print(length - test_size)
    permute = np.random.permutation(length)
    train = []
    test = []
    for d in data:
        d = d[permute]
        d_test = d[:test_size]
        d_train = d[test_size:]
        train.append(d_train)
        test.append(d_test)
    return train, test


def load_npy(data_config):
    ret = []
    # print(data_config)
    for item in data_config:
        # print(item)
        ret.append(np.load(item))
    return ret


def to_one_hot(arr, num_class):
    size = len(arr)
    lbl = np.zeros([size, num_class])
    for i in range(size):
        lbl[i, arr[i]] += 1
    return lbl

