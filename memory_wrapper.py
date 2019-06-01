from tensorflow.nn.rnn_cell import RNNCell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import gen_array_ops, tensor_array_ops
from tensorflow.nn.rnn_cell import LSTMStateTuple
import tensorflow as tf


class MemoryWrapper(RNNCell):
    def __init__(self, cell, memory, batch_size, embedding_size, hidden_size, mem_num, state_is_tuple=True,
                 beam_width=1):
        """Create a cell with input projection.

    Args:
      cell: an RNNCell, a projection of inputs is added before it.
      memory: memory words after embedding

    Raises:
      TypeError: if cell is not an RNNCell.
    """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not RNNCell.")
        self._cell = cell
        self.memory = memory
        self.embedding_size = embedding_size
        self.mem_num = mem_num
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self._state_is_tuple = state_is_tuple
        self.beam_width = beam_width

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        """Run the input projection and then the cell."""
        # Default scope: "InputProjectionWrapper"
        dtype = inputs.dtype
        with vs.variable_scope("memory_projection"):
            c_t, h_t = state
            w = vs.get_variable('memory_matrix', [self.hidden_size, self.embedding_size],
                                dtype=dtype, trainable=True, initializer=init_ops.random_normal_initializer)
            b = vs.get_variable('memory_bias', [self.embedding_size], dtype=dtype,
                                trainable=True, initializer=init_ops.zeros_initializer)

            # map h_t [batch_size x hidden_size]  to  [batch_size x embedding_size]
            v = math_ops.tanh(nn_ops.xw_plus_b(h_t, w, b))
            # v = tf.matmul(tf.reshape(self.mem, (self.batch_size * self.mem_num, -1)), w)  # [batch x mem_num, hidden]
            # print(v) # 80 x 64

            # need to solve problem when using beam search decoder
            # if v.get_shape()[0] != self.batch_size:
            #     print(self.beam_width)
            #     v = array_ops.reshape(v, [self.batch_size, self.beam_width, self.embedding_size])
            #
            # else:
            # inner product  [ batch_size, 1, topic_num]
            similarity = math_ops.matmul(array_ops.expand_dims(v, 1),  # batch_size, 1 , embedding_size
                                         array_ops.reshape(self.memory, [self.batch_size, self.embedding_size, -1]))

            weight = nn_ops.softmax(
                array_ops.squeeze(similarity)
            )

            # vv = tf.tanh(tf.matmul(tf.expand_dims(h_t, 1),  # [batch_size, hidden_size, 1]
            #                        tf.reshape(v, [self.batch_size, self.hidden_size,
            #                                       self.mem_num])))  # [batch_size, hidden_size, mem_num]
            # print(vv.shape) # 4, 1 , 20

            # weight = tf.nn.softmax(tf.squeeze(vv))
            weight_tile = gen_array_ops.tile(array_ops.expand_dims(weight, -1), [1, 1, self.embedding_size],
                                             name="weight")
            # print(weight_tile) # batch_size x num x embedding_size
            # print(memory * weight_tile)
            mt = math_ops.reduce_sum(self.memory * weight_tile, axis=1)

            # add mt to hidden state computation instead of concat with input
            # not work well
            # w_m = vs.get_variable('projection_m', [self.embedding_size, self.hidden_size],
            #                       dtype=dtype, trainable=True, initializer=init_ops.random_normal_initializer)
            # w_h = vs.get_variable('projection_h', [self.hidden_size, self.hidden_size],
            #                       dtype=dtype, trainable=True, initializer=init_ops.random_normal_initializer)

            # h_topic = math_ops.tanh(math_ops.matmul(mt, w_m) + math_ops.matmul(h_t, w_h))
            # new_state = LSTMStateTuple(c_t, h_topic)
            # return self._cell(inputs, new_state)
        return self._cell(tf.concat([inputs, mt], axis=1), state)


class MemoryWrapperBeam(RNNCell):
    def __init__(self, cell, memory, batch_size, embedding_size, hidden_size, mem_num, state_is_tuple=True,
                 beam_width=1):
        """
        Create a cell with input projection, with beam search support
            Args:
              cell: an RNNCell, a projection of inputs is added before it.
              memory: memory words after embedding

            Raises:
              TypeError: if cell is not an RNNCell.
        """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not RNNCell.")
        self._cell = cell
        self.memory = memory
        self.embedding_size = embedding_size
        self.mem_num = mem_num
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self._state_is_tuple = state_is_tuple
        self.beam_width = beam_width

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        """Run the input projection and then the cell."""
        # Default scope: "InputProjectionWrapper"
        dtype = inputs.dtype
        with vs.variable_scope("memory_projection"):
            c_t, h_t = state
            w = vs.get_variable('memory_matrix', [self.hidden_size, self.embedding_size],
                                dtype=dtype, trainable=True, initializer=init_ops.random_normal_initializer)
            b = vs.get_variable('memory_bias', [self.embedding_size], dtype=dtype,
                                trainable=True, initializer=init_ops.zeros_initializer)

            # map h_t [batch_size x hidden_size]  to  [batch_size x embedding_size]
            v = math_ops.tanh(nn_ops.xw_plus_b(h_t, w, b))
            # not sure if it is correct
            if v.get_shape()[0] != self.batch_size:
                v = array_ops.reshape(v, [self.batch_size, self.beam_width, self.embedding_size])
                similarity = math_ops.matmul(v,  # batch_size, 1 , embedding_size
                                             array_ops.reshape(self.memory, [self.batch_size, self.embedding_size, -1]))
                print(similarity)  # batch_size x beam_width x memory_num
                weight = nn_ops.softmax(similarity, axis=2)
                # print(weight) # 128, 5, 60
                weight = array_ops.reshape(weight,
                                           [-1, self.mem_num])  # 128 x 5, 60
                weight_tile = gen_array_ops.tile(array_ops.expand_dims(weight, -1), [1, 1, self.embedding_size],
                                                 name="weight_tile")
                memory_tile = gen_array_ops.tile(self.memory, [self.beam_width, 1, 1])
                mt = math_ops.reduce_sum(memory_tile * weight_tile, axis=1)  # 640 x embedding_size

                mt = array_ops.reshape(mt, [self.batch_size, self.beam_width, -1])
                inputs_s = array_ops.reshape(inputs, [self.batch_size, self.beam_width, -1])
                # print("mt: ", mt)
                # print("inputs: ", inputs.shape)
                inputs_ex = array_ops.expand_dims(inputs_s, axis=2)
                mt_ex = array_ops.expand_dims(mt, axis=2)
                concat = array_ops.concat([inputs_ex, mt_ex], axis=2)
                final_inputs = array_ops.reshape(concat, [self.batch_size * self.beam_width, -1])
                # print(final_inputs)
            else:
                # else:
                # inner product  [ batch_size, 1, topic_num]
                similarity = math_ops.matmul(array_ops.expand_dims(v, 1),  # batch_size, 1 , embedding_size
                                             array_ops.reshape(self.memory, [self.batch_size, self.embedding_size, -1]))

                weight = nn_ops.softmax(
                    array_ops.squeeze(similarity)  # batch_size, topic_num
                )
                weight_tile = gen_array_ops.tile(array_ops.expand_dims(weight, -1), [1, 1, self.embedding_size],
                                                 name="weight")
                # print(weight_tile) # batch_size x num x embedding_size
                # print(memory * weight_tile)
                mt = math_ops.reduce_sum(self.memory * weight_tile, axis=1)

                # add mt to hidden state computation instead of concat with input
                # not work well
                # w_m = vs.get_variable('projection_m', [self.embedding_size, self.hidden_size],
                #                       dtype=dtype, trainable=True, initializer=init_ops.random_normal_initializer)
                # w_h = vs.get_variable('projection_h', [self.hidden_size, self.hidden_size],
                #                       dtype=dtype, trainable=True, initializer=init_ops.random_normal_initializer)

                # h_topic = math_ops.tanh(math_ops.matmul(mt, w_m) + math_ops.matmul(h_t, w_h))
                # new_state = LSTMStateTuple(c_t, h_topic)
                # return self._cell(inputs, new_state)
                final_inputs = array_ops.concat([inputs, mt], axis=1)
        # return self._cell(inputs, state) # turn off memory can not work since kernel size changed
        return self._cell(final_inputs, state)


class MemoryWrapperEnhanced(RNNCell):
    def __init__(self, cell, memory, batch_size, embedding_size, hidden_size, mem_num,
                 uu, uv, w, b, state_is_tuple=True, beam_width=1, max_len=100, require_weight=False):
        """Create a cell with input projection.  update memory each step

    Args:
      cell: an RNNCell, a projection of inputs is added before it.
      memory: memory words after embedding

    Raises:
      TypeError: if cell is not an RNNCell.
    """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not RNNCell.")
        self._cell = cell
        self.memory = memory
        self.embedding_size = embedding_size
        self.mem_num = mem_num
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self._state_is_tuple = state_is_tuple
        self.beam_width = beam_width
        self.max_len = max_len

        self.uu = uu
        self.uv = uv
        self.w = w
        self.b = b
        self.weight_list = []
        self.require_weight = require_weight
        self.weights_array = tensor_array_ops.TensorArray(dtype=tf.float32, size=max_len)
        self.idx = tf.constant(0, dtype=tf.int32)

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def copy_params(self, cell):
        self.uu = array_ops.identity(cell.uu)
        self.uv = array_ops.identity(cell.uv)
        self.w = array_ops.identity(cell.w)
        self.b = array_ops.identity(cell.b)

    # def get_weight_array(self):
    #     print(len(self.weight_list))
    #     for i in range(len(self.weight_list)):
    #         self.weights_array.write(tf.constant(i, dtype=tf.int32), array_ops.identity(self.weight_list[i]))
    #     return self.weights_array.stack()

    def __call__(self, inputs, state, scope=None):
        """Run the input projection and then the cell."""
        memory = array_ops.identity(self.memory)

        # array_ops.ref_identity()
        # deep_copy(self.memory)
        with vs.variable_scope("memory_projection"):
            c_t, h_t = state

            v = math_ops.tanh(nn_ops.xw_plus_b(h_t, self.w, self.b))
            # not sure if it is correct
            if v.get_shape()[0] != self.batch_size:
                raise Exception("Beam Search Not supported now!")
            else:
                similarity = math_ops.matmul(array_ops.expand_dims(v, 1),  # batch_size, 1 , embedding_size
                                             array_ops.transpose(memory, [0, 2, 1]))

                weight = nn_ops.softmax(
                    array_ops.squeeze(similarity)  # batch_size, topic_num
                )
                # self.weight_list.append(weight)
                # self.weights_array.write(self.idx, similarity)
                # self.idx += 1
                weight_tile = gen_array_ops.tile(array_ops.expand_dims(weight, -1), [1, 1, self.embedding_size],
                                                 name="weight")
                # print(weight_tile) # batch_size x num x embedding_size
                # print(memory * weight_tile)
                mt = math_ops.reduce_sum(memory * weight_tile, axis=1)

            # update memory

            # memory [ batch_size, num, emb_size ]
            # inputs [batch_size, emb_size]
            # state [batch_size, hidden_size]
            gate = math_ops.matmul(memory,
                                   array_ops.expand_dims(inputs, axis=2))  # [batch_size, num, 1]
            gate = math_ops.sigmoid(gen_array_ops.squeeze(gate))  # batch_size x num

            inputs_expand = gen_array_ops.tile(array_ops.expand_dims(inputs, axis=1),
                                               [1, self.mem_num, 1])  # batch_size x num x embedding

            uu_tile = gen_array_ops.tile(array_ops.expand_dims(self.uu, axis=0),
                                         [self.batch_size, 1, 1])  # batch_size x embedding x embedding

            vv_tile = gen_array_ops.tile(array_ops.expand_dims(self.uv, axis=0),
                                         [self.batch_size, 1, 1])  # batch_size x embedding x embedding

            candidate = math_ops.add(
                math_ops.matmul(inputs_expand, uu_tile),
                math_ops.matmul(memory, vv_tile)
            )  # batch_size x num x embedding
            # print(gate)
            gate_tile = gen_array_ops.tile(array_ops.expand_dims(gate, 2),
                                           [1, 1, self.embedding_size])
            updated_mem = (1 - gate_tile) * memory + gate_tile * candidate
            self.memory = updated_mem

        if self.require_weight:
            return self._cell(tf.concat([inputs, mt], axis=1), state), weight
        else:
            return self._cell(tf.concat([inputs, mt], axis=1), state)


class AttentionMemWrapper(RNNCell):
    def __init__(self, cell, memory, encoder_outputs, batch_size, embedding_size, hidden_size, mem_num,
                 uu, uv, w, b, query_layer, memory_layer, attention_v,
                 state_is_tuple=True, attention_size=128, update_mem=True):
        """cell with attention and memory support

    Args:
      cell: an RNNCell, a projection of inputs is added before it.
      memory: memory words after embedding

    Raises:
      TypeError: if cell is not an RNNCell.
    """
        if not isinstance(cell, RNNCell):
            raise TypeError("The parameter cell is not RNNCell.")
        self._cell = cell
        self.memory = memory
        self.embedding_size = embedding_size
        self.mem_num = mem_num
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self._state_is_tuple = state_is_tuple
        self.encoder_outputs = encoder_outputs
        self.attention_size = attention_size
        self.update_mem = update_mem
        # memory variables
        self.uu = uu
        self.uv = uv
        self.w = w
        self.b = b

        # attention variable
        self.query_layer = query_layer  # densen layer to attention_size
        self.memory_layer = memory_layer  # dense layer to attention_size
        self.attention_v = attention_v  # attention variable [attention_size]

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        """Run the input projection and then the cell."""
        dtype = inputs.dtype
        memory = array_ops.identity(self.memory)

        # array_ops.ref_identity()
        # deep_copy(self.memory)
        with vs.variable_scope("memory_projection"):
            c_t, h_t = state

            # print(self.uu)
            # print(self.uv)
            # print(self.w)
            # print(self.b)
            # map h_t [batch_size x hidden_size]  to  [batch_size x embedding_size]
            v = math_ops.tanh(nn_ops.xw_plus_b(h_t, self.w, self.b))
            # not sure if it is correct
            if v.get_shape()[0] != self.batch_size:
                raise Exception("Beam Search Not supported now!")
            else:
                # else:
                # inner product  [ batch_size, 1, topic_num]
                similarity = math_ops.matmul(array_ops.expand_dims(v, 1),  # batch_size, 1 , embedding_size
                                             array_ops.transpose(memory, [0, 2, 1]))

                weight = nn_ops.softmax(
                    array_ops.squeeze(similarity)  # batch_size, topic_num
                )
                weight_tile = gen_array_ops.tile(array_ops.expand_dims(weight, -1), [1, 1, self.embedding_size],
                                                 name="weight")
                # print(weight_tile) # batch_size x num x embedding_size
                # print(memory * weight_tile)
                mt = math_ops.reduce_sum(memory * weight_tile, axis=1)

            # update memory

            # memory [ batch_size, num, emb_size ]
            # inputs [batch_size, emb_size]
            # state [batch_size, hidden_size]
            if self.update_mem:
                gate = math_ops.matmul(memory,
                                       array_ops.expand_dims(inputs, axis=2))  # [batch_size, num, 1]
                gate = math_ops.sigmoid(gen_array_ops.squeeze(gate))  # batch_size x num

                inputs_expand = gen_array_ops.tile(array_ops.expand_dims(inputs, axis=1),
                                                   [1, self.mem_num, 1])  # batch_size x num x embedding

                uu_tile = gen_array_ops.tile(array_ops.expand_dims(self.uu, axis=0),
                                             [self.batch_size, 1, 1])  # batch_size x embedding x embedding

                vv_tile = gen_array_ops.tile(array_ops.expand_dims(self.uv, axis=0),
                                             [self.batch_size, 1, 1])  # batch_size x embedding x embedding

                candidate = math_ops.add(
                    math_ops.matmul(inputs_expand, uu_tile),
                    math_ops.matmul(memory, vv_tile)
                )  # batch_size x num x embedding
                # print(gate)
                gate_tile = gen_array_ops.tile(array_ops.expand_dims(gate, 2),
                                               [1, 1, self.embedding_size])
                updated_mem = (1 - gate_tile) * memory + gate_tile * candidate
                self.memory = updated_mem

        with vs.variable_scope("attention_mechanism"):
            # [batch_size, max_len, attention_size]
            # print(self.encoder_outputs)  # [batch_size, max_len, hidden_size]

            encoder_processed = self.memory_layer(self.encoder_outputs)  # map to attention size
            # [batch_size,  hidden_size] -> [batch_size, 1, attention_size]
            query_processed = array_ops.expand_dims(self.query_layer(c_t), 1)

            scores = math_ops.reduce_sum(self.attention_v * math_ops.tanh(encoder_processed + query_processed), [2])
            print(scores)
            alpha = nn_ops.softmax(scores, axis=1)
            output_hidden_size = self.encoder_outputs.shape[2].value
            alpha_tile = gen_array_ops.tile(array_ops.expand_dims(alpha, -1), [1, 1, output_hidden_size],
                                            name="weight")
            # print(weight_tile) # batch_size x num x embedding_size
            # print(memory * weight_tile)
            weighted_sum = math_ops.reduce_sum(self.encoder_outputs * alpha_tile, axis=1)
            # print(weighted_sum)
        return self._cell(tf.concat([inputs, weighted_sum, mt], axis=1), state)
