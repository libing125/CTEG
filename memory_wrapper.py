from tensorflow.nn.rnn_cell import RNNCell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import gen_array_ops, tensor_array_ops
from tensorflow.nn.rnn_cell import LSTMStateTuple
import tensorflow as tf


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

            v = math_ops.tanh(nn_ops.xw_plus_b(h_t, self.w, self.b))
            if v.get_shape()[0] != self.batch_size:
                raise Exception("Beam Search Not supported now!")
            else:
                similarity = math_ops.matmul(array_ops.expand_dims(v, 1),  # batch_size, 1 , embedding_size
                                             array_ops.transpose(memory, [0, 2, 1]))

                weight = nn_ops.softmax(
                    array_ops.squeeze(similarity)  # batch_size, topic_num
                )
                weight_tile = gen_array_ops.tile(array_ops.expand_dims(weight, -1), [1, 1, self.embedding_size],
                                                 name="weight")
                mt = math_ops.reduce_sum(memory * weight_tile, axis=1)

            # update memory
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
            weighted_sum = math_ops.reduce_sum(self.encoder_outputs * alpha_tile, axis=1)
        return self._cell(tf.concat([inputs, weighted_sum, mt], axis=1), state)
