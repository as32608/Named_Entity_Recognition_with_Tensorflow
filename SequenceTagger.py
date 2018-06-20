import os

import numpy as np
import sklearn
import tensorflow as tf

import ner_model_utils


class SequenceTagger:

    def __init__(self,
                 word2ind,
                 ind2word,
                 char2ind=None,
                 use_chars=False,
                 save_path=None,
                 restore_path=None,
                 epochs=30,
                 batch_size=64,
                 num_layers=1,
                 rnn_size=256,
                 char_rnn_size=100,
                 n_hidden_dense=50,
                 embedding_dim=300,
                 char_embedding_dim=50,
                 keep_probability_i=0.825,
                 keep_probability_o=0.895,
                 keep_probability_h=0.86,
                 keep_probability_d=0.93,
                 keep_probability_e=0.986,
                 learning_rate=0.01,
                 learning_rate_decay=0.90,
                 learning_rate_decay_steps=100,
                 max_lr=0.1,
                 use_cyclic=False,
                 n_tags=10,
                 clip=5,
                 train_embeddings=True,
                 summary_dir=None,
                 np_embedding_matrix_path=None,
                 pad_token='<PAD>',
                 use_crf=False):
        """

        Args:
            word2ind: Lookup dict from word 2 index
            ind2word: Lookup dict from index back 2 words
            char2ind: Lookup dict form char 2 index
            use_chars: Boolean. Optionally use char level bidir lstm or not.
            save_path: Path to save the trained tf model to.
            restore_path: Path to restore a trained tf model from.
            epochs: Integer. Number of epochs to run in training.
            batch_size: Integer. Number of points in a single batch.
            num_layers: Integer.
            rnn_size: Integer. Number of hidden units in word level lstms.
            char_rnn_size: Integer. Number of hidden units in char level lstms.
            n_hidden_dense: Integer. Number of hidden units in dense layer.
            embedding_dim: Integer. Size of embedding vectors in word level
                           embedding matrix.
            char_embedding_dim: Integer. Size of embedding vectors in char level
                                embedding matrix.
            keep_probability_i: Float. Values inspired by Jeremy Howard's fast.ai course.
            keep_probability_o: Float. Values inspired by Jeremy Howard's fast.ai course.
            keep_probability_h: Float. Values inspired by Jeremy Howard's fast.ai course.
            keep_probability_d: Float. Values inspired by Jeremy Howard's fast.ai course.
            keep_probability_e: Float. Values inspired by Jeremy Howard's fast.ai course.
            learning_rate: Float.
            learning_rate_decay: Integer.
            n_tags: Integer. Number of target values.
            clip: Integer.
            train_embeddings: Boolean.
            summary_dir: Path to write tensorflow summaries to.
            np_embedding_matrix_path: Path to load given embedding matrix from.
            pad_token: Padding token.
            use_crf: Boolean.

        """
        self.word2ind = word2ind
        self.ind2word = ind2word
        self.save_path = save_path
        self.restore_path = restore_path
        self.epochs = epochs
        self.learning_rate_decay = learning_rate_decay
        if rnn_size % 2 != 0:
            rnn_size += 1
        self.rnn_size = rnn_size
        self.n_hidden_dense = n_hidden_dense
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.keep_probability_o = keep_probability_o
        self.keep_probability_i = keep_probability_i
        self.keep_probability_h = keep_probability_h
        self.keep_probability_d = keep_probability_d
        self.keep_probability_e = keep_probability_e
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.n_tags = n_tags
        self.vocab_size = len(self.word2ind)
        self.train_embeddings = train_embeddings
        self.clip = clip
        self.summary_dir = summary_dir
        self.np_embedding_matrix_path = np_embedding_matrix_path
        self.pad_id = self.word2ind[pad_token]
        self.use_crf = use_crf
        self.use_chars = use_chars
        self.learning_rate_decay_steps = learning_rate_decay_steps
        self.max_lr = max_lr
        self.use_cyclic = use_cyclic

        if self.use_chars:
            self.char2ind = char2ind
            self.char_pad_id = self.char2ind[pad_token]
            self.n_chars = len(self.char2ind)
            self.char_embedding_dim = char_embedding_dim
            if char_rnn_size % 2 != 0:
                char_rnn_size += 1
            self.char_rnn_size = char_rnn_size

    def build_graph(self):
        self.add_placeholders()
        self.add_word_embeddings_op()
        self.add_logits_op()
        self.add_pred_op()
        self.add_loss_op()
        self.add_train_op(self.clip)
        self.initialize_session()
        print('Graph built.')

    def add_placeholders(self):
        self.word_ids = tf.placeholder(tf.int32,
                                       shape=[None, None],
                                       name="word_ids")
        self.sequence_lengths = tf.placeholder(tf.int32,
                                               shape=[None],
                                               name="sequence_lengths")
        self.labels = tf.placeholder(tf.int32,
                                     shape=[None, None],
                                     name="labels")

        if self.use_chars:
            self.char_ids = tf.placeholder(tf.int32,
                                           shape=[None, None, None],
                                           name="char_ids")

            self.word_lengths = tf.placeholder(tf.int32,
                                               shape=[None, None],
                                               name="word_lengths")


    def make_lstm(self, rnn_size):
        """Creates LSTM cell wrapped with dropout.
        """
        lstm = tf.nn.rnn_cell.LSTMCell(rnn_size)
        lstm = tf.nn.rnn_cell.DropoutWrapper(lstm,
                                             input_keep_prob=self.keep_probability_i,
                                             output_keep_prob=self.keep_probability_o,
                                             state_keep_prob=self.keep_probability_h)
        return lstm


    def make_attention_cell(self, dec_cell, rnn_size, enc_output, lengths, alignment_history=False):
        """Wraps the given cell with Bahdanau Attention.
        """
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=rnn_size,
                                                                   memory=enc_output,
                                                                   memory_sequence_length=lengths,
                                                                   name='BahdanauAttention')

        return tf.contrib.seq2seq.AttentionWrapper(cell=dec_cell,
                                                   attention_mechanism=attention_mechanism,
                                                   attention_layer_size=None,
                                                   output_attention=False,
                                                   alignment_history=alignment_history)


    def blstm(self,
              inputs,
              seq_length,
              n_hidden,
              scope=None,
              initial_state_fw=None,
              initial_state_bw=None):
        """
        Creates a bidirectional lstm.
        Args:
            inputs: Array of input points.
            seq_length: Array of integers. Sequence lengths of the
                        input points.
            n_hidden: Integer. Number of hidden units to use for
                      rnn cell.
            scope: String.
            initial_state_fw: Initial state of foward cell.
            initial_state_bw: Initial state of backward cell.

        Returns: Tuple of fw and bw output.
                 Tuple of fw and bw state.

        """
        fw_cell = self.make_lstm(n_hidden)
        bw_cell = self.make_lstm(n_hidden)

        (out_fw, out_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=fw_cell,
            cell_bw=bw_cell,
            inputs=inputs,
            sequence_length=seq_length,
            initial_state_fw=initial_state_fw,
            initial_state_bw=initial_state_bw,
            dtype=tf.float32,
            scope=scope
        )
        return (out_fw, out_bw), (state_fw, state_bw)


    def triangular_lr(self, current_step):
        """cyclic learning rate - exponential range."""
        step_size = self.learning_rate_decay_steps
        base_lr = self.learning_rate
        max_lr = self.max_lr

        cycle = tf.floor(1 + current_step / (2 * step_size))
        x = tf.abs(current_step / step_size - 2 * cycle + 1)
        lr = base_lr + (max_lr - base_lr) * tf.maximum(0.0, tf.cast((1.0 - x), dtype=tf.float32)) * (0.99999 ** tf.cast(
            current_step,
            dtype=tf.float32))
        return lr


    def load_np_embedding_matrix(self):
        """
        Loads embedding matrix from .npy file.
        """
        embedding_matrix = np.load(self.np_embedding_matrix_path)
        if len(embedding_matrix.shape) == 3:
            embedding_matrix = embedding_matrix[0]
        print('Loaded embeddings from:', self.np_embedding_matrix_path)
        return embedding_matrix

    def add_word_embeddings_op(self):
        """
        Creates the embedding matrices on word anc char level,
        performs the lookup operations and creates the bidirectional
        lstm on char level.
        """
        with tf.variable_scope('embeddings_op'):
            with tf.variable_scope('words'):
                if self.np_embedding_matrix_path is not None:
                    embed = self.load_np_embedding_matrix()
                    self._word_embeddings = tf.Variable(embed,
                                                        name="w_embeddings",
                                                        dtype=tf.float32,
                                                        trainable=self.train_embeddings)
                else:
                    self._word_embeddings = tf.get_variable(name="w_embeddings",
                                                            dtype=tf.float32,
                                                            shape=[self.vocab_size + 1, self.embedding_dim])

                word_embeddings = tf.nn.embedding_lookup(self._word_embeddings,
                                                         self.word_ids,
                                                         name='word_embeddings')
                word_embeddings = tf.nn.dropout(word_embeddings,
                                                keep_prob=self.keep_probability_e,
                                                name='word_embeddings_dropout')
            with tf.variable_scope('chars'):
                if self.use_chars:
                    _char_embeddings = tf.get_variable(name="_char_embeddings",
                                                       dtype=tf.float32,
                                                       shape=[self.n_chars, self.char_embedding_dim])
                    char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                             self.char_ids,
                                                             name="char_embeddings")
                    char_embeddings = tf.nn.dropout(char_embeddings,
                                                    keep_prob=self.keep_probability_e,
                                                    name='char_embeddings_dropout')

                    s = tf.shape(char_embeddings)
                    char_embeddings = tf.reshape(char_embeddings,
                                                 shape=[s[0] * s[1], s[-2], self.char_embedding_dim])
                    word_lengths = tf.reshape(self.word_lengths,
                                              shape=[s[0] * s[1]])

                    _output = self.blstm(char_embeddings,
                                         word_lengths,
                                         self.char_rnn_size,
                                         scope='charlevel_LSTM')

                    # read and concat output
                    _, ((_, output_fw), (_, output_bw)) = _output
                    output_ch = tf.concat([output_fw, output_bw], axis=-1)
                    output_ch = tf.reshape(output_ch,
                                        shape=[s[0], s[1], 2 * self.char_rnn_size])
                    word_embeddings = tf.concat([word_embeddings, output_ch], axis=-1)

        self.word_embeddings = word_embeddings



    def add_logits_op(self):
        """
        On wordlevel.
        Bidirectional LSTM(s) + attention + fc layers
        Returns the logits.

        """
        with tf.variable_scope('logits_op'):
            with tf.variable_scope("bi_lstm"):
                inputs = self.word_embeddings
                seq_lengths = self.sequence_lengths


                initial_state_fw = None
                initial_state_bw = None
                for n in range(self.num_layers):
                    scope = 'wordlevel_BLSTM' + str(n)
                    (out_fw, out_bw), (state_fw, state_bw) = self.blstm(
                        inputs,
                        seq_lengths,
                        self.rnn_size // 2,
                        scope=scope,
                        initial_state_fw=initial_state_fw,
                        initial_state_bw=initial_state_bw
                    )

                    inputs = tf.concat([out_fw, out_bw], -1)
                    initial_state_fw = state_fw
                    initial_state_bw = state_bw

                bi_state_c = tf.concat((initial_state_fw.c, initial_state_fw.c), -1)
                bi_state_h = tf.concat((initial_state_fw.h, initial_state_fw.h), -1)
                bi_lstm_state = tf.nn.rnn_cell.LSTMStateTuple(c=bi_state_c, h=bi_state_h)
                state = tuple([bi_lstm_state] * self.num_layers)

            # LSTM + attention
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell([self.make_lstm(self.rnn_size) for _ in
                                                    range(self.num_layers)])
            attn_cell = self.make_attention_cell(lstm_cell,
                                                 self.rnn_size,
                                                 inputs,
                                                 self.sequence_lengths)

            initial_state = attn_cell.zero_state(self.batch_size, tf.float32).clone(cell_state=state)

            inputs, _ = tf.nn.dynamic_rnn(attn_cell,
                                          inputs,
                                          sequence_length=self.sequence_lengths,
                                          initial_state=initial_state,
                                          dtype=tf.float32)

            with tf.variable_scope('proj'):
                # fc layer + dropout layer + output layer
                nsteps = tf.shape(inputs)[1]
                output = tf.reshape(inputs, [-1, self.rnn_size])

                output = tf.layers.dense(inputs=output, units=self.n_hidden_dense, activation=tf.nn.relu)
                output = tf.layers.dropout(inputs=output, rate=self.keep_probability_d)

                pred = tf.layers.dense(inputs=output, units=self.n_tags)
                self.logits = tf.reshape(pred, [-1, nsteps, self.n_tags])


    def add_pred_op(self):
        if not self.use_crf:
            self.labels_pred = tf.cast(tf.argmax(self.logits, axis=-1),
                                       tf.int32)


    def add_loss_op(self):
        """
        Computes the loss.
        Option 1: crf log likelihood loss.
        Option 2: cross entropy.
        """
        with tf.variable_scope('loss_op'):
            if self.use_crf:
                log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                    self.logits,
                    self.labels,
                    self.sequence_lengths
                )

                self.trans_params = trans_params

                self.loss = tf.reduce_sum(-log_likelihood)

            else:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                        labels=self.labels)
                mask = tf.sequence_mask(self.sequence_lengths)
                losses = tf.boolean_mask(losses, mask)
                self.loss = tf.reduce_sum(losses)

            if self.summary_dir is not None:
                tf.summary.scalar('loss_value', self.loss)

    def add_train_op(self, clip):
        """
        Creates the training operation.
        Either cyclic learning rate or exponential one.
        Optionally gradients are clipped.
        """
        with tf.variable_scope('train_op'):
            self.global_step = tf.Variable(0, trainable=False)

            if self.use_cyclic:
                self.learning_rate = self.triangular_lr(self.global_step)
            else:
                self.learning_rate = tf.train.exponential_decay(self.learning_rate,
                                                                self.global_step,
                                                                decay_steps=self.learning_rate_decay_steps,
                                                                decay_rate=self.learning_rate_decay,
                                                                staircase=True)

            optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.7, beta2=0.99)

            if clip > 0:  # gradient clipping if clip is positive
                grads, vs = zip(*optimizer.compute_gradients(self.loss))
                grads, gnorm = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs),
                                                          global_step=self.global_step)
            else:
                self.train_op = optimizer.minimize(self.loss)

    def train(self, inputs, targets):
        """
        Performs the training operation.

        """
        with tf.variable_scope('train'):
            best_score = 0
            nepoch_no_imprv = 0
            if self.restore_path is not None:
                self.restore_session()

            inputs = np.array(inputs)

            if self.summary_dir is not None:
                self.add_summary()

            for epoch in range(self.epochs + 1):
                print('-------------------- Epoch {} of {} --------------------'.format(epoch,
                                                                                        self.epochs))
                # shuffle the input data before every epoch.
                shuffle_indices = np.random.permutation(len(inputs))
                inputs = inputs[shuffle_indices]

                score, acc_without = self.run_epoch(inputs, targets, epoch)

                if acc_without > best_score:
                    nepoch_no_imprv = 0
                    if self.save_path is not None:
                        if not os.path.exists(self.save_path):
                            os.makedirs(self.save_path)
                        self.saver.save(self.sess, self.save_path)
                    best_score = acc_without
                    print("--- new best score ---\n\n")
                else:
                    nepoch_no_imprv += 1
                    if nepoch_no_imprv >= 10:
                        print("- early stopping {} epochs without improvement".format(nepoch_no_imprv))
                        break
            # returning best loss for optimization with skopt.
            return best_score

    def run_epoch(self, inputs, targets, epoch):
        """
        Runs a single epoch.
        """
        batch_size = self.batch_size
        nbatches = (len(inputs) + batch_size - 1) // batch_size

        losses = []

        for i, (words, labels) in enumerate(ner_model_utils.minibatches(inputs, batch_size)):

            fd, sl = self.get_feed_dict(words,
                                        labels=labels)

            if self.summary_dir is not None:
                _, train_loss, summary = self.sess.run([self.train_op,
                                                        self.loss,
                                                        self.merged],
                                                       feed_dict=fd)
            else:
                _, train_loss = self.sess.run([self.train_op, self.loss],
                                              feed_dict=fd)

            if i % 50 == 0 or i == (nbatches - 1):
                print('Iteration: {} of {}\ttrain_loss: {:.4f}'.format(i,
                                                                       nbatches - 1,
                                                                       train_loss))
            losses.append(train_loss)

            if i % 50 == 0 and self.summary_dir is not None:
                self.file_writer.add_summary(summary, epoch * nbatches + i)

        avg_loss = self.sess.run(tf.reduce_mean(losses))
        print('Average Score for this Epoch: {}'.format(avg_loss))

        if epoch % 1 == 0:
            valid_loss, _, _, acc, acc_without, n_zeros, n_others, classif_report, classif_report_without = self.run_evaluate(
                targets,
                return_loss=True)
            if self.summary_dir is not None:
                tf.summary.scalar('acc', acc)
                tf.summary.scalar('acc_without', acc_without)

            print('Validation Loss on this epoch:', valid_loss)
            print('Accuracy on this Epoch: {}'.format(acc))
            print('Classifcation_report:\n', classif_report)
            print('Accuracy on this Epoch (except "O"s) : {}'.format(acc_without))
            print('Classifcation_report_without_zeros:\n', classif_report_without)
            print('Number of "O"s: {}\nNumber of other Entities: {}\nIn percent: {}\n'.format(
                n_zeros,
                n_others,
                n_zeros / (n_others + n_zeros))
            )

        return avg_loss, acc_without

    def run_evaluate(self, test, return_loss=False, restore_sess=False):
        """
        Runs a single evalutation epoch.
        """
        if restore_sess:
            self.restore_session()
        preds = []
        actuals = []
        preds_without = []
        actuals_without = []
        n_zeros = 0
        n_others = 0
        losses = []

        for words, labels in ner_model_utils.minibatches(test, self.batch_size):

            if return_loss:
                loss, labels_pred, sequence_lengths = self.predict_batch(words, labels)
                losses.append(loss)
            else:
                labels_pred, sequence_lengths = self.predict_batch(words)

            for lab, lab_pred, length in zip(labels,
                                             labels_pred,
                                             sequence_lengths):
                # we dont care about the padded ones
                lab = lab[:length]
                lab_pred = lab_pred[:length]

                for l, p in zip(lab, lab_pred):
                    if l == 8:
                        n_zeros += 1
                    else:
                        n_others += 1
                    preds.append(p)
                    actuals.append(l)
                    if l != 8:
                        actuals_without.append(l)
                        preds_without.append(p)

        acc = sklearn.metrics.accuracy_score(actuals, preds)

        acc_without = sklearn.metrics.accuracy_score(actuals_without,
                                                     preds_without)
        classif_report = sklearn.metrics.classification_report(actuals,
                                                               preds)
        classif_report_without = sklearn.metrics.classification_report(actuals_without,
                                                                       preds_without)
        if return_loss:
            avg_loss = self.sess.run(tf.reduce_mean(losses))
            return avg_loss, actuals, preds, acc, acc_without, n_zeros, n_others, classif_report, classif_report_without
        else:
            return actuals, preds, acc, acc_without, n_zeros, n_others, classif_report, classif_report_without

    def predict_batch(self, words, labels=None):
        if labels is not None:
            fd, sequence_lengths = self.get_feed_dict(words, labels)
        else:
            fd, sequence_lengths = self.get_feed_dict(words)

        if self.use_crf:
            viterbi_sequences = []
            if labels is not None:
                loss, logits, trans_params = self.sess.run([self.loss, self.logits, self.trans_params],
                                                           feed_dict=fd)
            else:
                logits, trans_params = self.sess.run([self.logits, self.trans_params],
                                                     feed_dict=fd)

            # iterate over the sentences because no batching in viterbi_decode
            for logit, sequence_length in zip(logits, sequence_lengths):
                logit = logit[:sequence_length]
                viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                    logit, trans_params)
                viterbi_sequences += [viterbi_seq]
            if labels is not None:
                return loss, viterbi_sequences, sequence_lengths
            else:
                return viterbi_sequences, sequence_lengths

        else:

            if labels is not None:
                loss, labels_pred = self.sess.run([self.loss, self.labels_pred],
                                                   feed_dict=fd)
                return loss, labels_pred, sequence_lengths

            else:
                labels_pred = self.sess.run(self.labels_pred,
                                            feed_dict=fd)
                return labels_pred, sequence_lengths

    def get_feed_dict(self, words, labels=None):
        """
        Creates the dictionary that is fed to the network's
        placeholders during training and inference.
        """
        if self.use_chars:
            char_ids, word_ids = [], []
            for sentence in words:
                ch, wo = zip(*sentence)
                char_ids.append(ch)
                word_ids.append(wo)

            word_ids, sequence_lengths = ner_model_utils.pad_sequences(word_ids,
                                                                       self.pad_id)
            char_ids, word_lengths = ner_model_utils.pad_sequences(char_ids,
                                                                   self.char_pad_id,
                                                                   chars=True)

        else:
            word_ids, sequence_lengths = ner_model_utils.pad_sequences(words,
                                                                       self.pad_id)

        feed = {self.word_ids: word_ids,
                self.sequence_lengths: sequence_lengths}

        if self.use_chars:
            feed[self.char_ids] = char_ids
            feed[self.word_lengths] = word_lengths

        if labels is not None:
            labels, _ = ner_model_utils.pad_sequences(labels, self.pad_id)
            feed[self.labels] = labels

        return feed, sequence_lengths

    def restore_session(self):
        self.saver.restore(self.sess, self.restore_path)
        print('Restored from', self.restore_path)

    def initialize_session(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def add_summary(self):
        """
        Summaries for TensorBoard.
        """
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.summary_dir,
                                                 self.sess.graph)


