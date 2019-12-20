import os
import pickle as pk
import numpy as np
from scipy import spatial
import tensorflow as tf
import dataset
import evaluate
from util import get_logger

class Model():
    def __init__(self, g, args):
        self.logger = get_logger()

        self.epochs = args.epochs
        self.train_file = './prepare/train_dataset.pk'
        self.eval_file = './prepare/eval_dataset.pk'
        self.valid_file_link = './prepare/valid_dataset.pk'
        self.test_file_link = './prepare/test_dataset.pk'
        self.valid_file_node = './prepare/valid_dataset_node.pk'
        self.test_file_node = './prepare/test_dataset_node.pk'

        self.g = g
        self.vertex_num = g.node_num()
        self.attr_num = g.attr_num()
        self.embed_size = args.embed_size
        self.lambda1 = args.lambda1
        self.neg_sample_num = args.neg_sample_num
        self.learning_rate = args.learning_rate
        self.task = args.task
        self.best_loss = 1e9 #if loss is lower than this, the best model will be saved

        self.train_batch_size = args.train_batch_size
        self.epoch_base = args.epoch_base
        self.summary_dir = './summary'
        self.model_dir = os.path.join(args.model_dir, 'model')

        sess_config = tf.ConfigProto()
        #sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self._build()
        self.saver = tf.train.Saver()
        self.merged_summary = tf.summary.merge_all()
        self.summary = tf.summary.FileWriter(
            self.summary_dir, graph=tf.get_default_graph())
        self.sess.run(tf.global_variables_initializer())

    def _build(self):
        tf.set_random_seed(0)
        self._build_placeholder()
        self._build_embedding()
        self._build_encode()
        self._build_loss()
        self._build_optimizer()
        self._build_summary()
        param_num = sum([np.prod(self.sess.run(tf.shape(v)))
                         for v in tf.trainable_variables()])
        self.logger.info('trainable_variables: %d', param_num)

    def _build_placeholder(self):
        with tf.variable_scope('placeholders'):
            self.center_id = tf.placeholder(
                tf.int32, shape=[None], name='center_id')
            self.center_attr = tf.placeholder(
                tf.float32, shape=[None, self.attr_num], name='center_attr')
            self.neighbor_id = tf.placeholder(
                tf.int32, shape=[None], name='neighbor_id')
            self.neighbor_attr = tf.placeholder(
                tf.float32, shape=[None, self.attr_num], name='neighbor_attr')
            self.Sij = tf.placeholder(tf.float32, shape=[None], name='Sij')

        self.logger.info('placeholder build finish')

    def _build_embedding(self):
        with tf.variable_scope('embedding'):
            self.neighbor_embed_matrix = tf.get_variable(
                'neighbor_embed_matrix', shape=[self.vertex_num, self.embed_size],
                initializer=tf.random_uniform_initializer(minval=-1, maxval=1))

            self.neighbor_bias = tf.zeros([self.vertex_num])

            self.attr_c = tf.get_variable(
                'attr_c', shape=[self.vertex_num, self.embed_size],
                initializer=tf.random_uniform_initializer(minval=-1, maxval=1)
            )

            self.unclip_attr_filter_matrix = tf.get_variable(
                'unclip_attr_filter_matrix', shape=[self.vertex_num, self.attr_num],
                initializer=tf.random_uniform_initializer(minval=-5, maxval=5))
            self.attr_filter_matrix = tf.nn.sigmoid(
                self.unclip_attr_filter_matrix, name='attr_filter_matrix')

        self.logger.info('embedding build finish')

    def _build_encode(self):
        with tf.variable_scope('encode'):
            with tf.device('/cpu:0'):
                self.center_attr_filter = tf.nn.embedding_lookup(
                    self.attr_filter_matrix, self.center_id, name='center_attr_filter')  # (B, attr_num)
                self.neighbor_attr_filter = tf.nn.embedding_lookup(
                    self.attr_filter_matrix, self.neighbor_id, name='neighbor_attr_filter')  # (B, attr_num)
                self.center_attr_c = tf.nn.embedding_lookup(
                    self.attr_c, self.center_id, name='center_attr_c')  # (B, d)
                self.neighbor_encode = tf.nn.embedding_lookup(
                    self.neighbor_embed_matrix, self.neighbor_id, name='neighbor_encode')  # (B, d)

            self.filtered_center_attr = tf.multiply(
                self.center_attr, self.center_attr_filter, name='filtered_center_attr')  # (B, attr_num)
            self.filtered_neighbor_attr = tf.multiply(
                self.neighbor_attr, self.neighbor_attr_filter, name='filtered_neighbor_attr')  # (B, attr_num)

            self.attr_W = tf.get_variable(
                'attr_W', shape=[self.attr_num, self.embed_size],
                initializer=tf.random_uniform_initializer(minval=-1, maxval=1))

            self.center_encode = tf.add(tf.matmul(
                self.filtered_center_attr, self.attr_W), self.center_attr_c, name='center_encode')  # (B, d)

        self.logger.info('encode build finish')

    def _build_loss(self):
        with tf.variable_scope('loss'):
            with tf.variable_scope('struct_loss'):
                self.struct_loss_train = tf.nn.sampled_softmax_loss(self.neighbor_embed_matrix, self.neighbor_bias,
                                                                    labels=tf.reshape(
                                                                        self.neighbor_id, [-1, 1]),
                                                                    inputs=self.center_encode,
                                                                    num_sampled=self.neg_sample_num,
                                                                    num_classes=self.vertex_num)
                self.struct_loss_train = tf.reduce_mean(self.struct_loss_train)

                logits = tf.matmul(self.center_encode, tf.transpose(
                    self.neighbor_embed_matrix))
                logits = tf.nn.bias_add(logits, self.neighbor_bias)
                labels_one_hot = tf.one_hot(tf.reshape(
                    self.neighbor_id, [-1, 1]), self.vertex_num)
                self.struct_loss_eval = tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=labels_one_hot, logits=logits)
                self.struct_loss_eval = tf.reduce_mean(
                    self.struct_loss_eval)  # float32

            with tf.variable_scope('struct_attr_loss'):
                struct_sim = self.Sij  # (B)
                x = tf.nn.l2_normalize(self.filtered_center_attr, 1)
                y = tf.nn.l2_normalize(self.filtered_neighbor_attr, 1)
                attr_sim = 1.0 - tf.losses.cosine_distance(x, y, axis=1)  # (B)
                self.struct_attr_loss = tf.multiply(struct_sim, attr_sim)
                self.struct_attr_loss = tf.reduce_mean(
                    self.struct_attr_loss)  # float32

            self.train_loss = self.struct_loss_train - self.lambda1 * \
                self.struct_attr_loss
            self.eval_loss = self.struct_loss_eval - self.lambda1 * \
                self.struct_attr_loss
            self.grid_search_loss = self.struct_loss_eval
        
        self.logger.info('build loss finish')

    def _build_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.train_loss)
        self.logger.info('build optimizer finish')

    def _build_summary(self):
        with tf.variable_scope('summary'):
            with tf.variable_scope('loss_summary'):
                tf.summary.scalar('train_loss', self.train_loss, family='loss')
                tf.summary.scalar(
                    'struct_loss', self.struct_loss_train, family='loss')
                tf.summary.scalar('struct_attr_loss', -
                                  self.struct_attr_loss, family='loss')
            with tf.variable_scope('neighbor_summary'):
                tf.summary.scalar('max_neighbor', tf.reduce_max(
                    self.neighbor_embed_matrix), family='neighbor')
                tf.summary.scalar('min_neighbor', tf.reduce_min(
                    self.neighbor_embed_matrix), family='neighbor')
                tf.summary.scalar('mean_neighbor', tf.reduce_mean(
                    self.neighbor_embed_matrix), family='neighbor')
            with tf.variable_scope('attr_c_summary'):
                tf.summary.scalar('max_attr_c', tf.reduce_max(
                    self.attr_c), family='attr_c')
                tf.summary.scalar('min_attr_c', tf.reduce_min(
                    self.attr_c), family='attr_c')
                tf.summary.scalar('mean_attr_c', tf.reduce_mean(
                    self.attr_c), family='attr_c')
            with tf.variable_scope('filter_summary'):
                tf.summary.histogram(
                    'unclip_filter', self.unclip_attr_filter_matrix, family='filter')
                tf.summary.histogram(
                    'filter', self.attr_filter_matrix, family='filter')

    def _train_epoch(self, train_batches, epoch):
        total_num = total_loss = 0
        last_num = last_loss = 0
        log_n = 1000
        for idx, batch in enumerate(train_batches, 1):
            feed_dict = {self.center_id: batch['center_id'],
                         self.center_attr: batch['center_attr'],
                         self.neighbor_id: batch['neighbor_id'],
                         self.neighbor_attr: batch['neighbor_attr'],
                         self.Sij: batch['Sij']}
            _, loss = self.sess.run(
                [self.optimizer,  self.train_loss], feed_dict=feed_dict)
            total_num += len(batch['center_id'])
            total_loss += loss
            if idx % log_n == 0:
                self.logger.info('loss from %d to %d : %lf', idx - log_n + 1,
                                 idx, (total_loss - last_loss) / (total_num - last_num))
                last_num = total_num
                last_loss = total_loss
                merged_summary = self.sess.run(
                    self.merged_summary, feed_dict=feed_dict)
                self.summary.add_summary(
                    merged_summary, (epoch-1) * self.train_batch_size + idx)

        return total_loss, total_num

    def train(self):
        for epoch in range(1, self.epochs + 1):
            train_batches = dataset.gen_batches(self.train_file)
            self.logger.info('start training epoch %d',
                             self.epoch_base + epoch)
            epoch_loss, sample_num = self._train_epoch(
                train_batches, self.epoch_base + epoch)
            self.logger.info('loss in epoch %d : %lf',
                             self.epoch_base + epoch, epoch_loss / sample_num)

            self.evaluate(self.epoch_base + epoch)
            self.save(name='model')
    
    def _link_predict_prepare(self, batches, embed_type='center'):
        '''
        embed_type can be concat, sum, center
        '''
        src_embed = []
        dst_embed = []
        edge_label = []
        for batch in batches:
            feed_dict = {self.center_id: batch['node_ids'],
                         self.center_attr: batch['node_attrs'], self.neighbor_id: batch['node_ids']}
            input_embeddings = self.sess.run(
                self.center_encode, feed_dict=feed_dict)  # (nodes, d)
            output_embeddings = self.sess.run(
                self.neighbor_encode, feed_dict=feed_dict)  # (nodes, d)
            if embed_type == 'sum':
                embedding = input_embeddings + output_embeddings  # (nodes, d)
            elif embed_type == 'concat':
                embedding = np.concatenate((input_embeddings, output_embeddings), axis=1)
            elif embed_type == 'center':
                embedding = input_embeddings

            for edge in batch['edges']:
                src_embed.append(embedding[edge[0]])
                dst_embed.append(embedding[edge[1]])
                edge_label.append(edge[2])

        return src_embed, dst_embed, edge_label
        
    def _link_predict(self, valid_batches, test_batches, name='validation', embed_type='center', epoch=None, save=False):
        src_embed, dst_embed, edge_label = self._link_predict_prepare(valid_batches, embed_type)
        train_dataset = {'src_embed':src_embed,
                        'dst_embed':dst_embed,
                        'edge_label':edge_label}

        src_embed, dst_embed, edge_label = self._link_predict_prepare(test_batches, embed_type)
        test_dataset = {'src_embed':src_embed,
                        'dst_embed':dst_embed,
                        'edge_label':edge_label}
        
        evaluate.link_predict(train_dataset, test_dataset, name=name, epoch=epoch, save=save)

    def _node_classify_prepare(self, batches, embed_type='center'):
        embed = []
        node_label = []
        for batch in batches:
            feed_dict = {self.center_id: batch['node_ids'],
                         self.center_attr: batch['node_attrs'], self.neighbor_id: batch['node_ids']}
            input_embeddings = self.sess.run(
                self.center_encode, feed_dict=feed_dict)  # (nodes, d)
            output_embeddings = self.sess.run(
                self.neighbor_encode, feed_dict=feed_dict)  # (nodes, d)
            if embed_type == 'sum':
                embedding = input_embeddings + output_embeddings  # (nodes, d)
            elif embed_type == 'concat':
                embedding = np.concatenate((input_embeddings, output_embeddings), axis=1)
            elif embed_type == 'center':
                embedding = input_embeddings

            embed.extend(list(embedding))
            node_label.extend(batch['node_labels'])
        
        return embed, node_label

    def _node_classify(self, valid_batches, test_batches, name='classification', embed_type='center', epoch=None, save=False):
        embed, node_label = self._node_classify_prepare(valid_batches, embed_type)
        train_dataset = {'embed': embed,
                         'node_label': node_label}
        
        embed, node_label = self._node_classify_prepare(test_batches, embed_type)
        test_dataset = {'embed': embed, 
                        'node_label': node_label}
        
        evaluate.node_classify(train_dataset, test_dataset, name=name, epoch=epoch, save=save)


    def _evaluate_loss(self, evaluate_batches, epoch):
        total_num = total_loss = total_grid_search_loss = 0
        for idx, batch in enumerate(evaluate_batches, 1):
            feed_dict = {self.center_id: batch['center_id'],
                         self.center_attr: batch['center_attr'],
                         self.neighbor_id: batch['neighbor_id'],
                         self.neighbor_attr: batch['neighbor_attr'],
                         self.Sij: batch['Sij']}
            loss, grid_search_loss = self.sess.run([self.eval_loss, self.grid_search_loss], feed_dict=feed_dict)
            total_num += len(batch['center_id'])
            total_loss += loss
            total_grid_search_loss += grid_search_loss
        
        self.logger.info('evaluation loss in epoch %d : %lf',
                            epoch, total_loss / total_num)
        self.logger.info('grid search loss in epoch %d : %lf',
                            epoch, total_grid_search_loss / total_num)
        
        return total_loss / total_num

    def evaluate(self, epoch):
        self.logger.info('start evaluating')
        eval_batches = dataset.gen_batches(self.eval_file)
        eval_loss = self._evaluate_loss(eval_batches, epoch)
        if eval_loss < self.best_loss:
            self.best_loss = eval_loss
            self.save('best_model')

    def predict(self):
        self.logger.info('start predicting')
        if self.task == 'link_predict' or self.task == 'all':
            valid_batches = dataset.gen_batches(self.valid_file_link)
            test_batches = dataset.gen_batches(self.test_file_link)
            self._link_predict(valid_batches, test_batches, name='prediction_sum', embed_type='sum', save='link_sum')
            valid_batches = dataset.gen_batches(self.valid_file_link)
            test_batches = dataset.gen_batches(self.test_file_link)
            self._link_predict(valid_batches, test_batches, name='prediction_concat', embed_type='concat', save='link_concat')
            valid_batches = dataset.gen_batches(self.valid_file_link)
            test_batches = dataset.gen_batches(self.test_file_link)
            self._link_predict(valid_batches, test_batches, name='prediction_center', embed_type='center', save='link_center')
        if self.task == 'node_classify' or self.task == 'all':
            valid_batches = dataset.gen_batches(self.valid_file_node)
            test_batches = dataset.gen_batches(self.test_file_node)
            self._node_classify(valid_batches, test_batches, name='prediction_sum', embed_type='sum', save='node_sum')
            valid_batches = dataset.gen_batches(self.valid_file_node)
            test_batches = dataset.gen_batches(self.test_file_node)
            self._node_classify(valid_batches, test_batches, name='prediction_concat', embed_type='concat', save='node_concat')
            valid_batches = dataset.gen_batches(self.valid_file_node)
            test_batches = dataset.gen_batches(self.test_file_node)
            self._node_classify(valid_batches, test_batches, name='prediction_center', embed_type='center', save='node_center')

    def save(self, name='model', best_loss=None):
        '''
        name can be model or best_model
        model: for future training
        best_model: for testing
        '''
        self.saver.save(self.sess, os.path.join(self.model_dir, name))
        self.logger.info('model saved to %s as %s', self.model_dir, name)
        if name == 'best_model':
            with open(os.path.join(self.model_dir, 'best_loss.pk'), 'wb') as fout:
                pk.dump(self.best_loss, fout)
            self.logger.info('best loss saved')

    def restore(self, name='model'):
        '''
        name can be model or best_model
        model: continue to train
        best_model: not to train, but to get embedding for further test
        '''
        self.saver.restore(self.sess, os.path.join(self.model_dir, name))
        self.logger.info('model restored from %s as %s', self.model_dir, name)
        if name == 'model':
            with open(os.path.join(self.model_dir, 'best_loss.pk'), 'rb') as fin:
                self.best_loss = pk.load(fin)
            self.logger.info('best loss restored')
