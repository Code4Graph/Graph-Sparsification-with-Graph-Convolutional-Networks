from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from models import GCN, MLP

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'


flags.DEFINE_string('learning_rate', '0.001', 'Initial learning rate.')
flags.DEFINE_string('epochs', '10', 'Number of epochs to train.')
flags.DEFINE_string('ADMM', '3', 'Number of epochs to train ADMM.')
flags.DEFINE_string('prune_ratio', '10.0', 'prune ratio.')
flags.DEFINE_string('count', '0', 'count.')
flags.DEFINE_string('target_acc', '0.80', 'target_acc')
flags.DEFINE_integer('num_layers', 2, 'Number of units in hidden layer 1.')


flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

learning_rate = float(FLAGS.learning_rate)
prune_ratio = float(FLAGS.prune_ratio)
ADMM_times = int(FLAGS.ADMM)
Total_epochs = int(FLAGS.epochs)
adj_count = int(FLAGS.count)
target_acc = float(FLAGS.target_acc)
print("learning_rate: ", learning_rate)
print("prune_ratio: ", prune_ratio)
print("ADMM_times: ", ADMM_times)
print("Total_epochs: ", Total_epochs)
print("target acc: ", target_acc)
# Load data

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
weights = []
#load weights.npy
for i in range(FLAGS.num_layers):
    weights.append(np.load('w'+str(i+1)+'.npy'))


# Some preprocessing
features = preprocess_features(features)

if FLAGS.model == 'gcn':
    support = np.array(preprocess_adj(adj), dtype=float)
    num_supports = 1
    num_weights = FLAGS.num_layers
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    num_weights = 2
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    num_weights = 2
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define placeholders
placeholders = {
    'weights': [tf.placeholder(tf.float32) for _ in range(num_weights)],
    'support': support,
    'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    'A': tf.placeholder(tf.float32, shape=support.shape),
    'B': tf.placeholder(tf.float32, shape=support.shape),
    'C': tf.placeholder(tf.float32, shape=support.shape),
    'D': tf.placeholder(tf.float32, shape=support.shape)
}
model = model_func(placeholders=placeholders, input_dim=features[2][1], num_layers=FLAGS.num_layers, logging=False)
admm_loss = model.loss
loss = model.loss
mytrainer = tf.train.AdamOptimizer(learning_rate=float(FLAGS.learning_rate))
admm_opt_op_adj = None
admm_opt_op_w = None


print("construct admm training")
partial_adj_mask = partial_mask(adj)
support1 = model.vars['gcn/graphconvolution_1_adj_vars/adj:0']
support2 = model.vars['gcn/graphconvolution_2_adj_vars/adj:0']


rho = 1e-3
A = placeholders['A']
B = placeholders['B']
C = placeholders['C']
D = placeholders['D']
adj_variable = tf.get_collection('adj')
admm_loss = admm_loss + rho * (tf.nn.l2_loss((support1 - np.identity(support1.shape[0])) - A + B) + tf.nn.l2_loss((support2 - np.identity(support2.shape[0])) - C + D))
adj_grads = mytrainer.compute_gradients(admm_loss, adj_variable)
adj_grads = update_gradients_adj(adj_grads, partial_adj_mask)
admm_opt_op_adj = mytrainer.apply_gradients(adj_grads)
highestacc = 0



# Define model evaluation function
def evaluate(features, weights, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, weights, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

cost_val = []
# Train model
Z1 = initialize(support)
Z2 = initialize(support)
U1 = np.zeros_like(Z1)
U2 = np.zeros_like(Z2)
non_zero_idx = np.count_nonzero(adj.toarray())

for j in range(ADMM_times):
    for epoch in range(Total_epochs):
        t = time.time()

        feed_dict = construct_feed_dict(features, weights, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        feed_dict.update({placeholders['A']: Z1})
        feed_dict.update({placeholders['B']: U1})
        feed_dict.update({placeholders['C']: Z2})
        feed_dict.update({placeholders['D']: U2})

        outs = sess.run([admm_opt_op_adj, admm_loss, model.accuracy], feed_dict=feed_dict)

        cost, acc, duration = evaluate(features, weights, y_val, val_mask, placeholders)
        cost_val.append(cost)
        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(
                cost_val[-(FLAGS.early_stopping + 1):-1]):
            print("Early stopping...")
            break

    adj1 = sess.run(model.vars['gcn/graphconvolution_1_adj_vars/adj:0'])
    Z1 = adj1 - np.identity(adj1.shape[0]) + U1
    Z1 = prune_adj2(Z1, non_zero_idx, percent=prune_ratio) - np.identity(adj1.shape[0])
    U1 = U1 + (adj1 - np.identity(adj1.shape[0])) - Z1

    adj2 = sess.run(model.vars['gcn/graphconvolution_2_adj_vars/adj:0'])
    Z2 = adj2 - np.identity(adj2.shape[0]) + U2
    Z2 = prune_adj2(Z2, non_zero_idx, percent=prune_ratio) - np.identity(adj2.shape[0])
    U2 = U2 + (adj2 - np.identity(adj2.shape[0])) - Z2

# finally project and prune to 0
adj1 = sess.run(model.vars['gcn/graphconvolution_1_adj_vars/adj:0'])
adj2 = sess.run(model.vars['gcn/graphconvolution_2_adj_vars/adj:0'])
adj1 = prune_adj2(adj1 - np.identity(adj1.shape[0]), non_zero_idx, percent=prune_ratio)
adj2 = prune_adj2(adj2 - np.identity(adj2.shape[0]), non_zero_idx, percent=prune_ratio)
sess.run(model.vars['gcn/graphconvolution_1_adj_vars/adj:0'].assign(adj1))
sess.run(model.vars['gcn/graphconvolution_2_adj_vars/adj:0'].assign(adj2))

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(features, weights, y_test, test_mask, placeholders)
print("Finally Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

# Test sparse adj
cur_adj1 = sess.run(model.vars['gcn/graphconvolution_1_adj_vars/adj:0'])
cur_adj2 = sess.run(model.vars['gcn/graphconvolution_2_adj_vars/adj:0'])
print("finish L1 training, num of edges *2 + diag in adj1:", np.count_nonzero(cur_adj1))
print("finish L1 training, num of edges * 2 + diag in adj2:", np.count_nonzero(cur_adj2))
print("symmetry result adj1: ", testsymmetry(cur_adj1))
print("symmetry result adj2: ", testsymmetry(cur_adj2))
print("is equal of two adj", isequal(cur_adj1, cur_adj2))

# store adj which accuracy larger than your define
if test_acc >= target_acc:
    export_dict = {'adj_' + str(adj_count): cur_adj1}
    for filename, var in export_dict.items():
        np.save(filename, var)
    print("saving")
