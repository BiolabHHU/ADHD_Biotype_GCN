import numpy as np
import tensorflow as tf
import os
from tensorflow.keras import layers
from spektral.layers import GCNConv
num_K = 3


def extract_upper_tri_vector(input_tensor):
    n = input_tensor.shape[1]
    mask = tf.linalg.band_part(tf.ones((n, n), dtype=tf.bool), 0, -1)
    mask = tf.linalg.set_diag(mask, tf.zeros([n], dtype=tf.bool))
    return tf.boolean_mask(input_tensor, mask, axis=1)

def edge_norm_tf(edge):

    num_nodes = tf.shape(edge)[-1]
    a = tf.eye(num_nodes, dtype=tf.float32)
    edge_plus_a = edge + a
    adj = tf.abs(edge_plus_a)
    deg = tf.reduce_sum(adj, axis=-1, keepdims=False)
    deg_inv = tf.math.reciprocal_no_nan(deg)
    deg_inv_sqrt = tf.pow(deg, -0.5)

    deg_inv_sqrt_diag = tf.linalg.diag(deg_inv_sqrt)

    edge_norm_intermediate = tf.matmul(deg_inv_sqrt_diag, adj)
    edge_norm = tf.matmul(edge_norm_intermediate, deg_inv_sqrt_diag)

    return edge_norm


def load_ADHD_data():
    base_path = r"D:\gjh\ADHD"
    fc_folder = os.path.join(base_path, "Data_FC")
    alff_folder = os.path.join(base_path, "Data_ALFF")
    site_prefixes = ("KKI", "NI", "NYU", "PU")

    all_FC, all_ALFF, all_labels = [], [], []
    for prefix in site_prefixes:
        all_FC.append(np.load(os.path.join(fc_folder, f"{prefix}_FC.npy")))
        all_ALFF.append(np.load(os.path.join(alff_folder, f"{prefix}_ALFF.npy")))
        all_labels.append(np.load(os.path.join(fc_folder, f"{prefix}_label.npy")))

    FC_data = np.concatenate(all_FC)
    ALFF_data = np.concatenate(all_ALFF)
    labels = np.concatenate(all_labels)
    l = labels[labels != 2]
    l[l==3]=2  # Droup out ADHD-HI
    fc =  FC_data[labels != 2]
    alff  = ALFF_data[labels != 2]
    # fc = edge_norm_tf(fc)
    return fc, alff, l



class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.gcn1 = GCNConv(6, activation='relu')
        self.gcn2 = GCNConv(6, activation='relu')
        self.fc1_0 = layers.Dense(220, activation='relu', kernel_initializer='he_normal')
        self.ln1 = layers.LayerNormalization()
        self.fc1_1 = layers.Dense(140, activation='relu', kernel_initializer='he_normal')
        self.ln2 = layers.LayerNormalization()
        self.fc2_mu = layers.Dense(num_K * 20)
        self.fc2_logvar = layers.Dense(num_K * 20)
        self.DB1 = tf.keras.Sequential([
            layers.Dense(220, kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.DB2 = tf.keras.Sequential([
            layers.Dense(140, kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

    def call(self, alff, fc):
        fc = edge_norm_tf(fc)
        h1 = self.gcn1([alff, fc])
        h1 = self.gcn2([h1, fc])
        h1 = tf.reshape(h1, (h1.shape[0], -1))
        h1 = self.DB1(h1)
        h1 = self.DB2(h1)
        mu = self.fc2_mu(h1)
        logvar = self.fc2_logvar(h1)
        mu = tf.reshape(mu, (-1, num_K, 20))
        logvar = tf.reshape(logvar, (-1, num_K, 20))
        return h1, mu, logvar


class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.gcn1 = GCNConv(3, activation='relu')
        self.gcn2 = GCNConv(3, activation='relu')
        self.fc1_1 = layers.Dense(140, activation='relu', kernel_initializer='he_normal')
        self.fc1_2 = layers.Dense(220, activation='relu', kernel_initializer='he_normal')
        self.fc1_3 = layers.Dense(348, activation='relu', kernel_initializer='he_normal')
        self.ln = layers.LayerNormalization()
        self.DB1 = tf.keras.Sequential([
            layers.Dense(140, kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.DB2 = tf.keras.Sequential([
            layers.Dense(220, kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])
        self.DB3 = tf.keras.Sequential([
            layers.Dense(348, kernel_initializer='he_normal'),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

    def call(self, z, fc):
        fc = edge_norm_tf(fc)
        h1 = self.DB1(z)
        h1 = self.DB2(h1)
        h1 = self.DB3(h1)
        h1 = tf.reshape(h1, [h1.shape[0], 58, 6])
        h1 = self.gcn1([h1, fc])
        h1 = self.gcn2([h1, fc])
        return h1


def residual_block(filters, apply_dropout=True):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Dense(filters, kernel_initializer='he_normal',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.2))
    result.add(tf.keras.layers.ReLU())

    result.add(tf.keras.layers.Dense(filters, kernel_initializer='he_normal',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.2))
    result.add(tf.keras.layers.ReLU())
    return result

class Classifier_resblock(tf.keras.Model):
    def __init__(self):
        super(Classifier_resblock,self).__init__()
        self.fc1 = layers.Dense(10,kernel_initializer='he_normal',activation='relu')
        self.block_stack_1 = residual_block(10, apply_dropout=False)
        self.block_stack_2 = residual_block(10, apply_dropout=False)
        self.fc3 = layers.Dense(num_K,activation='softmax')

    def call(self,h1):
        h1 = self.fc1(h1)
        h_ = h1
        h2 = self.block_stack_1(h1)
        h1 = h2 + h1
        h3 = self.block_stack_2(h1)
        h1 = h3 + h1
        # h1 = h_ + h1
        y_out = self.fc3(h1)
        return y_out


class VAE_ADHD_subtype(tf.keras.Model):
    def __init__(self):
        super(VAE_ADHD_subtype,self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.classifier_resblock = Classifier_resblock()
        self.relu = tf.keras.layers.ReLU()

    def reparameterize(self,mu,logvar):
        std = tf.exp(0.5 * logvar)
        eps = tf.random.normal(shape=tf.shape(std))
        return mu + eps * std

    def forward_1(self,alff,fc,flag,mu_ave,label):
        fc = edge_norm_tf(fc)
        _, mu, logvar = self.encoder.call(alff,fc)
        mu_ADHD = mu[:,1:num_K,:]
        mu_ADHD = tf.boolean_mask(mu_ADHD, label>0,axis=0)

        if flag == 1:
            distances = tf.norm( mu_ADHD - tf.reduce_mean(mu_ADHD,axis=0,keepdims=True),axis=2)
        else:
            tmp_ = mu_ave[1:num_K]
            tmp_ = tf.expand_dims(tmp_, axis=0)
            distances = tf.norm(mu_ADHD - tmp_, axis=2)

        closest_index = tf.argmin(distances,axis=1)
        closest_idx_all = tf.zeros([alff.shape[0],], dtype=tf.int64)

        indices = tf.reshape(tf.where(label > 0),[-1])
        closest_idx_all = tf.tensor_scatter_nd_update(closest_idx_all, tf.reshape(indices, (-1, 1)), closest_index+1)


        if flag == 1:
            closest_idx_all = label

        mu_ave_ = tf.zeros([num_K, 20], dtype=tf.float32)
        for i in range (num_K):
            mask = tf.equal(closest_idx_all, i)
            class_sample = tf.boolean_mask(mu, mask,axis=0)
            test = class_sample[:,i:i+1,:]
            mu_ave_ = tf.tensor_scatter_nd_update(mu_ave_, indices=tf.constant([[i]]), updates=tf.reduce_mean(test, axis=0))#切片操作

        if flag == 1:
            mu_ave = mu_ave_
        else:
            mu_ave = mu_ave_

        mu = tf.gather(mu, tf.cast(closest_idx_all, dtype=tf.int64), axis=1, batch_dims=1)
        logvar = tf.gather(logvar, tf.cast(closest_idx_all, dtype=tf.int64), axis=1, batch_dims=1)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decoder.call(z,fc)
        reconstructed_fc = self.relu(tf.matmul(reconstructed_x, tf.transpose(reconstructed_x, perm=[0, 2, 1])))
        class_output = self.classifier_resblock.call(z)
        return reconstructed_x,reconstructed_fc,mu,logvar,class_output,closest_idx_all,mu_ave,z



def loss_function_2(reconstructed_x,reconstructed_fc,alff,fc,label,class_output,mu,logvar):
    fc_abs = tf.abs(fc)
    result_fc = extract_upper_tri_vector(fc_abs)
    result_re_fc = extract_upper_tri_vector(reconstructed_fc)
    loss_mse1 = tf.keras.losses.MeanSquaredError()(reconstructed_x, alff)
    loss_mse2 = tf.keras.losses.MeanSquaredError()(result_re_fc, result_fc)

    loss_mse = 0.5 * (loss_mse1 + loss_mse2)
    loss_kld = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar))  # 012的mu对应的损失
    loss_ce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(label, class_output)
    return tf.cast(loss_mse, tf.float64), tf.cast(loss_kld, tf.float64),tf.cast(loss_ce, tf.float64)

def loss_function_diff(mu_ave):
    mu_ave_ = mu_ave-mu_ave[0]
    mu_ave_ = mu_ave_[1:num_K]
    energy_ =tf.norm(mu_ave_, ord=2, axis=1, keepdims=True)
    mu_ave_En = mu_ave_/energy_
    tmp_ = tf.matmul(mu_ave_En,mu_ave_En,transpose_b=True)
    MU = tf.reduce_sum(tf.abs(tmp_)) - tf.reduce_sum(tf.linalg.diag_part(tmp_))
    MU_dif_mse = tf.keras.losses.MeanSquaredError()(mu_ave_, tf.zeros([mu_ave_.shape[0], mu_ave_.shape[1]], dtype=tf.float32) )
    return tf.cast(MU, tf.float64),tf.cast(MU_dif_mse, tf.float64)

def loss_function_diff2(mu_ave):
    mu_ave = extract_upper_tri_vector(mu_ave)
    mu_ave_ = mu_ave-mu_ave[0]
    mu_ave_ = mu_ave_[1:num_K]
    energy_ =tf.norm(mu_ave_, ord=2, axis=1, keepdims=True)
    mu_ave_En = mu_ave_/energy_
    tmp_ = tf.matmul(mu_ave_En,mu_ave_En,transpose_b=True)
    MU = tf.reduce_sum(tf.abs(tmp_)) - tf.reduce_sum(tf.linalg.diag_part(tmp_))
    MU_dif_mse = tf.keras.losses.MeanSquaredError()(mu_ave_, tf.zeros([mu_ave_.shape[0], mu_ave_.shape[1]], dtype=tf.float32) )
    return tf.cast(MU, tf.float64),tf.cast(MU_dif_mse, tf.float64)








