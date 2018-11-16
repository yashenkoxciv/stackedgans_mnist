''' Stacked Independent training '''
import numpy as np
import tensorflow as tf

H1_DIM = 500
Z_DIM = 100

def op_name(name, op):
    return '_'.join([name, op])

class StackableGAN:
    def __init__(self, z_dim, h0, gh0, h1, f_cb, name):
        self.name = name
        self.z = tf.placeholder(tf.float32, z_dim, name=op_name(name, 'z'))
        self.h0 = h0 # features from encoder
        self.gh0 = gh0 # features from previous GAN
        self.ph0 = tf.placeholder(tf.float32, self.h0.get_shape().as_list(), name=op_name(name, 'ph0'))
        self.h1 = h1
        
        # train model
        self.gh1 = self.g(self.z, self.h0)
        
        self.d_fake = self.d(self.gh1)
        self.d_real = self.d(h1)
        
        self.rh0 = f_cb(self.gh1) # try to reconstruct given (h0) features
        self.rz = self.q(self.gh1) # try to reconstruct input (z) noise
        
        # inference model
        self.jgh1 = self.g(self.z, self.gh0)
        
        # corrupted inference
        self.xgh1 = self.g(self.z, self.ph0)
    
    def g(self, z, h):
        raise NotImplementedError()
    
    def d(self, h1):
        raise NotImplementedError()
    
    def q(self, h1):
        raise NotImplementedError()
    
    def build(self, real_label_dev, fake_label_dev, disc_con, mg, me, mc):
        d_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(self.d_real) + tf.random_uniform(tf.shape(self.d_real), minval=-real_label_dev, maxval=0.0),
                    logits=self.d_real,
                    name=op_name(self.name, 'd_real_loss')
            )
        )
        d_fake_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.zeros_like(self.d_fake) + tf.random_uniform(tf.shape(self.d_fake), minval=0.0, maxval=fake_label_dev),
                        logits=self.d_fake,
                        name=op_name(self.name, 'd_fake_loss')
                )
        )
        self.d_loss = tf.add(
                d_real_loss, d_fake_loss,
                name=op_name(self.name, 'd_loss'))
        
        self.g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=tf.ones_like(self.d_fake),
                        logits=self.d_fake,
                        name=op_name(self.name, 'g_loss')
                )
        )
        
        self.ent_loss = tf.reduce_mean(
                tf.square(self.z - self.rz),
                name=op_name(self.name, 'ent_loss')
        )
        if disc_con:
            self.con_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(
                            labels=self.h0, logits=self.rh0,
                            name=op_name(self.name, 'con_disc_loss')
                    )
            )
        else:
            self.con_loss = tf.reduce_mean(
                    tf.square(self.h0 - self.rh0),
                    name=op_name(self.name, 'con_con_loss')
            )
        all_variables = tf.trainable_variables()
        self.g_vars = [v for v in all_variables if v.name.startswith(op_name(self.name, 'g'))]
        self.q_vars = [v for v in all_variables if v.name.startswith(op_name(self.name, 'q'))]
        self.d_vars = [v for v in all_variables if v.name.startswith(op_name(self.name, 'd'))]
        
        self.d_opt = tf.train.AdamOptimizer().minimize(self.d_loss, var_list=self.d_vars)
        self.g_opt = tf.train.AdamOptimizer().minimize(
                mg*self.g_loss + me*self.ent_loss + mc*self.con_loss,
                var_list=self.g_vars + self.q_vars)
      
class GAN1(StackableGAN):
    def g(self, z, h):
        if hasattr(self, '_g_reuse'):
            self._g_reuse = True
        else:
            self._g_reuse = False
        h1 = tf.layers.dense(tf.concat((z, h), 1), 500, tf.nn.leaky_relu, name=op_name(self.name, 'g_h1'), reuse=self._g_reuse)
        xg = tf.layers.dense(h1, H1_DIM, tf.nn.tanh, name=op_name(self.name, 'g_xg'), reuse=self._g_reuse)
        return xg
    
    def d(self, h):
        if hasattr(self, '_d_reuse'):
            self._d_reuse = True
        else:
            self._d_reuse = False
        h1 = tf.layers.dense(h, 100, tf.nn.leaky_relu, name=op_name(self.name, 'd_h1'), reuse=self._d_reuse)
        y = tf.layers.dense(h1, 1, None, name=op_name(self.name, 'd_y'), reuse=self._d_reuse)
        return y
    
    def q(self, h):
        if hasattr(self, '_q_reuse'):
            self._q_reuse = True
        else:
            self._q_reuse = False
        h1 = tf.layers.dense(h, 100, tf.nn.leaky_relu, name=op_name(self.name, 'q_h1'), reuse=self._q_reuse)
        zr = tf.layers.dense(h1, Z_DIM, None, name=op_name(self.name, 'q_z'), reuse=self._q_reuse)
        return zr

class GAN2(StackableGAN):
    def g(self, z, h):
        if hasattr(self, '_g_reuse'):
            self._g_reuse = True
        else:
            self._g_reuse = False
        h1 = tf.layers.dense(tf.concat((z, h), 1), 3*3*20, tf.nn.leaky_relu, name=op_name(self.name, 'g_h1'), reuse=self._g_reuse)
        c0 = tf.reshape(h1, (-1, 3, 3, 20))
        c1 = tf.layers.conv2d_transpose(
                c0, filters=60, kernel_size=3, activation=tf.nn.leaky_relu, strides=2,
                padding='valid', name=op_name(self.name, 'g_c1'), reuse=self._g_reuse
        )
        c2 = tf.layers.conv2d_transpose(
                c1, filters=30, kernel_size=3, activation=tf.nn.leaky_relu, strides=2,
                padding='same', name=op_name(self.name, 'g_c2'), reuse=self._g_reuse
        )
        c3 = tf.layers.conv2d_transpose(
                c2, filters=30, kernel_size=3, activation=tf.nn.leaky_relu, strides=2,
                padding='same', name=op_name(self.name, 'g_c3'), reuse=self._g_reuse
        )
        xg = tf.layers.conv2d_transpose(
                c3, filters=1, kernel_size=3, activation=tf.nn.tanh, strides=1,
                padding='same', name=op_name(self.name, 'g_c4'), reuse=self._g_reuse
        )
        return xg
    
    def d(self, h):
        if hasattr(self, '_d_reuse'):
            self._d_reuse = True
        else:
            self._d_reuse = False
        c1 = tf.layers.conv2d(
            h, filters=30, kernel_size=4,
            activation=tf.nn.leaky_relu, strides=2, padding='same',
            name=op_name(self.name, 'd_c1'), reuse=self._d_reuse)
        c2 = tf.layers.conv2d(
                c1, filters=60, kernel_size=3,
                activation=tf.nn.leaky_relu, strides=2, padding='same',
                name=op_name(self.name, 'd_c2'), reuse=self._d_reuse)
        c3 = tf.layers.conv2d(
                c2, filters=60, kernel_size=3,
                activation=tf.nn.leaky_relu, strides=2, padding='valid',
                name=op_name(self.name, 'd_c3'), reuse=self._d_reuse) # 3, 3, 60
        h0 = tf.layers.flatten(c3)
        h1 = tf.layers.dense(h0, 100, tf.nn.leaky_relu, name=op_name(self.name, 'd_h1'), reuse=self._d_reuse)
        y = tf.layers.dense(h1, 1, None, name=op_name(self.name, 'd_y'), reuse=self._d_reuse)
        return y
    
    def q(self, h):
        if hasattr(self, '_q_reuse'):
            self._q_reuse = True
        else:
            self._q_reuse = False
        c1 = tf.layers.conv2d(
            h, filters=30, kernel_size=5,
            activation=None, strides=1, padding='same',
            name=op_name(self.name, 'q_c1'), reuse=self._q_reuse)
        a1 = tf.nn.leaky_relu(c1)
        p1 = tf.layers.max_pooling2d(a1, 2, 4)
        c2 = tf.layers.conv2d(
                p1, filters=60, kernel_size=4,
                activation=None, strides=1, padding='same',
                name=op_name(self.name, 'q_c2'), reuse=self._q_reuse)
        a2 = tf.nn.leaky_relu(c2)
        p2 = tf.layers.average_pooling2d(a2, 2, 4)
        h0 = tf.layers.flatten(p2)
        h1 = tf.layers.dense(h0, 500, tf.nn.tanh, name=op_name(self.name, 'q_h1'), reuse=self._q_reuse)
        zr = tf.layers.dense(h1, Z_DIM, None, name=op_name(self.name, 'q_y'), reuse=self._q_reuse)
        return zr

def z_next_batch(batch_size):
    return np.random.randn(batch_size, Z_DIM)*1.0

def encoder(x):
    if hasattr(encoder, 'reuse'):
        encoder.reuse = True
    else:
        encoder.reuse = False
    c1 = tf.layers.conv2d(
            x, filters=60, kernel_size=5,
            activation=None, strides=1, padding='same',
            name='encoder_c1', reuse=encoder.reuse)
    a1 = tf.nn.leaky_relu(c1)
    p1 = tf.layers.max_pooling2d(a1, 2, 4)
    
    c2 = tf.layers.conv2d(
            p1, filters=120, kernel_size=4,
            activation=None, strides=1, padding='same',
            name='encoder_c2', reuse=encoder.reuse)
    a2 = tf.nn.leaky_relu(c2)
    p2 = tf.layers.average_pooling2d(a2, 2, 2)
    
    h0 = tf.layers.flatten(p2)
    
    h1 = tf.layers.dense(h0, H1_DIM, tf.nn.tanh, name='encoder_h1', reuse=encoder.reuse)
    
    y = tf.layers.dense(h1, 10, None, name='encoder_y', reuse=encoder.reuse)
    return y, h1

def gen_y(fake_h1):
    if hasattr(encoder, 'reuse'):
        if not encoder.reuse:
            y_fake = tf.layers.dense(fake_h1, 10, None, name='encoder_y', reuse=not encoder.reuse)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()
    return y_fake

def gen_h1(fake_x):
    return encoder(fake_x)[1]

if __name__ == '__main__':
    z1 = tf.placeholder(tf.float32, (None, Z_DIM), name='g1_input_noise')
    z2 = tf.placeholder(tf.float32, (None, Z_DIM), name='g2_input_noise')
    y = tf.placeholder(tf.float32, (None, 10), name='input_labels')
    x = tf.placeholder(tf.float32, (None, 28, 28, 1), name='input_images')
    
    # encoder
    yh_logits, h1 = encoder(x)
    yh = tf.nn.softmax(yh_logits)
    
    # encoder (classification)
    encoder_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                    labels=y, logits=yh_logits
            )
    )
    enc_opt = tf.train.AdamOptimizer().minimize(encoder_loss)
    # auxilary metrics
    correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(yh_logits), 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # stacked GANs
    g1 = GAN1((None, Z_DIM), yh, y, h1, gen_y, 'gan1')
    g2 = GAN2((None, Z_DIM), h1, g1.gh1, x, gen_h1, 'gan2')
    
    g1.build(0.1, 0.1, True, 1, 1, 1)
    g2.build(0.1, 0.1, False, 1, 1, 1)
    
    sgan = [g1, g2]
    d_opts = [g.d_opt for g in sgan]
    g_opts = [g.g_opt for g in sgan]
    
    # summary
    for g in sgan:
        tf.summary.scalar(op_name(g.name, 'd_loss'), g.d_loss)
        tf.summary.scalar(op_name(g.name, 'g_loss'), g.g_loss)
        tf.summary.scalar(op_name(g.name, 'ent_loss'), g.ent_loss)
        tf.summary.scalar(op_name(g.name, 'con_loss'), g.con_loss)
    
    tf.summary.scalar('accuracy', accuracy)
    
    tf.summary.histogram('h1', h1)
    tf.summary.histogram('gh1', g1.gh1)
    
    tf.summary.image('x', x, 3)
    tf.summary.image('gx', g2.gh1, 3)
    tf.summary.image('joint_gx', g2.jgh1, 3)
    all_summary = tf.summary.merge_all()
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    writer_train = tf.summary.FileWriter('summary', sess.graph)
    saver = tf.train.Saver()
    
    # data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    y_train, y_test = tf.keras.utils.to_categorical(y_train, 10), tf.keras.utils.to_categorical(y_test, 10)
    x_train = ((x_train - 128.0) / 128.0).reshape((-1, 28, 28, 1))
    x_test = ((x_test - 128.0) / 128.0).reshape((-1, 28, 28, 1))
    
    def next_batch(xs, ys, batch_size):
        idx = np.random.randint(0, xs.shape[0], batch_size)
        x_batch = xs[idx]
        y_batch = ys[idx]
        return x_batch, y_batch
    
    epochs = 15
    batch_size = 100
    batches = x_train.shape[0] // batch_size
    batch_end = epochs*batches
    batch_step = 0
    summary_step = 0
    
    for epoch in range(1, epochs + 1):
        for batch in range(1, batches + 1):
            x_batch, y_batch = next_batch(x_train, y_train, batch_size)
            
            sess.run(enc_opt, {x: x_batch, y: y_batch})
            
            batch_step += 1
            print('\repoch {0} {1:3.0f} %'.format(
                    epoch, batch / batches * 100
                    ), end='', flush=True
            )
            
            if batch_step % 100 == 0:
                train_summary_str = sess.run(all_summary, {
                    x: x_batch, y: y_batch,
                    g1.z: z_next_batch(batch_size), g2.z: z_next_batch(batch_size)
                })
                writer_train.add_summary(train_summary_str, summary_step)
                #writer_train.flush()
                summary_step += 1
    
    print('\rEncoder training done', ' '*25, flush=True)
    
    epochs = 100
    batch_size = 100
    batches = x_train.shape[0] // batch_size
    batch_end = epochs*batches
    batch_step = 0
    summary_step = summary_step
    
    for epoch in range(1, epochs + 1):
        for batch in range(1, batches + 1):
            x_batch, y_batch = next_batch(x_train, y_train, batch_size)
            
            sess.run(d_opts, {
                    x: x_batch, y: y_batch,
                    g1.z: z_next_batch(batch_size), g2.z: z_next_batch(batch_size)
            })
            sess.run(g_opts, {
                    x: x_batch, y: y_batch,
                    g1.z: z_next_batch(batch_size), g2.z: z_next_batch(batch_size)
            })
            
            batch_step += 1
            print('\repoch {0} {1:3.0f} %'.format(
                    epoch, batch / batches * 100
                    ), end='', flush=True
            )
            
            if batch_step % 100 == 0:
                train_summary_str = sess.run(all_summary, {
                    x: x_batch, y: y_batch,
                    g1.z: z_next_batch(batch_size), g2.z: z_next_batch(batch_size)
                })
                writer_train.add_summary(train_summary_str, batch_step)
                #writer_train.flush()
                summary_step += 1
    
    print('\rStack training done', ' '*25, flush=True)
    #input()
    saver.save(sess, 'model/model.ckpt')
    writer_train.close()
    sess.close()
    

    
