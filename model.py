import keras 
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.numpy_ops import np_config

class nn_model:
    def __init__(self,optimiser):

        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(4,)),
                tf.keras.layers.Dense(128, activation="softplus"),
                tf.keras.layers.Dense(128, activation="softplus"),
                tf.keras.layers.Dense(1,)
            ]
            )
        
        self.optimiser = optimiser
        

    def train(self,x_train,xt_train,epochs,bathsize = 1):

        """  
        x_train  = [theta_1    , theta_2    , dt_theta_1    , dt_theta_2]    = [q_1    , q_2   , dt_q_1    , dt_q_2]
        xt_train = [dt_theta_1 , dt_theta_2 , dt_dt_theta_1 , dt_dt_theta_2] = [dt_q_1 , dt_q_2, dt_dt_q_1 , dt_dt_q_2]
        y_pred   = L !
        """        

        q , qt = np.split(x_train, 2) # q in N x 2 ; qt in N x 2        

        q  = tf.convert_to_tensor(q)
        qt = tf.convert_to_tensor(qt)

        for epoch in range(epochs):
            
            with tf.GradientTape() as g1:
                g1.watch(q)
                g1.watch(qt)
                with tf.GradientTape(persistent = True) as g2:
                    g2.watch(q)
                    g2.watch(qt)
                    lagrangian = self.model(x_train, training=True)   # lagrangian  in N x 1
                gradient_q  = g2.gradient(lagrangian, q)         # gradient_q  in N x 2
                gradient_qt = g2.gradient(lagrangian, qt)        # gradient_qt in N x 2
            
            hessian_qt_qt = g1.gradient(gradient_qt, qt) # hessian_qt_qt  in N x 1
            hessian_q_qt  = g1.gradient(gradient_q, qt)  # hessian_q_qt   in N x 1
                
            #reg = alpha * tf.eye()
            qtt = tf.linalg.inv(hessian_qt_qt) @ (gradient_q - hessian_q_qt@qt)
           
            predicted_values = np.concatenate([qt, qtt])
            loss  = tf.math.reduce_mean(tf.math.square(predicted_values - xt_train))

            # Minimise the combined loss with respect to the neural networks weights
            gradients = tf.gradients(loss, self.model.trainable_weights)
            self.optimiser.apply_gradients(zip(gradients, self.model.trainable_weights))

            
    def predict(self):
        pass