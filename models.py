import tensorflow as tf

def weighted_mse():
    """
    Weighted MSE loss function for the model.
    Returns a loss function that can be used during model compilation.
    """
    def loss(y_true, y_pred):
        # Calculate MSE
        mse = tf.square(y_true - y_pred)
        
        # Determine the number of timesteps in the sequence
        timesteps = tf.shape(y_true)[1]
        timestep_weights = tf.range(1, timesteps + 1, dtype=tf.float32)
        timestep_weights = tf.reshape(timestep_weights, (1, timesteps, 1, 1, 1))  # Shape to broadcast
        
        # Apply the timestep weights to each timestep's MSE
        weighted_mse = mse * timestep_weights
        # Reduce mean to get the final loss
        return weighted_mse    
    return loss

def csi(y_true, y_pred, threshold=20):
    """CSI for each of the 12 timesteps separately"""
    y_pred_binary = tf.cast(y_pred > threshold, tf.float32)
    y_true_binary = tf.cast(y_true > threshold, tf.float32)
    
    csi_values = []
    for t in range(12):
        y_t = y_true_binary[:, t]
        p_t = y_pred_binary[:, t]
        
        tp = tf.reduce_sum(y_t * p_t)
        fn = tf.reduce_sum(y_t * (1 - p_t))
        fp = tf.reduce_sum((1 - y_t) * p_t)
        
        csi_t = tp / tf.maximum(tp + fn + fp, 1.0)
        csi_values.append(csi_t)
    csi = csi_values[-1]
    return csi
