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

def csi(y_true, y_pred):
    """
    Critical Success Index (CSI) metric for model evaluation.
    """
    # Calculate true positives, possible positives, and false positives
    # everything above 5 is considered a positive
    threshold = 5
    y_pred = tf.cast(y_pred > threshold, dtype=y_pred.dtype)
    y_true = tf.cast(y_true > threshold, dtype=y_true.dtype)
    
    # Calculate True Positives (TP), False Negatives (FN), and False Positives (FP)
    true_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 1)), dtype=y_pred.dtype))
    false_negatives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 0)), dtype=y_pred.dtype))
    false_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 0), tf.equal(y_pred, 1)), dtype=y_pred.dtype))
    
    # Calculate CSI: TP / (TP + FN + FP)
    denominator = true_positives + false_negatives + false_positives
    csi_value = true_positives / tf.maximum(denominator, 1)  # Avoid division by zero
    return csi_value
