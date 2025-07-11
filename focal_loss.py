# Define the FocalLoss class in the global scope
class FocalLoss(tf.keras.losses.Loss):
    """
    Focal Loss for addressing class imbalance in medical datasets

    Formula:
        Focal Loss = -α(1 - p_t)^γ * log(p_t)

    Where:
    - p_t is the predicted probability for the true class.
    - α is the balancing factor to give more weight to underrepresented classes.
    - γ (gamma) reduces the loss contribution from easy examples and focuses on hard examples.
    """

    # Constructor: initialize alpha, gamma, and any additional kwargs
    def __init__(self, alpha=1.0, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma

    # This function defines the actual focal loss calculation
    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = self.alpha * tf.pow(1 - p_t, self.gamma)
        focal_loss = focal_weight * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1))

    # Save configuration to support model saving/loading
    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma
        })
        return config