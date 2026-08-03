import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable(package='CustomLosses')
class YOLOLoss(tf.keras.losses.Loss):
    def __init__(self, lambda_box, lambda_obj, lambda_noobj, lambda_class, name = 'yolo_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class

    def localization_loss(self, true_box, pred_box, object_mask):
        object_mask = tf.expand_dims(object_mask, axis = -1)
        box_loss = tf.square(true_box - pred_box)
        box_loss *= object_mask
        return tf.reduce_sum(box_loss, axis = (1,2,3,4))

    def objectness_loss(self, true_obj, pred_obj, object_mask):
        obj_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels = true_obj, 
            logits = pred_obj)
        obj_loss *= object_mask
        return tf.reduce_sum(obj_loss, axis = (1,2,3))

    def no_object_loss(self, true_obj, pred_obj, no_object_mask):
        no_obj_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            labels = true_obj, 
            logits = pred_obj)
        no_obj_loss *= no_object_mask
        return tf.reduce_sum(no_obj_loss, axis = (1,2,3))

    def classification_loss(self, true_class, pred_class, object_mask):
        class_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels = true_class,
            logits = pred_class
        )
        class_loss *= object_mask
        return tf.reduce_sum(class_loss, axis = (1,2,3))

    def compute_losses_components(self, y_true, y_pred):
        true_obj = y_true[..., 0]
        pred_obj = y_pred[..., 0]

        true_box = y_true[..., 1:5]
        pred_box = y_pred[..., 1:5]

        true_class = y_true[..., 5:]
        pred_class = y_pred[..., 5:]

        object_mask = tf.cast(true_obj > 0, tf.float32)
        no_object_mask = 1.0 - object_mask

        box_loss = self.localization_loss(true_box, pred_box, object_mask)
        obj_loss = self.objectness_loss(true_obj, pred_obj, object_mask)
        no_obj_loss = self.no_object_loss(true_obj, pred_obj, no_object_mask)
        class_loss = self.classification_loss(true_class, pred_class, object_mask)

        box_loss = tf.reduce_mean(box_loss)
        obj_loss = tf.reduce_mean(obj_loss)
        no_obj_loss = tf.reduce_mean(no_obj_loss)
        class_loss = tf.reduce_mean(class_loss)

        total_loss = (
            self.lambda_box * box_loss
            + self.lambda_obj * obj_loss
            + self.lambda_noobj * no_obj_loss
            + self.lambda_class * class_loss
        )

        return total_loss, box_loss, obj_loss, no_obj_loss, class_loss

    def call(self, y_true, y_pred):
        total_loss, _, _, _, _ = self.compute_losses(y_true, y_pred)
        return total_loss
    def get_config(self):
        config = super().get_config()

        config.update({
            "lambda_box": self.lambda_box,
            "lambda_obj": self.lambda_obj,
            "lambda_noobj": self.lambda_noobj,
            "lambda_class": self.lambda_class,
        })

        return config