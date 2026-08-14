import sys; sys.path.insert(0, '.')
from train_mobilenets import build_mobilenet_model, FocalLoss
import tensorflow as tf
import numpy as np

# Test focal loss
loss = FocalLoss(gamma=2.0)
y_true = tf.constant([[1,0,0],[0,1,0],[0,0,1]], dtype=tf.float32)
y_pred = tf.constant([[0.9,0.05,0.05],[0.1,0.8,0.1],[0.1,0.1,0.8]], dtype=tf.float32)
l = loss(y_true, y_pred)
print(f'Focal loss test: {l.numpy():.4f}')

# Build model and test save/load
model = build_mobilenet_model('v3small', 224, 7)
model.compile(optimizer='adam', loss=FocalLoss(gamma=2.0), metrics=['accuracy'])
model.save('/tmp/test_focal.keras')
model2 = tf.keras.models.load_model('/tmp/test_focal.keras', custom_objects={'FocalLoss': FocalLoss})
print(f'Model save/load OK - params: {model2.count_params()/1e6:.2f}M')
print('All tests passed!')
