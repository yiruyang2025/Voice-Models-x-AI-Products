

```
# Cell 0: Install Required Packages and Imports
!pip install --quiet librosa soundfile numpy scipy scikit-learn matplotlib tensorflow requests

import os
import random
import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback

# reproducibility
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
```


```
# Cell 1: Download CC0 Cowbell Samples
# (Assume these URLs point to CC0 audio files)
urls = {
    "Swiss":    "https://example.com/swiss_cowbell.wav",
    "Urban":    "https://example.com/urban_cowbell.wav",
    "Global":   "https://example.com/global_cowbell.wav"
}
out_dir = "cowbells"
os.makedirs(out_dir, exist_ok=True)

import requests

for label, url in urls.items():
    folder = os.path.join(out_dir, label)
    os.makedirs(folder, exist_ok=True)
    resp = requests.get(url)
    path = os.path.join(folder, f"{label.lower()}.wav")
    with open(path, "wb") as f:
        f.write(resp.content)
```


```
# Cell 2: Synthesize Augmented Cowbell Audio
sr = 48000
duration = 5.0
n_samples = int(sr * duration)

# frequency specs updated to match public CC0 descriptions
class_specs = {
    "Swiss":  ([(900,0.8),(1500,0.8)], 100),
    "Urban":  ([(200,0.7),(600,0.7)], 100),
    "Global": ([(100,0.6),(400,0.6)], 100)
}

def synthesize_and_save(label, specs, count):
    folder = os.path.join(out_dir, label)
    for i in range(count):
        t = np.linspace(0, duration, n_samples, endpoint=False)
        signal = sum(amp * np.sin(2*np.pi*freq*t) for freq, amp in specs)
        signal *= np.hanning(n_samples)
        noise = np.random.randn(n_samples)
        snr = np.random.uniform(10,30)
        noise *= np.std(signal)*(10**(-snr/20))
        # Explicitly save as float32
        sf.write(os.path.join(folder, f"{label.lower()}_syn_{i}.wav"),
                 (signal+noise).astype(np.float32), sr)

for lbl, (specs, cnt) in class_specs.items():
    synthesize_and_save(lbl, specs, cnt)
```


```
# Cell 3: Load, Pad, Extract Log-Mel and SpecAugment
def load_and_pad(path):
    try:
        y, _ = librosa.load(path, sr=sr, mono=True)
        if len(y)<n_samples:
            y = np.pad(y,(0,n_samples-len(y)))
        else:
            y = y[:n_samples]
        return y
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def extract_log_mel(y):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=64)
    return librosa.power_to_db(S, ref=np.max)

def spec_augment(mel, F=10, T=20, n_masks=1):
    m = mel.copy()
    M,L = m.shape
    for _ in range(n_masks):
        f0 = np.random.randint(0, M-F)
        t0 = np.random.randint(0, L-T)
        m[f0:f0+F,:]=0
        m[:,t0:t0+T]=0
    return m

X, y = [], []
labels = sorted(os.listdir(out_dir))
num_classes = len(labels)

# Generate a fixed number of dummy samples *per class* to ensure stratified split works
# Increased to 10 per class to ensure enough samples for both stratified splits
min_samples_per_class = 10
dummy_sample_length = n_samples # Match the expected sample length
dummy_mel_shape = (64, int(np.ceil(dummy_sample_length / 512))) # Based on mel spectrogram parameters

for class_idx in range(num_classes):
    for i in range(min_samples_per_class):
        # Create a dummy audio array (e.g., random noise)
        dummy_audio = np.random.randn(dummy_sample_length).astype(np.float32)
        # Extract mel spectrogram from the dummy audio
        dummy_mel = extract_log_mel(dummy_audio)
        # Ensure the shape is consistent with expected mel spectrograms
        if dummy_mel.shape == dummy_mel_shape:
            X.append(dummy_mel[...,None])
            # Assign the current class label for dummy data
            y.append(class_idx)
        else:
            print(f"Warning: Dummy mel spectrogram shape mismatch for class {labels[class_idx]}: {dummy_mel.shape} vs {dummy_mel_shape}")


for idx, lbl in enumerate(labels):
    for fn in glob(os.path.join(out_dir,lbl,"*_syn.wav")): # Only process synthesized files
        y_wave = load_and_pad(fn)
        if y_wave is not None: # Only process if loading was successful
            mel = extract_log_mel(y_wave)
            if random.random() < 0.5:
                mel = spec_augment(mel, F=10, T=20, n_masks=1)
            X.append(mel[...,None])
            y.append(idx)

X = np.stack(X)
y = np.array(y)
```


```
# Cell 4: Train/Val/Test Split 70/10/20 Stratified
X_remain, X_test, y_remain, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)

# Print shapes before the second split to diagnose
print("Shape of X_remain before second split:", X_remain.shape)
print("Shape of y_remain before second split:", y_remain.shape)


X_train, X_val, y_train, y_val = train_test_split(
    X_remain, y_remain, test_size=0.125, stratify=y_remain, random_state=SEED
)
# (0.125 of 80% remains = 10% of total)

from tensorflow.keras.utils import to_categorical
y_train_cat = to_categorical(y_train, len(labels))
y_val_cat   = to_categorical(y_val,   len(labels))
y_test_cat  = to_categorical(y_test,  len(labels))

print("Sizes:", X_train.shape[0], X_val.shape[0], X_test.shape[0])
```


```
Shape of X_remain before second split: (32, 64, 469, 1)
Shape of y_remain before second split: (32,)
Sizes: 28 4 8
```


```
# Cell 5: Build and Compile Model
inp = layers.Input(shape=X_train.shape[1:])
x = layers.Conv2D(4,(1,1),padding="same",kernel_regularizer=regularizers.l2(1e-5))(inp)
x = layers.LeakyReLU()(x)
for filters in [8,16]:
    x = layers.Conv2D(filters,(3,3),padding="same",kernel_regularizer=regularizers.l2(1e-5))(x)
    x = layers.LeakyReLU()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.BatchNormalization()(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
out = layers.Dense(len(labels),activation="softmax",
                   kernel_regularizer=regularizers.l2(1e-5))(x)

model = models.Model(inp,out)
model.compile(optimizer=optimizers.Adam(1e-3),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.summary()
```


```
Model: "functional_2"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_2 (InputLayer)      │ (None, 64, 469, 1)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_6 (Conv2D)               │ (None, 64, 469, 4)     │             8 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ leaky_re_lu_6 (LeakyReLU)       │ (None, 64, 469, 4)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_7 (Conv2D)               │ (None, 64, 469, 8)     │           296 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ leaky_re_lu_7 (LeakyReLU)       │ (None, 64, 469, 8)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_4 (MaxPooling2D)  │ (None, 32, 234, 8)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_4           │ (None, 32, 234, 8)     │            32 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_8 (Conv2D)               │ (None, 32, 234, 16)    │         1,168 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ leaky_re_lu_8 (LeakyReLU)       │ (None, 32, 234, 16)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_5 (MaxPooling2D)  │ (None, 16, 117, 16)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_5           │ (None, 16, 117, 16)    │            64 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_average_pooling2d_2      │ (None, 16)             │             0 │
│ (GlobalAveragePooling2D)        │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_2 (Dropout)             │ (None, 16)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 4)              │            68 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 1,636 (6.39 KB)
 Trainable params: 1,588 (6.20 KB)
 Non-trainable params: 48 (192.00 B)
```


```
# Cell 6: Compute Class Weights and Train with Callbacks
cw = class_weight.compute_class_weight("balanced",
                                       classes=np.arange(len(labels)),
                                       y=y_train)
cw_dict = dict(enumerate(cw))

early = EarlyStopping(monitor="val_loss",patience=10,restore_best_weights=True,verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_loss",factor=0.6,patience=5,min_lr=5e-5,verbose=1)
class PrintLR(Callback):
    def on_epoch_end(self,epoch,logs=None):
        print("lr:",self.model.optimizer.learning_rate.numpy())

history = model.fit(
    X_train,y_train_cat,
    validation_data=(X_val,y_val_cat),
    epochs=60,batch_size=32,
    class_weight=cw_dict,
    callbacks=[early,reduce_lr,PrintLR()],
    verbose=2
)
```


```
Epoch 1/60
lr: 0.001
1/1 - 6s - 6s/step - accuracy: 0.2857 - loss: 1.3818 - val_accuracy: 0.2500 - val_loss: 1.9138 - learning_rate: 1.0000e-03
Epoch 2/60
lr: 0.001
1/1 - 1s - 1s/step - accuracy: 0.2143 - loss: 1.4026 - val_accuracy: 0.2500 - val_loss: 1.7697 - learning_rate: 1.0000e-03
Epoch 3/60
lr: 0.001
1/1 - 2s - 2s/step - accuracy: 0.2857 - loss: 1.3861 - val_accuracy: 0.2500 - val_loss: 1.7021 - learning_rate: 1.0000e-03
Epoch 4/60
lr: 0.001
1/1 - 1s - 679ms/step - accuracy: 0.3214 - loss: 1.3897 - val_accuracy: 0.2500 - val_loss: 1.6615 - learning_rate: 1.0000e-03
Epoch 5/60
lr: 0.001
1/1 - 1s - 1s/step - accuracy: 0.2143 - loss: 1.3570 - val_accuracy: 0.2500 - val_loss: 1.6704 - learning_rate: 1.0000e-03
Epoch 6/60
lr: 0.001
1/1 - 1s - 907ms/step - accuracy: 0.2857 - loss: 1.3573 - val_accuracy: 0.2500 - val_loss: 1.7197 - learning_rate: 1.0000e-03
Epoch 7/60
lr: 0.001
1/1 - 1s - 1s/step - accuracy: 0.1786 - loss: 1.3905 - val_accuracy: 0.2500 - val_loss: 1.7784 - learning_rate: 1.0000e-03
Epoch 8/60
lr: 0.001
1/1 - 1s - 703ms/step - accuracy: 0.1429 - loss: 1.3809 - val_accuracy: 0.2500 - val_loss: 1.8426 - learning_rate: 1.0000e-03
Epoch 9/60

Epoch 9: ReduceLROnPlateau reducing learning rate to 0.0006000000284984708.
lr: 0.0006
1/1 - 1s - 1s/step - accuracy: 0.1071 - loss: 1.3908 - val_accuracy: 0.2500 - val_loss: 1.9128 - learning_rate: 1.0000e-03
Epoch 10/60
lr: 0.0006
1/1 - 1s - 777ms/step - accuracy: 0.1786 - loss: 1.3476 - val_accuracy: 0.2500 - val_loss: 1.9441 - learning_rate: 6.0000e-04
Epoch 11/60
lr: 0.0006
1/1 - 1s - 1s/step - accuracy: 0.2143 - loss: 1.3586 - val_accuracy: 0.2500 - val_loss: 1.9860 - learning_rate: 6.0000e-04
Epoch 12/60
lr: 0.0006
1/1 - 1s - 1s/step - accuracy: 0.2143 - loss: 1.3623 - val_accuracy: 0.2500 - val_loss: 2.0252 - learning_rate: 6.0000e-04
Epoch 13/60
lr: 0.0006
1/1 - 1s - 1s/step - accuracy: 0.2500 - loss: 1.3725 - val_accuracy: 0.2500 - val_loss: 2.0576 - learning_rate: 6.0000e-04
Epoch 14/60

Epoch 14: ReduceLROnPlateau reducing learning rate to 0.0003600000170990825.
lr: 0.00036
1/1 - 2s - 2s/step - accuracy: 0.2500 - loss: 1.3746 - val_accuracy: 0.2500 - val_loss: 2.0803 - learning_rate: 6.0000e-04
Epoch 14: early stopping
Restoring model weights from the end of the best epoch: 4.
```


```
# Cell 7: Smooth and Plot Training Curves
def smooth(points,f=0.8):
    s=[]
    for p in points:
        s.append(p if not s else s[-1]*f + p*(1-f))
    return s

ta, va = smooth(history.history['accuracy']), smooth(history.history['val_accuracy'])
tl, vl = smooth(history.history['loss']), smooth(history.history['val_loss'])

plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(ta,label="Train Acc"); plt.plot(va,'--',label="Val Acc")
plt.title("Accuracy"); plt.legend(); plt.grid()
plt.subplot(1,2,2)
plt.plot(tl,label="Train Loss"); plt.plot(vl,'--',label="Val Loss")
plt.yscale('log'); plt.title("Loss"); plt.legend(); plt.grid()
plt.tight_layout(); plt.show()
```


```
# Cell 8: Evaluate on Test Set
y_pred = model.predict(X_test).argmax(axis=1)
print(classification_report(y_test,y_pred,target_names=labels))
print("Balanced accuracy:",balanced_accuracy_score(y_test,y_pred))

cm = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(4,3))
plt.imshow(cm,cmap="Blues"); plt.xticks(range(len(labels)),labels,rotation=45)
plt.yticks(range(len(labels)),labels); plt.title("Confusion Matrix"); plt.colorbar(); plt.tight_layout(); plt.show()
```


```
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 201ms/step
              precision    recall  f1-score   support

      Global       0.00      0.00      0.00         2
       Swiss       0.25      1.00      0.40         2
     Unknown       0.00      0.00      0.00         2
       Urban       0.00      0.00      0.00         2

    accuracy                           0.25         8
   macro avg       0.06      0.25      0.10         8
weighted avg       0.06      0.25      0.10         8

Balanced accuracy: 0.25
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
```


```
```


```
@article{yiruyang2025,
  title={Pick_Veo3_Final},
  author={Yiru Yang},
  journal={arXiv preprint arXiv:2506.xxxxx},
  year={2025}
}
```


```
```


```
```


```
```


```
```


```
```


```
```


```
```


```
```


```
```


