import sys
sys.path.insert(0, '.')
import cv2
import torch
import numpy as np
from src.classification.model import PyTorchClassifier
from src.gtsrb_classes import GTSRB_CLASSES

img = cv2.imread(r'C:/Users/Sombuddha Ash/.gemini/antigravity/brain/fe6cd234-911f-40b2-91cb-15b94ed3dfc2/.user_uploaded/media_1791035481617.png')
print('Image shape:', img.shape)
h, w = img.shape[:2]

# Let's crop the actual sign inside the orange box
# In the screenshot, the orange box is roughly:
# y from 0.35 to 0.62, x from 0.18 to 0.58
crop = img[int(h*0.355):int(h*0.62), int(w*0.18):int(w*0.57)]
cv2.imwrite('scratch/user_sign_crop.png', crop)

clf = PyTorchClassifier()
res = clf.classify(crop)
print(f'Pipeline Classifier output: Class {res.class_id} ({res.class_name}), Confidence: {res.confidence:.4f}')

# Let's also look at raw model probabilities
inp = cv2.resize(crop, (32, 32))
inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
inp = inp.astype(np.float32) / 255.0
inp = np.transpose(inp, (2, 0, 1))
inp = np.expand_dims(inp, axis=0)
tensor = torch.from_numpy(inp)

with torch.no_grad():
    out = clf.model(tensor)
    probs = torch.softmax(out, dim=1).numpy()[0]
    top10_idx = np.argsort(probs)[::-1][:10]
    print("\nTop 10 Predictions:")
    for rank, idx in enumerate(top10_idx, 1):
        print(f"{rank:2d}. Class {idx:2d} ({GTSRB_CLASSES.get(idx, 'Unknown')}): {probs[idx]*100:.2f}% (logit={out[0, idx]:.3f})")
