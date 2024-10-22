# TabDPT: Scaling Tabular Foundation Models

## Example Usage 1
```
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tabdpt import TabDPTClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = TabDPTClassifier(path='checkpoints/tabdpt_76M.ckpt')
model.fit(X_train, y_train)
y_pred = model.predict(X_test, temperature=0.8, context_size=1024)
print(accuracy_score(y_test, y_pred))
```

## Example Usage 2
```
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tabdpt import TabDPTRegressor

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = TabDPTRegressor(path='checkpoints/tabdpt_76M.ckpt')
model.fit(X_train, y_train)
y_pred = model.predict(X_test, context_size=1024)
print(r2_score(y_test, y_pred))
```

## Model Weights Download

[Download TabDPT 76M model weights](https://drive.google.com/file/d/1v-kAFXMaBWmK1Kk6hLaDDlckdYLTCfV1/view?usp=sharing)

## Roadmap
- [ ] Release other model sizes
- [ ] Release training code
