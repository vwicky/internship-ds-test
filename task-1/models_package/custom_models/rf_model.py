from .classifier_interface import MnistClassifierInterface
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np

class RandomForestMnist(MnistClassifierInterface):  
  def __init__(self, n_estimators = 100, criterion='entropy', max_depth=20):
    self.model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth)
  
  def train(self, X_train, y_train) -> None:
    self.model.fit(X_train, y_train)
    print("> training ended")
  
  def predict(self, X_test) -> np.ndarray:
    y_pred = self.model.predict(X_test)
    return y_pred
  
  def evaluate(self, X_test, y_test) -> None:
    y_pred = self.model.predict(X_test)
    print(metrics.classification_report(y_test, y_pred))