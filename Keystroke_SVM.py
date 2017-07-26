#keystroke_ManhattanFiltered.py

from sklearn.svm import OneClassSVM
import numpy as np
np.set_printoptions(suppress = True)
import pandas
from EER import evaluateEER

class SVMDetector:
#just the training() function changes, rest all remains same.

    def __init__(self, subjects):
        self.u_scores = []
        self.i_scores = []
        self.mean_vector = []
        self.subjects = subjects
        
    
    def training(self):
        self.clf = OneClassSVM(kernel='rbf',gamma=26)
        self.clf.fit(self.train)
 
    def testing(self):
        self.u_scores = -self.clf.decision_function(self.test_genuine)
        self.i_scores = -self.clf.decision_function(self.test_imposter)
        self.u_scores = list(self.u_scores)
        self.i_scores = list(self.i_scores)
 
    def evaluate(self):
        eers = []
 
        for subject in subjects:        
            genuine_user_data = data.loc[data.subject == subject, \
                                         "H.period":"H.Return"]
            imposter_data = data.loc[data.subject != subject, :]
            
            self.train = genuine_user_data[:200]
            self.test_genuine = genuine_user_data[200:]
            self.test_imposter = imposter_data.groupby("subject"). \
                                 head(5).loc[:, "H.period":"H.Return"]
 
            self.training()
            self.testing()
            eers.append(evaluateEER(self.u_scores, \
                                     self.i_scores))
        return np.mean(eers)

path = "D:\\Keystroke\\keystroke.csv" 
data = pandas.read_csv(path)
subjects = data["subject"].unique()
print "average EER for SVM detector:"
print(SVMDetector(subjects).evaluate())
