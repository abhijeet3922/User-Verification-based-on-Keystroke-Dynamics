#keystroke_ManhattanFiltered.py

import numpy as np
np.set_printoptions(suppress = True)
import pandas
from EER import evaluateEER

class ManhattanScaledDetector:
#just the training() function changes, rest all remains same.

    def __init__(self, subjects):
        self.user_scores = []
        self.imposter_scores = []
        self.mean_vector = []
        self.subjects = subjects
        
    def training(self):
        self.mean_vector = self.train.mean().values
        #also calculating mean absolute deviation deviation of each feature
        self.mad_vector  = self.train.mad().values

    def testing(self):
        for i in range(self.test_genuine.shape[0]):
            cur_score = 0
            for j in range(len(self.mean_vector)):
                cur_score = cur_score + \
                            abs(self.test_genuine.iloc[i].values[j] - \
                            self.mean_vector[j]) / self.mad_vector[j]
            self.user_scores.append(cur_score)
 
        for i in range(self.test_imposter.shape[0]):
            cur_score = 0
            for j in range(len(self.mean_vector)):
                cur_score = cur_score + \
                            abs(self.test_imposter.iloc[i].values[j] - \
                            self.mean_vector[j]) / self.mad_vector[j]
            self.imposter_scores.append(cur_score)
 
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
            eers.append(evaluateEER(self.user_scores, \
                                     self.imposter_scores))
        return np.mean(eers)

path = "D:\\Keystroke\\keystroke.csv" 
data = pandas.read_csv(path)
subjects = data["subject"].unique()
print "average EER for Manhattan Scaled detector:"
print(ManhattanScaledDetector(subjects).evaluate())

