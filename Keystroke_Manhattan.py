#Keystroke_Manhattan.py

from scipy.spatial.distance import cityblock
import numpy as np
np.set_printoptions(suppress = True)
import pandas
from EER import evaluateEER

class ManhattanDetector:
 
    def __init__(self, subjects):
        self.user_scores = []
        self.imposter_scores = []
        self.mean_vector = []
        self.subjects = subjects
 
    def training(self):
        self.mean_vector = self.train.mean().values         
 
    def testing(self):
        for i in range(self.test_genuine.shape[0]):
            cur_score = cityblock(self.test_genuine.iloc[i].values, \
                                   self.mean_vector)
            self.user_scores.append(cur_score)
 
        for i in range(self.test_imposter.shape[0]):
            cur_score = cityblock(self.test_imposter.iloc[i].values, \
                                   self.mean_vector)
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
print "average EER for Manhattan detector:"
print(ManhattanDetector(subjects).evaluate())