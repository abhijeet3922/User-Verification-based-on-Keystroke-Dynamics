#keystroke_ManhattanFiltered.py

from scipy.spatial.distance import euclidean, cityblock
import numpy as np
np.set_printoptions(suppress = True)
import pandas
from EER import evaluateEER

class ManhattanFilteredDetector:
#just the training() function changes, rest all remains same.

    def __init__(self, subjects):
        self.user_scores = []
        self.imposter_scores = []
        self.mean_vector = []
        self.subjects = subjects
        
    def training(self):
        self.mean_vector = self.train.mean().values
        self.std_vector = self.train.std().values
        dropping_indices = []
        for i in range(self.train.shape[0]):
            cur_score = euclidean(self.train.iloc[i].values, 
                                   self.mean_vector)
            if (cur_score > 3*self.std_vector).all() == True:
                dropping_indices.append(i)
        self.train = self.train.drop(self.train.index[dropping_indices])
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
print "average EER for Manhattan Filtered detector:"
print(ManhattanFilteredDetector(subjects).evaluate())
