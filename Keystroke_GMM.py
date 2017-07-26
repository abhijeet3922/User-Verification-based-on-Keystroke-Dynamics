#keystroke_GMM.py

from sklearn.mixture import GMM
import pandas 
from EER_GMM import evaluateEER
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class GMMDetector:
#the training(), testing() and evaluateEER() function change, rest all is same.

    def __init__(self, subjects):
        self.user_scores = []
        self.imposter_scores = []
        self.mean_vector = []
        self.subjects = subjects
        
    def training(self):
        self.gmm = GMM(n_components = 2, covariance_type = 'diag', 
                        verbose = False )
        self.gmm.fit(self.train)
 
    def testing(self):
        for i in range(self.test_genuine.shape[0]):
            j = self.test_genuine.iloc[i].values
            cur_score = self.gmm.score(j)
            self.user_scores.append(cur_score)
 
        for i in range(self.test_imposter.shape[0]):
            j = self.test_imposter.iloc[i].values
            cur_score = self.gmm.score(j)
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
print "average EER for GMM detector:"
print(GMMDetector(subjects).evaluate())

