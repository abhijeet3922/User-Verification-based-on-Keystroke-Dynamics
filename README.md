# User-Verification-based-on-Keystroke-Dynamics
This project tries to verify a subject's identity based on his keystroke patterns, ie, his typing habits.

Best thing would be to follow my blog-post for implementation. The description about the steps to build the system from scratch can be read from my blog:

https://appliedmachinelearning.wordpress.com/2017/01/23/nlp-blog-post/

It is a Python implementation using following 5 models that measure the similarity between a subject's modelled typing behavior and a new typing sample:
  1) Manhattan Distance
  2) Manhattan Filtered Distance
  3) Manhattan Scaled Distance
  4) GMM
  5) SVM

The results has been reported as the average EER values on the CMU Keystroke Benchmark dataset.
The dataset, in form of a csv and an xls file, has been uploaded in the repo.

Note : Keep all the files in the same folder.
