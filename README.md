# Freesound Audio Tagging 2019 11st place

This is the code for freesound audio tagging 2019 on Kaggle, also DCASE 2019 task2. More details about the task can be found at:  
  http://dcase.community/challenge2019/task-audio-tagging,  
  https://www.kaggle.com/c/freesound-audio-tagging-2019/overview .

Requirement:  
pytorch >= 1.0.0 </br>
  librosa </br>
  numpy </br>
  matplotlib </br>
  pickle </br>
  random </br>
  pandas </br>
  sklearn </br>

### Final submission:
On curated dataset, 5 fold cv, inception v3, single model LB is 0.691 and 5 model average is 0.713; </br>
On curated dataset and noisy dataset, with specific loss function, 5 fold cv, 8 layer cnn, 5 model average 0.678; </br>
Geometric average of these two models, 0.731. </br>

### Code introduction:
0ExplorerDataAnalysis.ipynb </br>
-- explore the given dataset, to check file names, labels, </br>
-- calculate the summary information, </br>
-- listen to samples with specific label.

0data_preparation.ipynb </br>
-- extract the features from sound file, normalization, </br>
-- save as one file for train and test.

2dcnn-curated-single.ipynb </br>
-- single model train on curated subset, </br>
-- with data augment methods, mixup, SpecAugment, </br>
-- with disigned loss fucntion, weighted_BCE, </br>
-- with differenct learning rate setup, CosineAnnealing, ReduceLR, CyclicLR, </br>
-- with genernal cnn structures in image field.

2dcnn-curated-cv-tta1.ipynb </br>
-- 5 fold cross validation of the above single model train on curated subset.

2dcnn-all-cv.ipynb </br>
-- 5 fold cross validation on all the data, both noisy and curated. </br>
-- 4 different loss functions are tested.

More details can be found in the technical paper: </br>
http://dcase.community/documents/challenge2019/technical_reports/DCASE2019_Liu_38_t2.pdf
