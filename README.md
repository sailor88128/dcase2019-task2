# Freesound Audio Tagging 2019 11st place

This is the code for freesound audio tagging 2019 on Kaggle, also DCASE 2019 task2. More details about the task can be found as

http://dcase.community/challenge2019/task-audio-tagging

https://www.kaggle.com/c/freesound-audio-tagging-2019/overview

Requirement:
pytorch >= 1.0.0
librosa
numpy
matplotlib
pickle
random
pandas
sklearn

### Final submission:
On curated dataset, 5 fold cv, inception v3, single model LB is 0.691 and 5 model average is 0.713;
On curated dataset and noisy dataset, with specific loss function, 5 fold cv, 8 layer cnn, 5 model average 0.678;
Geometric average of these two models, 0.731.

### Code introduction:
0ExplorerDataAnalysis.ipynb
-- explore the given dataset, to check file names, labels,
-- calculate the summary information,
-- listen to samples with specific label.

0data_preparation.ipynb
-- extract the features from sound file, normalization,
-- save as one file for train and test.

2dcnn-curated-single.ipynb
-- single model train on curated subset,
-- with data augment methods, mixup, SpecAugment,
-- with disigned loss fucntion, weighted_BCE,
-- with differenct learning rate setup, CosineAnnealing, ReduceLR, CyclicLR,
-- with genernal cnn structures in image field.

2dcnn-curated-cv-tta1.ipynb
-- 5 fold cross validation of the above single model train on curated subset.

2dcnn-all-cv.ipynb
-- 5 fold cross validation on all the data, both noisy and curated.
-- 4 different loss functions are tested.

More details can be found in the technical paper:
http://dcase.community/documents/challenge2019/technical_reports/DCASE2019_Liu_38_t2.pdf
