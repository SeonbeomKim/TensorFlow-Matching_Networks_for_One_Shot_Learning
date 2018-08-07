# TensorFlow-Matching_Networks_for_One_Shot_Learning
Matching Networks for One Shot Learning

## paper
https://arxiv.org/abs/1606.04080

## dataset
MNIST (논문에서는 omniglot, ImageNet 으로 실험되어 있으므로 MNIST 실험 진행)

## Matching Network for One Shot Learning Model
논문에서 설명하는 모델  
사용 코드: one_shot_learning_class.py, get_S_B_MNIST.py, train_and_test.py

## CNN Model
Matching Network와 MNIST 분류 정확도 비교를 위한 CNN Model  
사용 코드: cnn.py, get_S_B_MNIST.py

## get_S_B_MNIST.py
class별로 원하는 개수 만큼의 MNIST 데이터를 추출하기 위한 코드

## one_shot_learning_class.py
Matching Network for One Shot Learning Model을 구현한 코드

## train_and_test.py
Matching Network for One Shot Learning Model을 학습하고 테스트 하는 코드.

## Test Result (CNN vs Matching Network for One Shot Learning)
Matching Network는 10-Way 5-Shot. (way: # classes, shot: # memory data per class)  
CNN의 학습 데이터: Matching Network의 학습 데이터 + shot (CNN은 shot이 필요 없으므로)
![testImage](./result/result.png)
