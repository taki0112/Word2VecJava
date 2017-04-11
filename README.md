# Word2Vec In Java
* https://code.google.com/archive/p/word2vec/source/default/source
* 위 코드를 Java로 바꾸었습니다.

## 사용방법
* 소스코드가 들어있는 폴더에 "Input.txt"를 넣습니다.
Input.txt 내용은 다음과 같습니다.
한 줄당, 하나의 문서가 들어갑니다.
이때 문서는 모두 전처리가 완료되어야 합니다.
* 전처리 : 문서를 형태소분석기를 이용하여, 단어들로 구분지어야함

## Input.txt 내용
* 문서1 : 김준호는 머신러닝과 딥러닝에 관심이 많다.
* 문서2 : 김준호는 전문연구요원채용에 관심이 많다.
```bash
김준호 머신러닝 딥러닝 관심
김준호 전문 연구 요원 채용 관심
```

## 주요 변수 설명
* See Line 894 (public static class Builder)
```bash
1. cbow = false
   cbow모델과 skip-gram모델중 어떤 모델로 학습할 것인지 선택하는것입니다.
   false : use skip gram
   true : use cbow model

2. startingAlpha = 0.025F
   learningrate라고 보면됨
   값이 작으면 작을수록, 학습이 정밀해지지만 학습속도가 느려짐

3. window = 5
   학습시, 주변단어를 몇개까지 볼것인가
   기본값은 5이며, 5개의 단어를 본다는 의미

4. negative = 0
   계산속도의 효율성을 증진시키는데 이용한다고 보면됨
   방법론은 Hierarchical Softmax와 Negative Sampling이 있음
   0일경우, Hierarchical Softmax
   그 이외의 숫자일 경우 Negative Sampling (이때 숫자의 기본값은 5~10)

5. minCount = 5
   문서에서 최소 몇개이상 나온 단어들만 보겠다 라는 의미
   모든 단어를 학습시키고 싶다면, minCount = 0

6. layerOneSize = 200
   단어 벡터의 dimension을 의미
   기본값은 200
   dimension이 높을수록, 정밀해지지만 학습속도는 느려짐
```
