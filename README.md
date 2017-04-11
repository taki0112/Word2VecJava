# Word2VecInJava
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
