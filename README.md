# BeatGAN 모델 학습
- 모델 입력 데이터 길이: 320
- 학습할 항목: SO2
```python
python main.py --elename SO2 --isize 320
```
- 재생성 옵션 추가
```python
python main.py --elename SO2 --isize 320 --generated
```

# BeatGAN 모델 테스트 (테스트 할 때도 가중치 파일 아래 그림처럼 상위 디렉토리로 빼야함)

![](2022-03-04-14-45-21.png)

- 위 그림처럼 해당 데이터의 가중치파일 상위 디렉토리에 배치해야함
------


- 모델 입력 데이터 길이: 320
- 테스트 할 항목: SO2
```python
python main.py --elename SO2 --isize 320  --istest
```
## BeatGAN Fake Data 생성
1. 학습된 BeatGAN weight 파일 result/beatgan/NIER/model/에 배치  

![](/images/2022-01-16-18-40-32.png)

![](/images/2022-01-16-18-42-09.png)

2. utils/data_generator.py 스크립트 사용
    ```python
    python utils/data_generator.py --elename SO2 --isize 320  --generated
    ```
    - 위 커맨드를 통해 fakedata가 생성됨. 학습할 땐 이 fakedata를 정상 데이터로 간주하고 학습이 진행됨.
    - 테스트 시에는 정확한 성능 산출을 위해 fakedata는 포함하지 않음

# Pycaret을 활용한 Anomaly detection model 성능 평가  
1. Pycaret 전용 Dictionary 데이터 생성  
   1.1 BeatGAN Fake Data 생성 (위 항목 확인)  
   1.2 Utils/preprocess.py의 NumpySave 함수 save 인자 True로 변경  
   ```python
   numpySave(test_nt, test_na, test_nv, test_nc, test_ant, test_ana, test_anv, test_anc, save=True)
   ```
   1.3 numpySave 함수 호출부 바로 밑에 exit() 함수 호출해서 프로그램 종료
   ```python
   numpySave(test_nt, test_na, test_nv, test_nc, test_ant, test_ana, test_anv, test_anc, save=True)
   exit()
   ```
2. Pycaret 전용 데이터를 통한 평가 진행
- data/NIER_dataset/ 에서 생성된 데이터 확인
- evaluate_pycaret.ipynb 파일로 생성 데이터를 활용하여 **best f1-score** 산출
![](/images/2022-01-16-18-56-33.png)  

- 위 그림에서 보이는 G의 의미는 model이 ....
