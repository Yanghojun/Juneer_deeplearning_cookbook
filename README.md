# BeatGAN 모델 테스트
- 모델 입력 데이터 길이: 320
- 테스트 할 항목: SO2
```python
python main.py --elename SO2 --isize 320  --istest
```

# Pycaret 라이브러리 평가  
1. Pycaret 전용 Dictionary 데이터 생성  
   1.1 Utils/preprocess.py의 NumpySave 함수 save 인자 True로 변경  
   ```python
   numpySave(test_nt, test_na, test_nv, test_nc, test_ant, test_ana, test_anv, test_anc, save=True)
   ```
   1.2 numpySave 함수 호출부 바로 밑에 exit() 함수 호출해서 프로그램 종료
   ```python
   numpySave(test_nt, test_na, test_nv, test_nc, test_ant, test_ana, test_anv, test_anc, save=True)
   exit()
   ```
2. 모델이 생성한