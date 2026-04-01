import pickle
import numpy as np
import os
import glob

# ====================================================
# 1. 설정: 실험 파라미터 (저장된 실험 코드와 일치해야 함)
# ====================================================
n_fixed = 100        # 고정된 n 크기
m_max = 2000         # 최대 m 크기
num_points = 10      # 실험 포인트 개수

# Sample Size 배열 재생성 (행 라벨용)
# m varies from 0 to 2000
MM1 = np.linspace(0, m_max, num_points, dtype=int)

# ====================================================
# 2. 파일 찾기 및 불러오기
# ====================================================
# 'PowerDict_unlabeled_fixed_*.pkl' 패턴으로 시작하는 가장 최근 파일 자동 탐색
# file_path = './PowerDict_0.1_[0, -2, -1]_20251128_221830.pkl'
# file_path = './PowerDict_0.1_slice(None, None, None)_20251128_221837.pkl'
# file_path = './PowerDict_0.95_[0, -2, -1]_20251128_221449.pkl'
# file_path = './PowerDict_0.95_slice(None, None, None)_20251128_221756.pkl'
# file_path = 'TypeIErrorDict_0.95_20251129_181040.pkl'
# file_path = 'TimeDict_0.1_slice(None, None, None)_20251129_045207.pkl'
# file_path = 'TimeDict_0.1_[0, -2, -1]_20251129_045144.pkl'
# file_path = 'TimeDict_0.95_slice(None, None, None)_20251129_045055.pkl'
file_path = 'TimeDict_0.95_[0, -2, -1]_20251129_044930.pkl'
try:
    with open(file_path, 'rb') as f:
        results_dict = pickle.load(f)
except Exception as e:
    print(f"Error loading pickle file: {e}")
    exit()

methods = list(results_dict.keys())

# ====================================================
# 3. 표 출력
# ====================================================
print("\n" + "="*85)
print(f"Power Results Summary (Fixed n={n_fixed})")
print(f"File: {os.path.basename(file_path)}")
print("="*85)

# 헤더 출력
# 'Sample (n, m)' 칸은 22칸 확보, 각 메서드는 13칸씩 확보
header = f"{'Sample (n, m)':<22}" + "".join([f"{m:>13}" for m in methods])
print(header)
print("-" * len(header))

# 데이터 행 출력
for j in range(len(MM1)):
    # 행 라벨 생성 (n은 고정, m은 변화)
    n_val = n_fixed
    m_val = MM1[j]
    row_label = f"n={n_val}, m={m_val}"
    
    row_str = f"{row_label:<22}"
    
    for method in methods:
        data = results_dict[method]
        
        # 저장된 데이터가 평균(1D)인지 전체 Trial(2D)인지 확인 후 처리
        if isinstance(data, np.ndarray):
            if data.ndim > 1:
                # 2차원이면 (Trials, Points) -> 평균 계산
                val = data.mean(axis=0)[j]
            else:
                # 1차원이면 (Points) -> 그대로 사용
                val = data[j]
        else:
            val = 0.0 # 예외 처리
            
        # row_str += f"{val:>13.3f}"  # for power comparison
        row_str += f"{1000*val:>13.3f}"  # for running time comparison
    
    print(row_str)

print("="*85 + "\n")


# import pickle
# import numpy as np

# # 1. 실제 저장된 피클 파일 이름으로 수정해주세요.
# # (PowerDict 등 검정력 결과를 읽고 싶다면 파일명만 바꾸면 똑같이 작동합니다.)
# # filename = 'TimeDict_0.95_slice(None, None, None)_20251129_045055.pkl'
# filename = 'TimeDict_0.95_[0, -2, -1]_20251129_044930.pkl'

# with open(filename, 'rb') as f:
#     loaded_dict = pickle.load(f)

# # 2. 실험할 때 사용했던 메서드와 샘플 사이즈 세팅을 그대로 재현합니다.
# methods = ['MMD-perm', 'xMMD', 'xssMMD(knn)', 'xssMMD(ker)', 'xssMMD(rf)']
# num_points = 10
# NN1 = np.linspace(20, 200, num_points, dtype=int)    # n 사이즈 배열
# MM1 = np.linspace(100, 1000, num_points, dtype=int)  # m 사이즈 배열

# # 3. 예쁘게 표 형태로 출력하기
# print("\n" + "="*85)
# print(f"Results from: {filename}")
# print("="*85)

# # 헤더(Header) 생성
# header = f"{'Sample (n, m)':<20}" + "".join([f"{m:>13}" for m in methods])
# print(header)
# print("-" * len(header))

# # 각 행(Row) 생성 및 출력
# for j in range(num_points):
#     n_val = NN1[j]
#     m_val = MM1[j]
#     row_label = f"n={n_val}, m={m_val}"
    
#     row_str = f"{row_label:<20}"
    
#     for method in methods:
#         # 파일에서 j번째 결과값 가져오기
#         val = loaded_dict[method][j]
#         row_str += f"{val:>13.5f}"
    
#     print(row_str)

# print("="*85 + "\n")