# gear_decode 구현 완료

## 완료된 작업

### 1. C++ 구현 ([gear_generate.cpp](gear_decode/gear_generate.cpp))
- `simple.cpp`와 유사한 baseline llama.cpp 함수 구현
- 완전한 decode loop을 C++ 내부에서 실행
- 토큰별 시간 측정 (time-per-output-token) 기능 추가
- 커스터마이징 가능한 파라미터:
  - `n_threads`: 스레드 수 설정
  - `enable_flash_attn`: Flash Attention 활성화
  - `ngl`: GPU layer 오프로딩
  - `use_instruct`: Instruct 모드 템플릿 적용

### 2. Python 래퍼 ([gear_generate.py](gear_decode/gear_generate.py))
- `token_generator.py`를 참고한 ctypes 기반 래퍼
- 주요 차이점:
  - 모델 instantiate와 decode loop 통합 (분리하지 않음)
  - C++ 내부에서 looping이 완전히 끝난 후 결과 반환
  - 토큰 생성시마다 기록한 time-per-output-token list 제공
  - 최종 output 결과와 통계 정보 제공

### 3. CMakeLists.txt 수정
- `common` 라이브러리 링크 추가
- 올바른 include 디렉토리 설정
- 빌드 출력 디렉토리 설정
- `flash_attn` → `flash_attn_type`으로 수정

## 사용 방법

### Python API 사용

```python
from gear_decode.gear_generate import GearGenerator

# Generator 초기화
generator = GearGenerator()

# 텍스트 생성
result = generator.generate(
    model_path="/path/to/model.gguf",
    prompt="What is Python?",
    n_predict=50,
    use_instruct=True,
    n_threads=4,
    enable_flash_attn=False,
    n_gpu_layers=99
)

# 결과 확인
if result.is_success:
    print(f"출력: {result.output_text}")
    print(f"생성된 토큰 수: {result.n_tokens_generated}")
    print(f"전체 시간: {result.total_time_ms:.2f} ms")
    print(f"토큰/초: {result.tokens_per_second:.2f}")
    
    # 토큰별 시간 확인
    for i, time_ms in enumerate(result.time_per_token):
        tps = 1000.0 / time_ms if time_ms > 0 else 0
        print(f"Token {i+1}: {time_ms:.2f} ms ({tps:.2f} tok/sec)")
```

### 커맨드 라인 사용

```bash
python gear_decode/gear_generate.py \
    -m /path/to/model.gguf \
    -p "What is the capital of France?" \
    -n 100 \
    -t 4 \
    --use-instruct \
    -ngl 99
```

## 빌드

```bash
cd llama.cpp/build
cmake .. -DLLAMA_BUILD_EXAMPLES=ON
make gear_decode
```

빌드된 라이브러리: `build/lib/libgear_decode.so`

## 테스트

```bash
# 라이브러리 로딩 테스트
python3 gear_decode/test_gear_decode.py

# 모델과 함께 전체 테스트
python3 gear_decode/test_gear_decode.py /path/to/model.gguf
```

## GenerationResult 속성

- `output_text` (str): 생성된 전체 텍스트
- `n_tokens_generated` (int): 생성된 토큰 수
- `total_time_ms` (float): 전체 생성 시간 (밀리초)
- `time_per_token` (List[float]): 각 토큰별 생성 시간 리스트 (밀리초)
- `average_time_per_token` (float): 평균 토큰별 시간
- `tokens_per_second` (float): 초당 토큰 수
- `error_code` (int): 0=성공, 그 외=에러
- `is_success` (bool): 성공 여부

## custom_gen과의 차이점

| 기능 | gear_decode | custom_gen |
|------|-------------|------------|
| Decode loop | C++ 내부에서 완전 실행 | Python에서 토큰별로 호출 |
| 반환 방식 | 모든 결과 한번에 반환 | 스트리밍 방식 |
| Timing 데이터 | 토큰별 시간 리스트 | 토큰별 결과에 포함 |
| 사용 사례 | 벤치마킹, 배치 분석 | 인터랙티브 스트리밍 |

## 파일 구조

```
gear_decode/
├── CMakeLists.txt          # 빌드 설정
├── gear_generate.cpp       # C++ 구현
├── gear_generate.py        # Python 래퍼
├── test_gear_decode.py     # 테스트 스크립트
└── README.md               # 문서
```
