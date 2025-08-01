# pllm/CMakeLists.txt

# 실행 파일 이름과 소스
add_executable(test_dvfs
    test_dvfs.c
    ${CMAKE_SOURCE_DIR}/ggml/src/ggml-cpu/ggml-dvfs.c
)

# ggml 헤더 검색 경로
target_include_directories(test_dvfs PRIVATE
    ${CMAKE_SOURCE_DIR}/ggml/include
)

# ggml-cpu 백엔드 라이브러리 (또는 ggml 전체 라이브러리) 링크
target_link_libraries(test_dvfs PRIVATE
    ggml-cpu
)
