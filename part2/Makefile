BUILD=build
CC=g++
FLAGS=-mavx512f

naive: build
	$(CC) $(FLAGS) convolution.c -o $(BUILD)/$@

blocking: build
	$(CC) $(FLAGS) convolution.c -D OPTIMIZE_BLOCKING -o $(BUILD)/$@

prefetch: build
	$(CC) $(FLAGS) convolution.c -D OPTIMIZE_PREFETCH -o $(BUILD)/$@

simd: build
	$(CC) $(FLAGS) convolution.c -D OPTIMIZE_SIMD -o $(BUILD)/$@

blocking-prefetch: build
	$(CC) $(FLAGS) convolution.c -D OPTIMIZE_BLOCKING_PREFETCH -o $(BUILD)/$@

blocking-simd: build
	$(CC) $(FLAGS) convolution.c -D OPTIMIZE_BLOCKING_SIMD -o $(BUILD)/$@

simd-prefetch: build
	$(CC) $(FLAGS) convolution.c -D OPTIMIZE_SIMD_PREFETCH -o $(BUILD)/$@

blocking-simd-prefetch: build
	$(CC) $(FLAGS) convolution.c -D OPTIMIZE_BLOCKING_SIMD_PREFETCH -o $(BUILD)/$@

all: build
	$(CC) $(FLAGS) convolution.c -D OPTIMIZE_BLOCKING -D OPTIMIZE_SIMD -D OPTIMIZE_PREFETCH -D OPTIMIZE_BLOCKING_PREFETCH -D OPTIMIZE_BLOCKING_SIMD -D OPTIMIZE_SIMD_PREFETCH -D OPTIMIZE_BLOCKING_SIMD_PREFETCH -o $(BUILD)/$@

clean:
	@rm -rf $(BUILD)
	@rm -f out.txt

build:
	@mkdir -p $(BUILD)