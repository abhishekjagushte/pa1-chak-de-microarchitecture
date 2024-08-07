BUILD=build
CC=g++
FLAGS=-mavx512f

naive: build
	$(CC) $(FLAGS) convolution.c -o $(BUILD)/$@

simd: build
	$(CC) $(FLAGS) convolution.c -D OPTIMIZE_SIMD -o $(BUILD)/$@

all: build
	$(CC) $(FLAGS) convolution.c -D OPTIMIZE_SIMD -o $(BUILD)/$@

clean:
	@rm -rf $(BUILD)
	@rm -f out.txt

build:
	@mkdir -p $(BUILD)