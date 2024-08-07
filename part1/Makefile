BUILD=build
CC=g++
FLAGS=-mavx512f

naive: build
	$(CC) $(FLAGS) transpose.c -o $(BUILD)/$@

blocking: build
	$(CC) $(FLAGS) transpose.c -D OPTIMIZE_BLOCKING -o $(BUILD)/$@

prefetch: build
	$(CC) $(FLAGS) transpose.c -D OPTIMIZE_PREFETCH -o $(BUILD)/$@


blocking-prefetch: build
	$(CC) $(FLAGS) transpose.c -D OPTIMIZE_BLOCKING_PREFETCH -o $(BUILD)/$@


all: build
	$(CC) $(FLAGS) transpose.c -D OPTIMIZE_BLOCKING -D OPTIMIZE_PREFETCH -D OPTIMIZE_BLOCKING_PREFETCH -o $(BUILD)/$@

clean:
	@rm -rf $(BUILD)
	@rm -f out.txt

build:
	@mkdir -p $(BUILD)