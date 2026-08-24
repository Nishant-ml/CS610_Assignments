# Makefile for Problem 2

CXX = g++
CXXFLAGS = -std=c++17
PTHREAD = -pthread

# Sources and outputs
SRC_ORIGINAL = original.cpp
SRC_PADDED   = prob2_padded.cpp
SRC_FINAL    = prob2_final.cpp

OUT_ORIGINAL = original
OUT_PADDED   = prob2_padded
OUT_FINAL    = prob2_final

all: $(OUT_ORIGINAL) $(OUT_PADDED) $(OUT_FINAL)

$(OUT_ORIGINAL): $(SRC_ORIGINAL)
	$(CXX) $(CXXFLAGS) $(SRC_ORIGINAL) -o $(OUT_ORIGINAL) $(PTHREAD)

$(OUT_PADDED): $(SRC_PADDED)
	$(CXX) $(CXXFLAGS) $(SRC_PADDED) -o $(OUT_PADDED) $(PTHREAD)

$(OUT_FINAL): $(SRC_FINAL)
	$(CXX) -O2 $(CXXFLAGS) $(SRC_FINAL) -o $(OUT_FINAL) $(PTHREAD)

run_original: $(OUT_ORIGINAL)
	./$(OUT_ORIGINAL) $(ARGS)

run_padded: $(OUT_PADDED)
	./$(OUT_PADDED) $(ARGS)

run_final: $(OUT_FINAL)
	./$(OUT_FINAL) $(ARGS)

clean:
	rm -f $(OUT_ORIGINAL) $(OUT_PADDED) $(OUT_FINAL)
