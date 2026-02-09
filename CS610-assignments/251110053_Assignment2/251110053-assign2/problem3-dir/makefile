# Makefile for Problem 3

CXX = g++
CXXFLAGS = -std=c++17
PTHREAD = -pthread

SRC = 251110053-prob3.cpp
OUT = prob3

all: $(OUT)

$(OUT): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(OUT) $(PTHREAD)

run: $(OUT)
	./$(OUT)

clean:
	rm -f $(OUT)
