# Makefile for Problem 1

CXX = g++
CXXFLAGS = -std=c++17
LDFLAGS = -lpapi

SRC = 251110053-prob1.cpp
OUT = prob1

all: $(OUT)

$(OUT): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(OUT) $(LDFLAGS)

run: $(OUT)
	./$(OUT)

clean:
	rm -f $(OUT)
