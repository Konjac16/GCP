APP      := gcp
VALID    := validator
CXX      := g++
CXXFLAGS ?= -O3 -std=gnu++17 -Wall -Wextra

BIN      := bin
SRC_APP  := src/gcp.cpp
SRC_VAL  := src/validator.cpp

.PHONY: all run test clean
all: $(BIN)/$(APP) $(BIN)/$(VALID)

$(BIN)/$(APP): $(SRC_APP) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BIN)/$(VALID): $(SRC_VAL) | $(BIN)
	$(CXX) $(CXXFLAGS) -o $@ $<

$(BIN):
	mkdir -p $(BIN)

run: all
	./$(BIN)/$(APP) 5 12345 < data/0.txt > logs/result/0.out
	./$(BIN)/$(VALID) data/0.txt logs/result/0.out

test: all
	./scripts/tests.sh $(BIN)/$(APP) $(BIN)/$(VALID) tests/cases.list

clean:
	rm -rf $(BIN) logs *.log *.tmp
