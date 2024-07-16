BUILD_DIR = build

.PHONY: all deep_neural_network common clean

all: deep_neural_network

#logistic_regression:
#	$(MAKE) -C logistic_regression

deep_neural_network:
	$(MAKE) -C deep_neural_network

common:
	$(MAKE) -C common

clean:
#	$(MAKE) -C logistic_regression clean
	$(MAKE) -C deep_neural_network clean
	$(MAKE) -C common clean
	rm -rf obj
	rm -rf $(BUILD_DIR)

rgas0x04F1rchess1@

