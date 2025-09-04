## What is ML2HLS?
ML2HLS is a tool for determining the optimal parameters for FPGA deployment of neural networks. ML2HLS is based on the HLS4ML framework.

### What is ML2HLS, with an example?
A user might be interested in examining the model accuracy of an FPGA inferred CNN under the precision modes fixed<8,2>, fixed<16,2>, while at the same time exploring different input shapes (16, 16) and (32, 32) for the CNN. 

In the example quantizing is expected to reduce BRAM usage on an FPGA, with an accuracy penalty. Similarly, reducing input shapes reduces not only BRAM usage, but also DSP, FF and LUT usage, at the price of an additional accuracy loss.

ML2HLS builds an HLS4ML project for each one of the four pairs:
precision fixed<8,2> - input shape (16, 16)
precision fixed<8,2> - input shape (32, 32)
precision fixed<16,2> - input shape (16, 16)
precision fixed<16,2> - input shape (32, 32)

This allows us to precisely examine all the advantages and disadvantages of using each pair, and based on our specifications decide on the optimal.

Each experiment produces an HLS4ML project, which is synthesized to FPGA RTL via Vitis HLS, allowing performance comparison, resource usage, and precision trade-offs across configurations.

## Configurable parameters:
### 1. Dataset <- TODO
### 2. Model Architecture <- TODO
### 3. Model Pruning <- TODO
### 4. Model Training <- TODO
### 5. Model Quantization <- TODO
### 6. HLS4ML Config & Converter

