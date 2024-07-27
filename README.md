# SHRINK: Data Compression by Semantic Extraction and Residuals

Welcome to the code for compressing data series based on Semantic Extraction with Residual Encoding.

In this work, we propose a framework to combine data-leve (Semantics) coding with bit-level (Residuals) encoding to build a novel error-bounded data compression method.

We used Turbo-Range-Coder(TRC) as our entropy coding compression as explained in the paper. It can be installed through the link: https://github.com/powturbo/Turbo-Range-Coder

## Requirements
1. Create a new Python virtual environment
2. Install the dependencies via pip install -r requirements.txt

## Model Architecture
The high-level architecture of the SHRIN is shown in the following figure. It first extracts data semantics/sketches in the form of line segments under a base error threshold that adapts to data variability and then merges these semantics into a holistic knowledge base that encodes the underlying data and filters redundancies. Still, these coarse-grained semantics fail at applications that require high accuracy. To serve this goal, we augment its representation with residuals, calculated by a simple subtraction operation, which drastically reduces bit-level redundancy by virtue of their small variance, contributing to a high compression ratio. 

![image](https://github.com/sunguoy/SHRINK/assets/14194254/6efae3df-92c4-424e-a233-cd80442bed46)



## Running
1. Running TestSHRINK under the new framework to test the performance of SHRINK
2. It's free to set different hyperparameters, such $\epsilon_b$ and $\lambda$

## Results
![image](https://github.com/sunguoy/SHRINK/assets/14194254/2f59bc9c-a7d4-4752-96bb-4075885e6103)





