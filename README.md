TTK4550 Specialization Project 2026

This is the code for the Specialization Project "Residual Multipath Correction using Lightweight
Convolutional Neural Networks".

How to run:

1. Install requirements.txt
2. Run python synth_data.py / synth_data_RT.py to generate synthetic training data
3. Run python data_processing.py to process experimental data
4. Run python classical_doa.py to create MUSIC DOA estimates
6. Run python neural_networks/main.py to run the Neural Correction Models for correcting the MUSIC DOA estimates.







