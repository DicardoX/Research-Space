# Individual Paper Notes

[https://github.com/DicardoX/Individual_Paper_Notes](https://github.com/DicardoX/Individual_Paper_Notes)

> This repository is established to **record individual notes for reading papers.**
>
> **Involving Field**: *AI System, Distributed Training, Cluster Schduling, Workload Trace Analysis, Inference, AI Compilation, Memory Storage, etc.*



###### Content

1. **Cluster Scheduling**: 
    - *AntMan: Dynamic Scaling on GPU Clusters for Deep Learning* (OSDI20)
    - *ElasticFlow: An Elastic Serverless Training Platform for Distributed Deep Learning* (ASPLOS23)
    - *Gandiva: Introspective Cluster Scheduling  for Deep Learning* (OSDI18)
    - *(Gavel) Heterogeneity-Aware Cluster Scheduling Policies for Deep Learning Workloads* (OSDI20)
    - *Hare: Exploiting Inter-job and Intra-job Parallelism of Distributed Machine Learning on Heterogeneous GPUs* (HPDC22)
    - *HiveD: Sharing a GPU Cluster for Deep Learning with Guarantees* (OSDI20)
    - *Liquid: Intelligent Resource Estimation and Network-Efficient Scheduling for Deep Learning Jobs on Distributed GPU Clusters* (TPDS22)
    - *Lucid: A Non-intrusive, Scalable and Interpretable Scheduler for Deep Learning Training Jobs* (ASPLOS23)
    - *Lyra: Elastic Scheduling for Deep Learning Clusters* (EuroSys23)
    - *Optimus: An Efficient Dynamic Resource Scheduler for Deep Learning Clusters* (EuroSys18)
    - *Pollux: Co-adaptive Cluster Scheduling  for Goodput-Optimized Deep Learning* (OSDI21)
    - *Sia: Heterogeneity-aware, goodput-optimized ML-cluster scheduling* (SOSP23)
    - *(Synergy) Looking Beyond GPUs for DNN Scheduling  on Multi-Tenant Clusters* (OSDI22)
    - *Tiresias: A GPU Cluster Manager  for Distributed Deep Learning* (NSDI19)
2. **Hybrid Parallelism**:
    - *Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning* (OSDI22)
    - *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism* (arxiv20)
    - *Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates* (SOSP23)
    - *TUTEL: Adaptive Mixture-of-Experts at Scale* (arxiv22)
3. **Data Parallelism**:
    - *(BytePS) A Unified Architecture for Accelerating Distributed DNN Training in Heterogeneous GPU/CPU Clusters* (OSDI20)
    - *Horovod: fast and easy distributed deep learning in TensorFlow* (arxiv20)
    - *Scaling Distributed Machine Learning  with the Parameter Server* (OSDI14)
4. **Tensor Parallelism**:
    - *GSPMD: General and Scalable Parallelization for ML Computation Graphs* (arxiv21)
    - *(Tofu) Supporting Very Large Models using Automatic Dataflow Graph Partitioning* (EuroSys19)
5. **Pipeline**:
    - *DAPPLE: A Pipelined Data Parallel Approach for Training Large Models* (PPoPP21)
    - *GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism* (NIPS19)
    - *PipeDream: Generalized Pipeline Parallelism for DNN Training* (SOSP19)
6. **Graph Optimization**:
    - *(FlexFlow) BEYOND DATA AND MODEL PARALLELISM FOR DEEP NEURAL NETWORKS* (SysML19)
    - *(OptCNN) Exploring Hidden Dimensions in Parallelizing Convolutional Neural Networks* (ICML18)
    - *TASO: Optimizing Deep Learning Computation with Automatic Generation of Graph Substitutions* (SOSP19)
    - *(MetaFlow) OPTIMIZING DNN COMPUTATION WITH RELAXED GRAPH SUBSTITUTIONS* (SysML19)
    - *Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations  and Parallelization* (OSDI22)
    - *Whale: Efficient Giant Model Training over Heterogeneous GPUs* (ATC22)
7. **Memory/Cache Storage**:
    - *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (arxiv20)
    - *Gemini: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints* (SOSP23)
8. **AI Compilation**:
    - *Operator Fusion in XLA: Analysis and Evaluation* (arxiv23)
9. **Training Hyperparameters**:
    - *(DistBelief) Large Scale Distributed Deep Networks* (x)
    - *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour* (arxiv18)
10. **Communication**:
    - *Optimus-CC: Efficient Large NLP Model Training with 3D Parallelism Aware Communication Compression* (ASPLOS23)
    - *Overlap Communication with Dependent Computation via Decomposition in Large Deep Learning Models* (ASPLOS23)
    - *TOPOOPT: Co-optimizing Network Topology and Parallelization Strategy for Distributed Training Jobs* (NSDI23)
11. **Inference**:
    - *Nexus: A GPU Cluster Engine for Accelerating DNN-Based Video Analysis* (SOSP19)
    - *Paella: Low-latency Model Serving with Software-defined GPU Scheduling* (SOSP23)
    - *(vLLM) Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP23)
12. **Cluster Trace Analysis**:
    - *(Helios) Characterization and Prediction of Deep Learning Workloads in Large-Scale GPU Datacenters* (SC21)
    - *(PAI) MLaaS in the Wild: Workload Analysis and Scheduling in Large-Scale Heterogeneous GPU Clusters* (NSDI22)
    - *(Philly) Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads* (ATC19)
13. **Serverless**:
    - *AQUATOPE: QoS-and-Uncertainty-Aware Resource Management for Multi-stage Serverless Workflows* (ASPLOS23)
14. **HPC**:
    - *(ESLURM) Towards Scalable Resource Management for Supercomputers* (SC22)
15. **Performance Modeling**:
    - *Daydream: Accurately Estimating the Efficacy of Optimizations for DNN Training* (ATC20)
    - *DistSim: A performance model of large-scale hybrid distributed DNN training* (CF23)
    - *dPRO: A Generic Performance Diagnosis and Optimization Toolkit for Expediting Distributed DNN Training* (MLSys22)
    - *FasterMoE: Modeling and Optimizing Training of Large-Scale Dynamic Pre-Trained Models* (PPoPP22)
    - *Habitat: A Runtime-Based Computational Performance Predictor for Deep Neural Network Training* (ATC21)
    - *MAD Max Beyond Single-Node: Enabling Large Machine Learning Model Acceleration on Distributed Systems* (arxiv23)
    - *Machine Learning-enabled Performance Model for DNN Applications and AI Accelerator* (HPCC22)
    - *(MPE) Fast Performance Prediction for Efficient Distributed DNN Training* (x)
    - *PALEO: A PERFORMANCE MODEL FOR DEEP NEURAL NETWORKS* (ICLR17)
