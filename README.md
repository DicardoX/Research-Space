# Research Space

[https://github.com/DicardoX/Research-Space](https://github.com/DicardoX/Research-Space)

> This repository is established to store personal notes and annotated papers during daily research.
>
> **Involving Field**: *ML System, LLM, Distributed Training, Cluster Schduling, Inference, Workload Trace Analysis, AI Compilation, Memory/Cache Storage, etc.*



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
    - *SiloD: A Co-design of Caching and Scheduling for Deep Learning Clusters* (EuroSys23)
    - *(Synergy) Looking Beyond GPUs for DNN Scheduling  on Multi-Tenant Clusters* (OSDI22)
    - *Tiresias: A GPU Cluster Manager  for Distributed Deep Learning* (NSDI19)

2. **Hybrid Parallelism**:
    - *Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning* (OSDI22)
    - *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism* (arxiv20)
    - *Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates* (SOSP23)
    - *Piper: Multidimensional Planner for DNN Parallelization* (NIPS21)
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
    - *DynaPipe: Optimizing Multi-task Training through Dynamic Pipelines* (EuroSys24)
    - *GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism* (NIPS19)
    - *PipeDream: Generalized Pipeline Parallelism for DNN Training* (SOSP19)
    - *Tessel: Boosting Distributed Execution of Large DNN Models via Flexible Schedule Search* (arxiv23)
    - *(ZeroBubble) NEAR ZERO BUBBLE PIPELINE PARALLELISM* (ICLR24)
    
6. **Graph Optimization**:
    - *(FlexFlow) BEYOND DATA AND MODEL PARALLELISM FOR DEEP NEURAL NETWORKS* (SysML19)
    - *(OptCNN) Exploring Hidden Dimensions in Parallelizing Convolutional Neural Networks* (ICML18)
    - *TASO: Optimizing Deep Learning Computation with Automatic Generation of Graph Substitutions* (SOSP19)
    - *(MetaFlow) OPTIMIZING DNN COMPUTATION WITH RELAXED GRAPH SUBSTITUTIONS* (SysML19)
    - *Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations  and Parallelization* (OSDI22)
    - *Whale: Efficient Giant Model Training over Heterogeneous GPUs* (ATC22)

7. **Memory/Cache Storage**:
    - *Bagpipe: Accelerating Deep Recommendation Model Training* (SOSP23)
    - *Gemini: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints* (SOSP23)
    - *MAGIS: Memory Optimization via Coordinated Graph Transformation and Scheduling for DNN* (ASPLOS24)
    - *UGache: A Unified GPU Cache for Embedding-based Deep Learning* (SOSP23)
    - *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (arxiv20)

8. **AI Compilation**:

    - *FlexTensor: An Automatic Schedule Exploration and Optimization Framework for Tensor Computation on Heterogeneous System* (ASPLOS20)

    - *Welder: Scheduling Deep Learning Memory Access via Tile-graph* (OSDI23)
    - *Operator Fusion in XLA: Analysis and Evaluation* (arxiv23)

9. **Training Hyperparameters**:
    - *(DistBelief) Large Scale Distributed Deep Networks* (x)
    - *Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour* (arxiv18)

10. **Communication**:
    - *Optimus-CC: Efficient Large NLP Model Training with 3D Parallelism Aware Communication Compression* (ASPLOS23)
    - *Overlap Communication with Dependent Computation via Decomposition in Large Deep Learning Models* (ASPLOS23)
    - *TOPOOPT: Co-optimizing Network Topology and Parallelization Strategy for Distributed Training Jobs* (NSDI23)

11. **Training**:
      - *Aceso: Efficient Parallel DNN Training through Iterative Bottleneck Alleviation* (EuroSys24)
      - *AdapterFusion: Non-Destructive Task Composition for Transfer Learning* (arxiv21)
      - *AMSP: Super-Scaling LLM Training via Advanced Model States Partitioning* (arxiv23)
      - *Capuchin: Tensor-based GPU Memory Management for Deep Learning* (ASPLOS20)
      - *Centauri: Enabling Efficient Scheduling for Communication-Computation Overlap in Large Model Training via Communication Partitioning* (ASPLOS24)
      - *(Diff-Pruning) Parameter-Efficient Transfer Learning with Diff Pruning* (ACL21)
      - *DistMM: Accelerating Distributed Multimodal  Model Training* (NSDI24)
      - *DLoRA: Distributed Parameter-Efficient Fine-Tuning Solution for Large Language Model* (arxiv24)
      - *Egeria: Efficient DNN Training with Knowledge-Guided Layer Freezing* (EuroSys23)
      - *FlexLLM: A System for Co-Serving Large Language Model Inference and Parameter-Efficient Finetuning* (arxiv24)
      - *(Fuyou) Adding NVMe SSDs to Enable and Accelerate 100B Model Fine-tuning on a Single GPU* (arxiv24)
      - *HAP: SPMD DNN Training on Heterogeneous GPU Clusters with Automated Program Synthesis* (EuroSys24)
      - *Harmony: Overcoming the Hurdles of GPU Memory Capacity to Train Massive DNN Models on Commodity Servers* (VLDB22)
      - *HiFT: A Hierarchical Full Parameter Fine-Tuning Strategy* (arxiv24)
      - *Hydro: Surrogate-Based Hyperparameter Tuning Service in Datacenters* (OSDI23)
      - *InternEvo: Efficient Long-Sequence Large Language Model Training via Hybrid Parallelism and Redundant Sharding* (arxiv24)
      - *MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs* (NSDI24)
      - *(Megatron-LM-Large-Scale) Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM* (arxiv21)
      - *Mobius: Fine Tuning Large-Scale Models on Commodity GPU Servers* (ASPLOS23)
      - *(ProPETL) One Network, Many Masks: Towards More Parameter-Efficient Transfer Learning* (arxiv23)
      - *Retiarii: A Deep Learning Exploratory-Training Framework* (OSDI20)
      - *(SeqParallel) Reducing Activation Recomputation in Large Transformer Models* (arxiv22)
      - *ScheMoE: An Extensible Mixture-of-Experts Distributed Training System with Tasks Scheduling* (EuroSys24)
      - *(Unified-PEFT) TOWARDS A UNIFIED VIEW OF PARAMETER-EFFICIENT TRANSFER LEARNING* (ICLR22)
      - *Ymir: A Scheduler for Foundation Model Fine-tuning Workloads* (ICS24)
      - *ZeRO-Offload: Democratizing Billion-Scale Model Training* (ATC21)
    
12. **Inference**:
      - *AlpaServe: Statistical Multiplexing with Model Parallelism for Deep Learning Serving* (OSDI23)
      - *(Brainstorm) Optimizing Dynamic Neural Networks with Brainstorm* (OSDI23)
      - *(DeepPlan) Fast and Efficient Model Serving Using Multi-GPUs with Direct-Host-Access* (EuroSys23)
      - *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving* (arxiv24)
      - *FaaSwap: SLO-Aware, GPU-Efficient Serverless Inference via Model Swapping* (arxiv24)
      - *(FastServe) Fast Distributed Inference Serving for Large Language Models* (arxiv23)
      - *FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU* (ICML23)
      - *Helix: Distributed Serving of Large Language Models via Max-Flow on Heterogeneous GPUs* (arxiv24)
      - *Infinite-LLM: Efficient LLM Service for Long Context with DistAttention and Distributed KVCache* (arxiv24)
      - *LoongServe: Efficiently Serving Long-context Large Language Models with Elastic Sequence Parallelism* (arxiv24)
      - *Nexus: A GPU Cluster Engine for Accelerating DNN-Based Video Analysis* (SOSP19)
      - *ORCA: A Distributed Serving System for Transformer-Based Generative Models* (OSDI22)
      - *Paella: Low-latency Model Serving with Software-defined GPU Scheduling* (SOSP23)
      - *Parrot: Efficient Serving of LLM-based Applications with Semantic Variable* (OSDI24)
      - *PETALS: Collaborative Inference and Fine-tuning of Large Models* (arxiv23)
      - *PetS: A Unified Framework for Parameter-Efficient Transformers Serving* (ATC22)
      - *PUNICA: MULTI-TENANT LORA SERVING* (arxiv23)
      - *SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills* (arxiv23)
      - *(Sarathi-Serve) Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve* (arxiv24)
      - *(SGLang) Efficiently Programming Large Language Models using SGLang* (arxiv24)
      - *Shepherd: Serving DNNs in the Wild* (NSDI23)
      - *S-LORA: SERVING THOUSANDS OF CONCURRENT LORA ADAPTERS* (arxiv23)
      - *Tabi: An Efficient Multi-Level Inference System for Large Language Models* (EuroSys23)
      - *(TetriInfer) Inference without Interference: Disaggregate LLM Inference for Mixed Downstream Workloads* (arxiv24)
      - *TurboTransformers: An Efficient GPU Serving System For Transformer Models* (PPoPP21)
      - *(vLLM) Efficient Memory Management for Large Language Model Serving with PagedAttention* (SOSP23)
      - *(VTC) Fairness in Serving Large Language Models* (arxiv23)
    
13. **Cluster Trace Analysis**:
      - *(Helios) Characterization and Prediction of Deep Learning Workloads in Large-Scale GPU Datacenters* (SC21)
      - *(PAI) MLaaS in the Wild: Workload Analysis and Scheduling in Large-Scale Heterogeneous GPU Clusters* (NSDI22)
      - (Philly) Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads* (ATC19)

14. **Serverless**:
      - *AQUATOPE: QoS-and-Uncertainty-Aware Resource Management for Multi-stage Serverless Workflows* (ASPLOS23)

15. **HPC**:
      - *(ESLURM) Towards Scalable Resource Management for Supercomputers* (SC22)

16. **Performance Modeling**:
      - *Daydream: Accurately Estimating the Efficacy of Optimizations for DNN Training* (ATC20)
      - *DistSim: A performance model of large-scale hybrid distributed DNN training* (CF23)
      - *dPRO: A Generic Performance Diagnosis and Optimization Toolkit for Expediting Distributed DNN Training* (MLSys22)
      - *FasterMoE: Modeling and Optimizing Training of Large-Scale Dynamic Pre-Trained Models* (PPoPP22)
      - *Habitat: A Runtime-Based Computational Performance Predictor for Deep Neural Network Training* (ATC21)
      - *MAD Max Beyond Single-Node: Enabling Large Machine Learning Model Acceleration on Distributed Systems* (arxiv23)
      - *Machine Learning-enabled Performance Model for DNN Applications and AI Accelerator* (HPCC22)
      - *(MPE) Fast Performance Prediction for Efficient Distributed DNN Training* (x)
      - *PALEO: A PERFORMANCE MODEL FOR DEEP NEURAL NETWORKS* (ICLR17)

17. **Survey**:
         - *(fine-tuning survey) Learn From Model Beyond Fine-Tuning: A Survey* (arxiv23)
         - *(peft survey) Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey* (arxiv24)

18. **Quantization**:
        - *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale* (NIPS22)
        - *8-BIT OPTIMIZERS VIA BLOCK-WISE QUANTIZATION* (ICLR22)

19. **Sparsity**
       - *PIT: Optimization of Dynamic Sparse Deep Learning Models via Permutation Invariant Transformation* (SOSP23)
