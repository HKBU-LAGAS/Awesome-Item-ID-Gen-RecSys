<h1 align="center">🌟 Awesome Item Identification and Item Tokenization in Gen-RecSys 🌟</h1>
<div align="center">

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 
[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

📌 Pinned

 We are actively tracking the latest research and will keep maintaining and updating the repository. We also highly welcome contributions to our repository and survey paper. If your studies are relevant, please feel free to create an issue or a pull request.
#### Survey Paper: A Survey of Item Identifiers in Generative Recommendation: Construction, Alignment, and Generation

[ReseachGate](https://www.researchgate.net/publication/399898960_A_Survey_of_Item_Identifiers_in_Generative_Recommendation_Construction_Alignment_and_Generation) [TechRxiv](https://www.techrxiv.org/doi/full/10.36227/techrxiv.176945895.52184668/)

</div>



## Contents

- [Contents](#contents)
- [📑 Paper List](#-paper-list)
  - [🦄 Creation of Item IDs](#-creation-of-item-ids)
    - [Types of Item IDs](#types-of-item-ids)
      - [Numeric IDs](#numeric-ids)
      - [Textual IDs](#textual-ids)
      - [Multi-facet IDs](#multi-facet-ids)
      - [Semantic IDs](#semantic-ids)
    - [SID Tokenization](#sid-tokenization)
      - [Cluster-based Tokenizers](#cluster-based-tokenizers)
      - [Codebook-based Tokenizers](#codebook-based-tokenizers)
      - [Other Tokenizers](#other-tokenizers)
      - [Augmentation](#augmentation)
  - [👫 Alignment of Item IDs](#-alignment-of-item-ids)
    - [Alignment in LLM-as-Gen-RecSys](#alignment-in-llm-as-gen-recsys)
      - [Collaborative Alignment](#collaborative-alignment)
      - [Multi-Modal Alignment](#multi-modal-alignment)
    - [Alignment in SID-based Gen-RecSys](#alignment-in-sid-based-gen-recsys)
      - [Multi-Task Training](#multi-task-training)
      - [Joint Training](#joint-training)
      - [Others](#others)
  - [🔮 Generation of Next Item IDs](#-generation-of-next-item-ids)
    - [Validity-Aware Generation](#validity-aware-generation)
      - [Direct Generation (DG)](#direct-generation-dg)
      - [Closed-Set Generation (CSG)](#closed-set-generation-csg)
      - [Post-Grounding (PG)](#post-grounding-pg)
      - [Constrained Decoding (CD)](#constrained-decoding-cd)
    - [Decoding Acceleration](#decoding-acceleration)
      - [Speculative Decoding (SD)](#speculative-decoding-sd)
      - [Pruned Search (PS)](#pruned-search-ps)
      - [Parallel Decoding (PD)](#parallel-decoding-pd)
      - [General Optimizations (GO)](#general-optimizations-go)
  - [🗂️ Benchmarks and Datasets](#️-benchmarks-and-datasets)
## 📑 Paper List

### 🦄 Creation of Item IDs


#### Types of Item IDs
##### Numeric IDs
*   (P5) **Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5).** RecSys 2022. [[paper](https://dl.acm.org/doi/10.1145/3523227.3546767)][[code](https://github.com/jeykigung/P5)]![GitHub Repo stars](https://img.shields.io/github/stars/jeykigung/P5)
*   (E4SRec) **E4SRec: An Elegant Effective Efficient Extensible Solution of Large Language Models for Sequential Recommendation.** arXiv 2023. [[paper](https://arxiv.org/abs/2312.02443)][[code](https://github.com/HestiaSky/E4SRec/)]![GitHub Repo stars](https://img.shields.io/github/stars/HestiaSky/E4SRec)
*   (CID) **How to Index Item IDs for Recommendation Foundation Models.** SIGIR 2023. [[paper](https://dl.acm.org/doi/10.1145/3624918.3625339)][[code](https://github.com/Wenyueh/LLM-RecSys-ID)]![GitHub Repo stars](https://img.shields.io/github/stars/Wenyueh/LLM-RecSys-ID)
*   (PAP-Rec) **Pap-rec: Personalized automatic prompt for recommendation language model.** arXiv 2024. [[paper](https://arxiv.org/abs/2402.00284)][[code](https://github.com/rutgerswiselab/PAP-REC)]![GitHub Repo stars](https://img.shields.io/github/stars/rutgerswiselab/PAP-REC)
*   (SID-Token) **Better Generalization with Semantic Ids: A Case Study in Ranking for Recommendations.** RecSys 2024. [[paper](https://dl.acm.org/doi/10.1145/3640457.3688190)]
*   (A-LLMRec) **Large language models meet collaborative filtering: An efficient all-round llm-based recommender system.** KDD 2024. [[paper](https://dl.acm.org/doi/10.1145/3637528.3671931)][[code](https://github.com/ghdtjr/A-LLMRec)]![GitHub Repo stars](https://img.shields.io/github/stars/ghdtjr/A-LLMRec)

##### Textual IDs
*   (PALR) **Palr: Personalization aware llms for recommendation.** arXiv 2023. [[paper](https://arxiv.org/abs/2305.07622)]
*   (PBNR) **Pbnr: Prompt-based news recommender system.** arXiv 2023. [[paper](https://arxiv.org/abs/2304.07862)]
*   (BookGPT) **Bookgpt: A general framework for book recommendation empowered by large language model.** Electronics 2023. [[paper](https://arxiv.org/abs/2305.15673)][[code](https://github.com/zhiyulee-RUC/bookgpt)]![GitHub Repo stars](https://img.shields.io/github/stars/zhiyulee-RUC/bookgpt)
*   (ChatGPT-News) **A preliminary study of chatgpt on news recommendation: Personalization, provider fairness, fake news.** arXiv 2023. [[paper](https://arxiv.org/abs/2306.10702)][[code](https://github.com/imrecommender/ChatGPT-News)]![GitHub Repo stars](https://img.shields.io/github/stars/imrecommender/ChatGPT-News)
*   (LLMRank) **Large language models are zero-shot rankers for recommender systems.** ECIR 2024. [[paper](https://dl.acm.org/doi/10.1007/978-3-031-56060-6_24)][[code](https://github.com/RUCAIBox/LLMRank)]![GitHub Repo stars](https://img.shields.io/github/stars/RUCAIBox/LLMRank)
*   (GenRec) **Genrec: Large language model for generative recommendation.** ECIR 2024. [[paper](https://dl.acm.org/doi/abs/10.1007/978-3-031-56063-7_42)][[code](https://github.com/rutgerswiselab/GenRec)]![GitHub Repo stars](https://img.shields.io/github/stars/rutgerswiselab/GenRec)
*   (LLaRA) **LLaRA: Large Language-Recommendation Assistant.** SIGIR 2024. [[paper](https://dl.acm.org/doi/10.1145/3626772.3657690)][[code](https://github.com/ljy0ustc/LLaRA)]![GitHub Repo stars](https://img.shields.io/github/stars/ljy0ustc/LLaRA)
*   (BIGRec) **A Bi-Step Grounding Paradigm for Large Language Models in Recommendation Systems.** TORS 2025. [[paper](https://dl.acm.org/doi/10.1145/3716393)][[code](https://github.com/SAI990323/BIGRec)]![GitHub Repo stars](https://img.shields.io/github/stars/SAI990323/BIGRec)
*   (Instruct-Rec) **Recommendation as Instruction Following: A Large Language Model Empowered Recommendation Approach.** TOIS 2025. [[paper](https://dl.acm.org/doi/10.1145/3708882)]
  
##### Multi-facet IDs

* (Chat-REC) **Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System.** arXiv 2023. [[paper](https://arxiv.org/abs/2303.14524)]
* (TransRec) **Bridging Items and Language: A Transition Paradigm for Large Language Model-based Recommendation.** KDD 2024. [[paper](https://arxiv.org/pdf/2310.06491)] [[code](https://github.com/Linxyhaha/TransRec/)]![GitHub Repo stars](https://img.shields.io/github/stars/Linxyhaha/TransRec)
* (CLLM4Rec) **Collaborative Large Language Model for Recommender Systems.** WWW 2024. [[paper](https://dl.acm.org/doi/10.1145/3589334.3645347)] [[code](https://github.com/yaochenzhu/llm4rec)]![GitHub Repo stars](https://img.shields.io/github/stars/yaochenzhu/llm4rec)
* (LLM4POI) **Large language models for next point-of-interest recommendation.** arXiv 2024. [[paper](https://arxiv.org/pdf/2404.17591)]
* (ID-Lang-Barrier) **Break the ID-Language Barrier: An Adaption Framework for Sequential Recommendation.** arXiv 2024. [[paper](https://arxiv.org/abs/2411.18262)]
* (GNPR-SID) **Generative next POI Recommendation with Semantic ID** KDD 2025. [[paper](https://arxiv.org/abs/2506.01375)][[code](https://github.com/wds1996/GNPR-SID)]![GitHub Repo stars](https://img.shields.io/github/stars/wds1996/GNPR-SID)
* (CoLLM) **CoLLM: Integrating Collaborative Embeddings into Large Language Models for Recommendation.** TKDE 2025. [[paper](https://dl.acm.org/doi/10.1109/TKDE.2025.3540912)] [[code](https://github.com/zyang1580/CoLLM)]![GitHub Repo stars](https://img.shields.io/github/stars/zyang1580/CoLLM)
* (URM) **Large Language Model as Universal Retriever in Industrial-Scale Recommender System.** arXiv 2025. [[paper](https://arxiv.org/abs/2502.03041)]



##### Semantic IDs

* (VQ-Rec) **Learning vector-quantized item representation for transferable sequential recommenders.** WWW 2023. [[paper](https://arxiv.org/abs/2210.12316)][[code](https://github.com/RUCAIBox/VQ-Rec)] ![GitHub Repo stars](https://img.shields.io/github/stars/RUCAIBox/VQ-Rec)
* (Tiger) **Recommender Systems with Generative Retrieval.** NeurIPS 2023. [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/20dcab0f14046a5c6b02b61da9f13229-Abstract-Conference.html)] [[code](https://github.com/EdoardoBotta/RQ-VAE-Recommender)] ![GitHub Repo stars](https://img.shields.io/github/stars/EdoardoBotta/RQ-VAE-Recommender)
* (ColaRec) **Content-Based Collaborative Generation for Recommender Systems.** CIKM 2024. [[paper](https://arxiv.org/abs/2403.18480)][[code](https://github.com/Junewang0614/ColaRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Junewang0614/ColaRec)
* (LETTER) **Learnable Item Tokenization for Generative Recommendation.** CIKM 2024. [[paper](https://dl.acm.org/doi/10.1145/3627673.3679569)][[code](https://github.com/HonghuiBao2000/LETTER)] ![GitHub Repo stars](https://img.shields.io/github/stars/HonghuiBao2000/LETTER)
* (EAGER) **EAGER: Two-Stream Generative Recommender with Behavior-Semantic Collaboration.** KDD 2024. [[paper](https://dl.acm.org/doi/10.1145/3637528.3671775)][[code](https://github.com/yewzz/EAGER)] ![GitHub Repo stars](https://img.shields.io/github/stars/yewzz/EAGER)
* (MMGRec) **MMGRec: Multimodal Generative Recommendation with Transformer Model.** arXiv 2024. [[paper](https://arxiv.org/abs/2404.16555)]
* (OneRec) **OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment.** arXiv 2025. [[paper](https://arxiv.org/abs/2502.18965)][[code](https://github.com/Kuaishou-OneRec/OpenOneRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Kuaishou-OneRec/OpenOneRec)
* (UTGRec) **Universal Item Tokenization for Transferable Generative Recommendation.** arXiv 2025. [[paper](https://arxiv.org/abs/2504.04405)][[code](https://github.com/RUCAIBox/UTGRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/RUCAIBox/UTGRec)
* (RPG) **Generating Long Semantic Ids in Parallel for Recommendation.** KDD 2025. [[paper](https://arxiv.org/abs/2506.05781)][[code](https://github.com/facebookresearch/RPG_KDD2025)] ![GitHub Repo stars](https://img.shields.io/github/stars/facebookresearch/RPG_KDD2025)
* (FORGE) **FORGE: Forming Semantic Identifiers for Generative Retrieval in Industrial Datasets.** arXiv 2025. [[paper](https://arxiv.org/abs/2509.20904)][[code](https://github.com/selous123/al_sid)] ![GitHub Repo stars](https://img.shields.io/github/stars/selous123/al_sid)
* (PLUM) **PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations.** arXiv 2025. [[paper](https://arxiv.org/abs/2510.07784)]

#### SID Tokenization

##### Cluster-based Tokenizers 
* (CID) **How to Index Item IDs for Recommendation Foundation Models.** SIGIR 2023. [[paper](https://arxiv.org/pdf/2305.06569)] [[code](https://github.com/Wenyueh/LLM-RecSys-ID)] ![GitHub Repo stars](https://img.shields.io/github/stars/Wenyueh/LLM-RecSys-ID)
* (SEATER) **Generative Retrieval with Semantic Tree-Structured Item Identifiers via Contrastive Learning.** arXiv 2023. [[paper](https://arxiv.org/pdf/2309.13375)] [[code](https://github.com/Ethan00Si/SEATER_Generative_Retrieval)] ![GitHub Repo stars](https://img.shields.io/github/stars/Ethan00Si/SEATER_Generative_Retrieval)
* (ColaRec) **Content-Based Collaborative Generation for Recommender Systems.** CIKM 2024. [[paper](https://arxiv.org/abs/2403.18480)][[code](https://github.com/Junewang0614/ColaRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Junewang0614/ColaRec)
* (EAGER) **EAGER: Two-Stream Generative Recommender with Behavior-Semantic Collaboration.** KDD 2024. [[paper](https://dl.acm.org/doi/10.1145/3637528.3671775)][[code](https://github.com/yewzz/EAGER)] ![GitHub Repo stars](https://img.shields.io/github/stars/yewzz/EAGER)
* (MERGE) **MERGE: Next-Generation Item Indexing Paradigm for Large-Scale Streaming Recommendation** [[paper](https://arxiv.org/abs/2601.20199)]
##### Codebook-based Tokenizers 
* (VQ-Rec) **Learning vector-quantized item representation for transferable sequential recommenders.** WWW 2023. [[paper](https://arxiv.org/abs/2210.12316)][[code](https://github.com/RUCAIBox/VQ-Rec)] ![GitHub Repo stars](https://img.shields.io/github/stars/RUCAIBox/VQ-Rec)
* (Tiger) **Recommender Systems with Generative Retrieval.** NeurIPS 2023. [[paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/20dcab0f14046a5c6b02b61da9f13229-Abstract-Conference.html)] [[code](https://github.com/EdoardoBotta/RQ-VAE-Recommender)] ![GitHub Repo stars](https://img.shields.io/github/stars/EdoardoBotta/RQ-VAE-Recommender)
* (CoST) **CoST: Contrastive Quantization Based Semantic Tokenization for Generative Recommendation** RecSys 2024. [[paper](https://arxiv.org/abs/2404.14774)]
* (MBGen) **Multi-Behavior Generative Recommendation** CIKM 2024 [[paper](https://arxiv.org/abs/2405.16871)] [[code](https://github.com/anananan116/MBGen)] ![GitHub Repo stars](https://img.shields.io/github/stars/anananan116/MBGen)
* (LMIndexer) **Language Models as Semantic Indexers** [[paper](https://arxiv.org/abs/2310.07815)][[code](https://github.com/PeterGriffinJin/LMIndexer)]![GitHub Repo stars](https://img.shields.io/github/stars/PeterGriffinJin/LMIndexer)
* (UIST) **Discrete Semantic Tokenization for Deep CTR Prediction** WWW 2024 [[paper](https://arxiv.org/abs/2403.08206)] [[code](https://github.com/Jyonn/SemanticTokenizer)] ![GitHub Repo stars](https://img.shields.io/github/stars/Jyonn/SemanticTokenizer)
* (ETEGRec) **Generative Recommender with End-to-End Learnable Item Tokenization** SIGIR 2025 [[paper](https://arxiv.org/abs/2409.05546)] [[code](https://github.com/RUCAIBox/ETEGRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/RUCAIBox/ETEGRec)
* (COBRA) **Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations** arXiv 2025. [[paper](https://arxiv.org/abs/2503.02453)]
* (BBQRec) **BBQRec: Behavior-Bind Quantization for Multi-Modal Sequential Recommendation** arXiv 2025. [[paper](https://arxiv.org/abs/2504.06636)]
* (UTGRec) **Universal Item Tokenization for Transferable Generative Recommendation.** arXiv 2025. [[paper](https://arxiv.org/abs/2504.04405)][[code](https://github.com/RUCAIBox/UTGRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/RUCAIBox/UTGRec)
* (DiscRec) **DiscRec: Disentangled Semantic-Collaborative Modeling for Generative Recommendation.** arXiv 2025. [[paper](https://arxiv.org/pdf/2506.15576)] [[code](https://github.com/Ten-Mao/DiscRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Ten-Mao/DiscRec)
* (MMQ) **MMQ: Multimodal Mixture-of-Quantization Tokenization for Semantic ID Generation and User Behavioral Adaptation** WSDM 2026 [[paper](https://arxiv.org/abs/2508.15281)]
* (DQ-VAE) **Representation Quantization for Collaborative Filtering Augmentation** [[paper](https://arxiv.org/abs/2508.11194)]
* (HiD-VAE) **HiD-VAE: Interpretable Generative Recommendation via Hierarchical and Disentangled Semantic IDs.** arXiv 2025. [[paper](https://arxiv.org/pdf/2508.04618)]
* (OneLoc) **OneLoc: Geo-Aware Generative Recommender Systems for Local Life Service** arXiv 2025. [[paper](https://arxiv.org/abs/2508.14646)]
* (PCR-CA) **PCR-CA: Parallel Codebook Representations with Contrastive Alignment for Multiple-Category App Recommendation** arXiv 2025 [[paper](https://arxiv.org/abs/2508.18166)]
* (COSETTE) **Closing the Performance Gap in Generative Recommenders with Collaborative Tokenization and Efficient Modeling** Arxiv 2025 [[paper](https://arxiv.org/abs/2508.14910)]
* (PSRQ) **Progressive Semantic Residual Quantization for Multimodal-Joint Interest Modeling in Music Recommendation** CIKM 2025 [[paper](https://arxiv.org/abs/2508.20359)]
* (OneSearch) **Onesearch: A preliminary exploration of the unified end-to-end generative framework for e-commerce search** Arxiv 2025 [[paper](https://arxiv.org/abs/2509.03236)]
* (MMQ-v2) **MMQ-v2: Align, Denoise, and Amplify: Adaptive Behavior Mining for Semantic IDs Learning in Recommendation** Arxiv 2025 [[paper](https://arxiv.org/abs/2510.25622)]
* (STORE) **The Best of the Two Worlds: Harmonizing Semantic and Hash IDs for Sequential Recommendation** Arxiv 2025[[paper](https://arxiv.org/abs/2511.18805)]
* (ReaSeq) **ReaSeq: Unleashing World Knowledge via Reasoning for Sequential Modeling** Arxiv 2025 [[paper](https://arxiv.org/abs/2512.21257)]
* (HiGR) **HiGR: Efficient Generative Slate Recommendation via Hierarchical Planning and Multi-Objective Preference Alignment** [[paper](https://www.arxiv.org/abs/2512.24787)]
* (S²GR) **S²GR: Stepwise Semantic-Guided Reasoning in Latent Space for Generative Recommendation** Arxiv 2026 [[paper](https://www.arxiv.org/abs/2601.18664)]
* (TRM) **Farewell to Item IDs: Unlocking the Scaling Potential of Large Ranking Models via Semantic Tokens** Arxiv 2026 [[paper](https://arxiv.org/abs/2601.22694)]
* (GR2) **Generative Reasoning Re-ranker** Arxiv 2026 [[paper](https://arxiv.org/abs/2602.07774)]
##### Other Tokenizers 
* (GPTRec) **Generative Sequential Recommendation with GPTRec.** Gen-IR @ SIGIR 2023 workshop [[paper](https://arxiv.org/abs/2306.11114)]
* (IDGenRec) **IDGenRec: LLM-RecSys Alignment with Textual ID Learning** SIGIR 2024 [[paper](https://arxiv.org/abs/2403.19021)] [[code](https://github.com/agiresearch/IDGenRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/agiresearch/IDGenRec)
* (ActionPiece) **ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation** ICML 2025 [[paper](https://arxiv.org/abs/2502.13581)] [[code](https://github.com/google-deepmind/action_piece)] ![GitHub Repo stars](https://img.shields.io/github/stars/google-deepmind/action_piece)
* (SETRec) **Order-Agnostic Identifier for Large Language Model-based Generative Recommendation.** SIGIR 2025. [[paper](https://dl.acm.org/doi/10.1145/3726302.3730053)] [[code](https://github.com/Linxyhaha/SETRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Linxyhaha/SETRec)

##### Augmentation 
* (MMGRec) **MMGRec: Multimodal Generative Recommendation with Transformer Model.** arXiv 2024. [[paper](https://arxiv.org/abs/2404.16555)]
* (LETTER) **Learnable Item Tokenization for Generative Recommendation.** CIKM 2024. [[paper](https://dl.acm.org/doi/10.1145/3627673.3679569)][[code](https://github.com/HonghuiBao2000/LETTER)] ![GitHub Repo stars](https://img.shields.io/github/stars/HonghuiBao2000/LETTER)
* (DiscRec) **DiscRec: Disentangled Semantic-Collaborative Modeling for Generative Recommendation.** arXiv 2025. [[paper](https://arxiv.org/pdf/2506.15576)] [[code](https://github.com/Ten-Mao/DiscRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Ten-Mao/DiscRec)
* (COSETTE) **Closing the Performance Gap in Generative Recommenders with Collaborative Tokenization and Efficient Modeling** Arxiv 2025 [[paper](https://arxiv.org/abs/2508.14910)]
* (PSRQ) **Progressive Semantic Residual Quantization for Multimodal-Joint Interest Modeling in Music Recommendation** CIKM 2025 [[paper](https://arxiv.org/abs/2508.20359)]
* (OneSearch) **Onesearch: A preliminary exploration of the unified end-to-end generative framework for e-commerce search** Arxiv 2025 [[paper](https://arxiv.org/abs/2509.03236)]
* (MMQ) **MMQ: Multimodal Mixture-of-Quantization Tokenization for Semantic ID Generation and User Behavioral Adaptation** WSDM 2026 [[paper](https://arxiv.org/abs/2508.15281)]
* (MMQ-v2) **MMQ-v2: Align, Denoise, and Amplify: Adaptive Behavior Mining for Semantic IDs Learning in Recommendation** Arxiv 2025 [[paper](https://arxiv.org/abs/2510.25622)]
### 👫 Alignment of Item IDs 
#### Alignment in LLM-as-Gen-RecSys

##### Collaborative Alignment
* (E4SRec) **E4SRec: An Elegant Effective Efficient Extensible Solution of Large Language Models for Sequential Recommendation.** arXiv 2023. [[paper](https://arxiv.org/abs/2312.02443)] [[code](https://github.com/HestiaSky/E4SRec/)] [![GitHub stars](https://img.shields.io/github/stars/HestiaSky/E4SRec?style=social)](https://github.com/HestiaSky/E4SRec/)
* (CLLM4Rec) **Collaborative Large Language Model for Recommender Systems.** WWW 2024. [[paper](https://dl.acm.org/doi/10.1145/3589334.3645347)] [[code](https://github.com/yaochenzhu/llm4rec)] [![GitHub stars](https://img.shields.io/github/stars/yaochenzhu/llm4rec?style=social)](https://github.com/yaochenzhu/llm4rec)
* (LC-Rec) **Adapting Large Language Models by Integrating Collaborative Semantics for Recommendation.** ICDE 2024. [[paper](https://ieeexplore.ieee.org/document/10597986)] [[code](https://github.com/RUCAIBox/LC-Rec)] [![GitHub stars](https://img.shields.io/github/stars/RUCAIBox/LC-Rec?style=social)](https://github.com/RUCAIBox/LC-Rec)
* (LLaRA) **LLaRA: Large Language-Recommendation Assistant.** SIGIR 2024. [[paper](https://dl.acm.org/doi/10.1145/3626772.3657690)] [[code](https://github.com/ljy0ustc/LLaRA)] [![GitHub stars](https://img.shields.io/github/stars/ljy0ustc/LLaRA?style=social)](https://github.com/ljy0ustc/LLaRA)
* (A-LLMRec) **Large Language Models meet Collaborative Filtering: An Efficient All-round LLM-based Recommender System.** KDD 2024. [[paper](https://dl.acm.org/doi/10.1145/3637528.3671931)] [[code](https://github.com/ghdtjr/A-LLMRec)] [![GitHub stars](https://img.shields.io/github/stars/ghdtjr/A-LLMRec?style=social)](https://github.com/ghdtjr/A-LLMRec)
* (IDLE-Adapter) **Break the ID-Language Barrier: An Adaption Framework for Sequential Recommendation.** arXiv 2024. [[paper](https://arxiv.org/abs/2411.18262)]
* (EAGER-LLM) **EAGER-LLM: Enhancing Large Language Models as Recommenders through Exogenous Behavior-Semantic Integration.** WWW 2025. [[paper](https://dl.acm.org/doi/10.1145/3696410.3714933)]
* (CoLLM) **CoLLM: Integrating Collaborative Embeddings into Large Language Models for Recommendation.** TKDE 2025. [[paper](https://dl.acm.org/doi/10.1109/TKDE.2025.3540912)] [[code](https://github.com/zyang1580/CoLLM)] [![GitHub stars](https://img.shields.io/github/stars/zyang1580/CoLLM?style=social)](https://github.com/zyang1580/CoLLM)
* (Align³GR) **Align³GR: Unified Multi-Level Alignment for LLM-based Generative Recommendation.** AAAI 2026. [[paper](https://arxiv.org/abs/2511.11255)]

##### Multi-Modal Alignment
* (I-LLMRec) **Image is All You Need: Towards Efficient and Effective Large Language Model-based Recommender Systems.** arXiv 2025. [[paper](https://arxiv.org/abs/2503.06238)] [[code](https://github.com/rlqja1107/torch-I-LLMRec)] [![GitHub stars](https://img.shields.io/github/stars/rlqja1107/torch-I-LLMRec?style=social)](https://github.com/rlqja1107/torch-I-LLMRec)


#### Alignment in SID-based Gen-RecSys

##### Multi-Task Training
* (ColaRec) **Content-Based Collaborative Generation for Recommender Systems.** CIKM 2024. [[paper](https://arxiv.org/abs/2403.18480)][[code](https://github.com/Junewang0614/ColaRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Junewang0614/ColaRec)
* (MBGen) **Multi-Behavior Generative Recommendation** CIKM 2024 [[paper](https://arxiv.org/abs/2405.16871)] [[code](https://github.com/anananan116/MBGen)] ![GitHub Repo stars](https://img.shields.io/github/stars/anananan116/MBGen)
* (Eager) **Eager: Two-stream Generative Recommender with Behavior-semantic Collaboration** KDD 2024. [[paper](https://arxiv.org/abs/2406.14017)]
* (SemanticConvergence) **Semantic Convergence: Harmonizing Recommender Systems via Two-stage Alignment and Behavioral Semantic Tokenization** AAAI 2025 [[paper](https://arxiv.org/abs/2412.13771)]

##### Joint Training
* (IDGenRec) **IDGenRec: LLM-RecSys Alignment with Textual ID Learning** SIGIR 2024 [[paper](https://arxiv.org/abs/2403.19021)] [[code](https://github.com/agiresearch/IDGenRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/agiresearch/IDGenRec)
* (ETEGRec) **Generative Recommender with End-to-End Learnable Item Tokenization** SIGIR 2025 [[paper](https://arxiv.org/abs/2409.05546)] [[code](https://github.com/RUCAIBox/ETEGRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/RUCAIBox/ETEGRec)
* (COBRA) **Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations** arXiv 2025. [[paper](https://arxiv.org/abs/2503.02453)]
* (MMQ) **MMQ: Multimodal Mixture-of-Quantization Tokenization for Semantic ID Generation and User Behavioral Adaptation** WSDM 2026 [[paper](https://arxiv.org/abs/2508.15281)]

##### Others
* (TokenRec) **TokenRec: Learning to Tokenize ID for LLM-based Generative Recommendations** TKDE 2025 [[paper](https://arxiv.org/abs/2406.10450)] [[code](https://github.com/Quhaoh233/TokenRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Quhaoh233/TokenRec)
* (BBQRec) **BBQRec: Behavior-Bind Quantization for Multi-Modal Sequential Recommendation** arXiv 2025. [[paper](https://arxiv.org/abs/2504.06636)]

### 🔮 Generation of Next Item IDs

#### Validity-Aware Generation 

##### Direct Generation (DG)
* (P5) **Recommendation as Language Processing (RLP): A Unified Pretrain, Personalized Prompt & Predict Paradigm (P5).** RecSys 2022. [[paper](https://arxiv.org/abs/2203.13366)] [[code](https://github.com/jeykigung/P5)] ![GitHub Repo stars](https://img.shields.io/github/stars/jeykigung/P5)
* (GenRec) **GenRec: Large Language Model for Generative Recommendation.** ECIR 2024. [[paper](https://arxiv.org/pdf/2307.00457)] [[code](https://github.com/rutgerswiselab/GenRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/rutgerswiselab/GenRec)
* (DEALRec) **Data-Efficient Fine-Tuning for LLM-based Recommendation.** SIGIR 2024. [[paper](https://arxiv.org/pdf/2401.17197)] [[code](https://github.com/Linxyhaha/DEALRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Linxyhaha/DEALRec)
* (LLM4POI) **Large language models for next point-of-interest recommendation.** arXiv 2024. [[paper](https://arxiv.org/pdf/2404.17591)]

##### Closed-Set Generation (CSG)
* (Chat-REC) **Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System.** arXiv 2023. [[paper](https://arxiv.org/pdf/2303.14524)]
* (TALLRec) **TALLRec: An Effective and Efficient Tuning Framework to Align Large Language Model with Recommendation.** RecSys 2023. [[paper](https://arxiv.org/pdf/2305.00447)] [[code](https://github.com/SAI990323/TALLRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/SAI990323/TALLRec)
* (CLLM4Rec) **Collaborative Large Language Model for Recommender Systems.** WWW 2024. [[paper](https://arxiv.org/pdf/2311.01343)] [[code](https://github.com/yaochenzhu/llm4rec)] ![GitHub Repo stars](https://img.shields.io/github/stars/yaochenzhu/llm4rec)
* (InstructRec) **Recommendation as Instruction Following: A Large Language Model Empowered Recommendation Approach.** TOIS 2025. [[paper](https://arxiv.org/abs/2305.07001)]
* (ContRec) **Generative Recommendation with Continuous-Token Diffusion.** arXiv 2025. [[paper](https://arxiv.org/pdf/2504.12007)] [[code](https://github.com/Quhaoh233/ContRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Quhaoh233/ContRec)

##### Post-Grounding (PG)
* (BIGRec) **A Bi-step Grounding Paradigm for Large Language Models in Recommendation Systems.** TORS 2025. [[paper](https://dl.acm.org/doi/10.1145/3716393)] [[code](https://github.com/SAI990323/BIGRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/SAI990323/BIGRec)
* (SETRec) **Order-Agnostic Identifier for Large Language Model-based Generative Recommendation.** SIGIR 2025. [[paper](https://dl.acm.org/doi/10.1145/3726302.3730053)] [[code](https://github.com/Linxyhaha/SETRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Linxyhaha/SETRec)

##### Constrained Decoding (CD)
* (CID) **How to Index Item IDs for Recommendation Foundation Models.** SIGIR 2023. [[paper](https://arxiv.org/pdf/2305.06569)] [[code](https://github.com/Wenyueh/LLM-RecSys-ID)] ![GitHub Repo stars](https://img.shields.io/github/stars/Wenyueh/LLM-RecSys-ID)
* (SEATER) **Generative Retrieval with Semantic Tree-Structured Item Identifiers via Contrastive Learning.** arXiv 2023. [[paper](https://arxiv.org/pdf/2309.13375)] [[code](https://github.com/Ethan00Si/SEATER_Generative_Retrieval)] ![GitHub Repo stars](https://img.shields.io/github/stars/Ethan00Si/SEATER_Generative_Retrieval)
* (IDGenRec) **IDGenRec: LLM-RecSys Alignment with Textual ID Learning.** SIGIR 2024. [[paper](https://dl.acm.org/doi/10.1145/3626772.3657821)] [[code](https://github.com/agiresearch/IDGenRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/agiresearch/IDGenRec)
* (TransRec) **Bridging Items and Language: A Transition Paradigm for Large Language Model-based Recommendation.** KDD 2024. [[paper](https://arxiv.org/pdf/2310.06491)] [[code](https://github.com/Linxyhaha/TransRec/)] ![GitHub Repo stars](https://img.shields.io/github/stars/Linxyhaha/TransRec)
* (ColaRec) **Content-based Collaborative Generation for Recommender Systems.** CIKM 2024. [[paper](https://arxiv.org/pdf/2403.18480)] [[code](https://github.com/Junewang0614/ColaRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Junewang0614/ColaRec)
* (RPG) **Generating Long Semantic IDs in Parallel for Recommendation.** KDD 2025. [[paper](https://arxiv.org/pdf/2506.05781)] [[code](https://github.com/facebookresearch/RPG_KDD2025)] ![GitHub Repo stars](https://img.shields.io/github/stars/facebookresearch/RPG_KDD2025)
* (DiscRec) **DiscRec: Disentangled Semantic-Collaborative Modeling for Generative Recommendation.** arXiv 2025. [[paper](https://arxiv.org/pdf/2506.15576)] [[code](https://github.com/Ten-Mao/DiscRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Ten-Mao/DiscRec)
* (HiD-VAE) **HiD-VAE: Interpretable Generative Recommendation via Hierarchical and Disentangled Semantic IDs.** arXiv 2025. [[paper](https://arxiv.org/pdf/2508.04618)]
* (PROMISE) **PROMISE: Process Reward Models Unlock Test-Time Scaling Laws in Generative Recommendations.** arXiv 2026. [[paper](https://arxiv.org/abs/2601.04674)]
#### Decoding Acceleration

##### Speculative Decoding (SD)
* (SpecGR) **Inductive Generative Recommendation via Retrieval-based Speculation** AAAI 2026 [[paper](https://arxiv.org/abs/2410.02939)] [[code](https://github.com/Jamesding000/SpecGR)] ![GitHub Repo stars](https://img.shields.io/github/stars/Jamesding000/SpecGR)
* (AtSpeed) **Efficient Inference for Large Language Model-Based Generative Recommendation** ICLR 2025 [[paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/e4bf5c3245fd92a4554a16af9803b757-Paper-Conference.pdf)] [[code](https://github.com/Linxyhaha/AtSpeed)] ![GitHub Repo stars](https://img.shields.io/github/stars/Linxyhaha/AtSpeed)
* (RPG) **Generating Long Semantic Ids in Parallel for Recommendation** KDD 2025 [[paper](https://arxiv.org/abs/2506.05781)] [[code](https://github.com/facebookresearch/RPG_KDD2025)] ![GitHub Repo stars](https://img.shields.io/github/stars/facebookresearch/RPG_KDD2025)
* (NEZHA) **NEZHA: A Zero-sacrifice and Hyperspeed Decoding Architecture for Generative Recommendations** arXiv 2025 [[paper](https://arxiv.org/abs/2511.18793)]

##### Pruned Search (PS)
* (Semantic-ID-Generation) **Better Generalization with Semantic Ids: A Case Study in Ranking for Recommendations** RecSys 2024 [[paper](https://arxiv.org/abs/2306.08121)] [[code](https://github.com/justinhangoebl/Semantic-ID-Generation)] ![GitHub Repo stars](https://img.shields.io/github/stars/justinhangoebl/Semantic-ID-Generation)
* (COBRA) **Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations** arXiv 2025 [[paper](https://arxiv.org/abs/2503.02453)]
* (Text2Tracks) **Text2Tracks: Prompt-based Music Recommendation via Generative Retrieval** arXiv 2025 [[paper](https://arxiv.org/abs/2503.24193)] [[code](https://github.com/mayurbhangale/text2tracks)] ![GitHub Repo stars](https://img.shields.io/github/stars/mayurbhangale/text2tracks)
* (GRAM) **Generative Retrieval and Alignment Model: A New Paradigm for E-commerce Retrieval** WWW 2025 [[paper](https://arxiv.org/abs/2504.01403)]
* (GenSAR) **Unified Generative Search and Recommendation** RecSys 2025 [[paper](https://arxiv.org/abs/2504.05730)]
* (MTGR) **MTGR: Industrial-Scale Generative Recommendation Framework in Meituan** CIKM 2025 [[paper](https://arxiv.org/abs/2505.18654)]
* (DiscRec) **DiscRec: Disentangled Semantic-Collaborative Modeling for Generative Recommendation** arXiv 2025 [[paper](https://arxiv.org/abs/2506.15576)] [[code](https://github.com/Ten-Mao/DiscRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Ten-Mao/DiscRec)
* (HLLM-Creator) **HLLM-Creator: Hierarchical LLM-based Personalized Creative Generation** arXiv 2025 [[paper](https://arxiv.org/abs/2508.18118)] [[code](https://github.com/bytedance/HLLM)] ![GitHub Repo stars](https://img.shields.io/github/stars/bytedance/HLLM)
* (FORGE) **FORGE: Forming Semantic Identifiers for Generative Retrieval in Industrial Datasets** arXiv 2025 [[paper](https://arxiv.org/abs/2509.20904)] [[code](https://github.com/selous123/al_sid)] ![GitHub Repo stars](https://img.shields.io/github/stars/selous123/al_sid)
* (UniSearch) **UniSearch: Rethinking Search System with a Unified Generative Architecture** arXiv 2025 [[paper](https://arxiv.org/abs/2509.06887)]
* (QARM) **QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou** CIKM 2025 [[paper](https://arxiv.org/abs/2411.11739)]

##### Parallel Decoding (PD)
* (RPG) **Generating Long Semantic Ids in Parallel for Recommendation** KDD 2025 [[paper](https://arxiv.org/abs/2506.05781)] [[code](https://github.com/facebookresearch/RPG_KDD2025)] ![GitHub Repo stars](https://img.shields.io/github/stars/facebookresearch/RPG_KDD2025)
* (SETRec) **Order-Agnostic Identifier for Large Language Model-based Generative Recommendation** SIGIR 2025 [[paper](https://arxiv.org/abs/2502.10833)] [[code](https://github.com/Linxyhaha/SETRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Linxyhaha/SETRec)
* (LLaDA-Rec) **LLaDA-Rec: Discrete Diffusion for Parallel Semantic ID Generation in Generative Recommendation** arXiv 2025 [[paper](http://arxiv.org/abs/2511.06254)] [[code](https://github.com/TengShi-RUC/LLaDA-Rec)] ![GitHub Repo stars](https://img.shields.io/github/stars/TengShi-RUC/LLaDA-Rec)

##### General Optimizations (GO)
* (Chat-REC) **Chat-REC: Towards Interactive and Explainable LLMs-Augmented Recommender System** arXiv 2023 [[paper](https://arxiv.org/abs/2303.14524)]
* (UIST) **Discrete Semantic Tokenization for Deep CTR Prediction** WWW 2024 [[paper](https://arxiv.org/abs/2403.08206)] [[code](https://github.com/Jyonn/SemanticTokenizer)] ![GitHub Repo stars](https://img.shields.io/github/stars/Jyonn/SemanticTokenizer)
* (GenRec) **Genrec: Large language model for generative recommendation** ECIR 2024 [[paper](https://arxiv.org/abs/2307.00457)] [[code](https://github.com/rutgerswiselab/GenRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/rutgerswiselab/GenRec)
* (DealREC) **Data-Efficient Fine-tuning for LLM-based Recommendation** SIGIR 2024 [[paper](https://arxiv.org/abs/2401.17197)] [[code](https://github.com/Linxyhaha/DEALRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/Linxyhaha/DEALRec)
* (A-LLMRec) **Large language models meet collaborative filtering: An efficient all-round llm-based recommender system** KDD 2024 [[paper](https://arxiv.org/abs/2404.11343)] [[code](https://github.com/ghdtjr/A-LLMRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/ghdtjr/A-LLMRec)
* (CoST) **CoST: Contrastive Quantization Based Semantic Tokenization for Generative Recommendation** RecSys 2024. [[paper](https://arxiv.org/abs/2404.14774)]
* (MBGen) **Multi-Behavior Generative Recommendation** CIKM 2024 [[paper](https://arxiv.org/abs/2405.16871)] [[code](https://github.com/anananan116/MBGen)] ![GitHub Repo stars](https://img.shields.io/github/stars/anananan116/MBGen)
* (HLLM) **HLLM: Enhancing Sequential Recommendations via Hierarchical Large Language Models for Item and User Modeling** arXiv 2024 [[paper](https://arxiv.org/abs/2409.12740)] [[code](https://github.com/bytedance/HLLM)] ![GitHub Repo stars](https://img.shields.io/github/stars/bytedance/HLLM)
* (URM) **Large Language Model as Universal Retriever in Industrial-Scale Recommender System** arxiv 2025 [[paper](https://arxiv.org/abs/2502.03041)]
* (ActionPiece) **ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation** ICML 2025 [[paper](https://arxiv.org/abs/2502.13581)] [[code](https://github.com/google-deepmind/action_piece)] ![GitHub Repo stars](https://img.shields.io/github/stars/google-deepmind/action_piece)
* (CCFRec) **Bridging Textual-Collaborative Gap through Semantic Codes for Sequential Recommendation** KDD 2025 [[paper](https://arxiv.org/abs/2503.12183)] [[code](https://github.com/RUCAIBox/CCFRec)] ![GitHub Repo stars](https://img.shields.io/github/stars/RUCAIBox/CCFRec)
* (BBQRec) **BBQRec: Behavior-Bind Quantization for Multi-Modal Sequential Recommendation** arXiv 2025 [[paper](https://arxiv.org/abs/2504.06636)]
* (GenRank) **Towards Large-scale Generative Ranking** arXiv 2025 [[paper](https://arxiv.org/abs/2505.04180)]
* (HGPO) **LLM-CoT Enhanced Graph Neural Recommendation with Harmonized Group Policy Optimization** arXiv 2025 [[paper](https://arxiv.org/abs/2505.12396)] [[code](https://anonymous.4open.science/r/LLM-Rec)]
* (GNPR-SID) **Generative next POI Recommendation with Semantic ID** KDD 2025. [[paper](https://arxiv.org/abs/2506.01375)] [[code](https://github.com/wds1996/GNPR-SID)] ![GitHub Repo stars](https://img.shields.io/github/stars/wds1996/GNPR-SID)
* (EARN) **EARN: Efficient Inference Acceleration for LLM-based Generative Recommendation by Register Tokens** KDD 2025. [[paper](https://arxiv.org/abs/2507.00715)] [[code](https://github.com/transcend-0/EARN)] ![GitHub Repo stars](https://img.shields.io/github/stars/transcend-0/EARN)
* (OneLoc) **OneLoc: Geo-Aware Generative Recommender Systems for Local Life Service** arXiv 2025. [[paper](https://arxiv.org/abs/2508.14646)]
* (PCR-CA) **PCR-CA: Parallel Codebook Representations with Contrastive Alignment for Multiple-Category App Recommendation** arXiv 2025 [[paper](https://arxiv.org/abs/2508.18166)]
* (Onerecv2) **OneRec-V2 Technical Report** arXiv 2025 [[paper](https://arxiv.org/abs/2508.20900)]
* (PROMISE) **PROMISE: Process Reward Models Unlock Test-Time Scaling Laws in Generative Recommendations.** arXiv 2026. [[paper](https://arxiv.org/abs/2601.04674)]

### 🗂️ Benchmarks and Datasets
* (GRID) **Generative Recommendation with Semantic IDs: A Practitioner’s Handbook** CIKM 2025. [[paper](https://arxiv.org/abs/2507.22224)] [[code](https://github.com/snap-research/GRID)] ![GitHub Repo stars](https://img.shields.io/github/stars/snap-research/GRID)
* (FORGE) **FORGE: Forming Semantic Identifiers for Generative Retrieval in Industrial Datasets** arXiv 2025 [[paper](https://arxiv.org/abs/2509.20904)] [[code](https://github.com/selous123/al_sid)] ![GitHub Repo stars](https://img.shields.io/github/stars/selous123/al_sid)
* (MMGRid) **MMGRid: Navigating Temporal-aware and Cross-domain Generative Recommendation via Model Merging** arXiv 2026 [[paper](https://arxiv.org/abs/2601.15930)] [[code](https://github.com/Joinn99/MMGRid)] ![GitHub Repo stars](https://img.shields.io/github/stars/Joinn99/MMGRid)
  
## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=HKBU-LAGAS/Awesome-Item-ID-Gen-RecSys&type=date&legend=top-left)](https://www.star-history.com/#HKBU-LAGAS/Awesome-Item-ID-Gen-RecSys&type=date&legend=top-left)
