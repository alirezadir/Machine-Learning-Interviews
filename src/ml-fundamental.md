# <a name="breadth"></a> 3. ML Fundamentals (Breadth)
As the name suggests, this interview is intended to evaluate your general knowledge of ML concepts both from theoretical and practical perspectives. Unlike ML depth interviews, the breadth interviews tend to follow a pretty similar structure and coverage amongst different interviewers and interviewees.

The best way to prepare for this interview is to review your notes from ML courses as well some high quality online courses and material. In particular, I found the following resources pretty helpful.

# 1. Courses and review material:
- [Andrew Ng's Machine Learning Course](https://www.coursera.org/learn/machine-learning) (you can also find the [lectures on Youtube](https://www.youtube.com/watch?v=PPLop4L2eGk&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN) )
- [Structuring Machine Learning Projects](https://www.coursera.org/learn/machine-learning-projects)
- [Udacity's deep learning nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101) or  [Coursera's Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) (for deep learning)

If you already know the concepts, the following resources are pretty useful for a quick review of different concepts:
- [StatQuest Machine Learning videos](https://www.youtube.com/watch?v=Gv9_4yMHFhI&list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF)
- [StatQuest Statistics](https://www.youtube.com/watch?v=qBigTkBLU6g&list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9) (for statistics review - most useful for Data Science roles)
- [Machine Learning cheatsheets](https://ml-cheatsheet.readthedocs.io/en/latest/)
- [Chris Albon's ML falshcards](https://machinelearningflashcards.com/)

# 2. ML Fundamentals Topics 

Below are the most important topics to cover:
## 1. Classic ML Concepts
### ML Algorithms' Categories
  - Supervised, unsupervised, and semi-supervised learning (with examples)
    - Classification vs regression vs clustering
  - Parametric vs non-parametric algorithms
  - Linear vs Nonlinear algorithms
### Supervised learning
  - Linear Algorithms
    - Linear regression
      - least squares, residuals,  linear vs multivariate regression
    - Logistic regression
      - cost function (equation, code),  sigmoid function, cross entropy
    - Support Vector Machines
    - Linear discriminant analysis

  - Decision Trees
    - Logits
    - Leaves
    - Training algorithm
      - stop criteria
    - Inference
    - Pruning

  - Ensemble methods
    - Bagging and boosting methods (with examples)
    - Random Forest
    - Boosting
      - Adaboost
      - GBM
      - XGBoost
  - Comparison of different algorithms
    - [TBD: LinkedIn lecture]

  - Optimization
    - Gradient descent (concept, formula, code)
    - Other variations of gradient descent
      - SGD
      - Momentum
      - RMSprop
      - ADAM
  - Loss functions
    - Logistic Loss function 
    - Cross Entropy (remember formula as well)
    - Hinge loss (SVM)

- Feature selection
  - Feature importance
- Model evaluation and selection
  - Evaluation metrics
    - TP, FP, TN, FN
    - Confusion matrix
    - Accuracy, precision, recall/sensitivity, specificity, F-score
      - how do you choose among these? (imbalanced datasets)
      - precision vs TPR (why precision)
    - ROC curve (TPR vs FPR, threshold selection)
    - AUC (model comparison)
    - Extension of the above to multi-class (n-ary) classification
    - algorithm specific metrics [TBD]
  - Model selection
    - Cross validation
      - k-fold cross validation (what's a good k value?)

### Unsupervised learning
  - Clustering
    - Centroid models: k-means clustering
    - Connectivity models: Hierarchical clustering
    - Density models: DBSCAN
  - Gaussian Mixture Models
  - Latent semantic analysis
  - Hidden Markov Models (HMMs)
    - Markov processes
    - Transition probability and emission probability
    - Viterbi algorithm [Advanced]
  - Dimension reduction techniques
    - Principal Component Analysis (PCA)
    - Independent Component Analysis (ICA)
    - T-sne


### Bias / Variance (Underfitting/Overfitting)
- Regularization techniques
  - L1/L2 (Lasso/Ridge)
### Sampling
- sampling techniques
  - Uniform sampling
  - Reservoir sampling
  - Stratified sampling
### Handling  data
 - Missing data 
 - Imbalanced data 
 - Data distribution shifts 

### Computational complexity of ML algorithms
- [TBD]

## 2. Deep learning
- Feedforward NNs
  - In depth knowledge of how they work
  - [EX] activation function for classes that are not mutually exclusive
- RNN
  - backpropagation through time (BPTT)
  - vanishing/exploding gradient problem
- LSTM
  - vanishing/exploding gradient problem
  - gradient?
- Dropout
  - how to apply dropout to LSTM?
- Seq2seq models
- Attention
  - self-attention
- * Transformer architecture (in details, no kidding!)
  - [Illustrated transformer](http://jalammar.github.io/illustrated-transformer/) 
- Embeddings (word embeddings)


## 3. Statistical ML
###  Bayesian algorithms
  - Naive Bayes
  - Maximum a posteriori (MAP) estimation
  - Maximum Likelihood (ML) estimation
### Statistical significance
- R-squared
- P-values

## 4. Other topics:
  - Outliers
  - Similarity/dissimilarity metrics
    - Euclidean, Manhattan, Cosine, Mahalanobis (advanced)

## 5. Foundation Models & Large Language Models (LLMs)

> The biggest shift in 2026 ML interviews: LLMs / foundation models are now expected **breadth**, not a specialty. Below are the core concepts interviewers probe. For end-to-end GenAI **system design** (RAG, agents, serving), see [Chapter 4](./MLSD/ml-system-design.md) and the [Agentic AI Systems repo](https://github.com/alirezadir/Agentic-AI-Systems.git).

### Transformer & LLM internals
- Self-attention recap: scaled dot-product attention, **why scale by √dₖ** (keeps softmax out of saturation → stable gradients), multi-head attention
- Attention variants (memory / throughput tradeoffs): **MHA → MQA (multi-query) → GQA (grouped-query) → MLA (multi-head latent, DeepSeek)**
- Positional encodings: absolute / learned, **RoPE** (rotary), ALiBi; long-context extension (position interpolation, YaRN)
- **KV cache**: caches per-layer keys/values so each new token is O(n) instead of recomputing O(n²); dominates memory at long context × large batch
- **FlashAttention** (IO-aware, tiled attention in SRAM) — why long-context training became practical
- Block internals: pre-norm vs post-norm, **RMSNorm**, **SwiGLU / GeGLU** activations
- **Mixture-of-Experts (MoE)**: sparse expert routing, load balancing, active vs total params (Mixtral, DeepSeek-V3)
- Tokenization: **BPE / byte-level BPE / SentencePiece**, vocab size, context window
- Scaling laws (Chinchilla compute-optimal), emergent abilities

### Training pipeline (pretraining → post-training)
1. **Pretraining** — self-supervised next-token prediction on web-scale corpora
2. **SFT / instruction tuning** — supervised on (prompt → response) demonstrations
3. **Preference alignment (RLHF & alternatives)** — align to human preferences & safety
4. **Reasoning RL** — verifiable-reward RL for chain-of-thought reasoning models (o-series, DeepSeek-R1)

### Post-training: SFT & RL / alignment algorithms (2026)
A central 2026 interview theme is **picking the right post-training method for your data & compute**. Modern stacks *layer* them (SFT → preference optimization → RL) rather than using one monolithic method.

| Algorithm | Data required | Extra models needed | Rel. cost | Use when |
|---|---|---|---|---|
| **SFT** | Curated instruction (prompt→response) demos | none | low | Teach format / instruction-following (always first) |
| **RLHF (PPO)** | Human preference labels | reward model + critic/value + reference | high | Classic; mostly displaced by simpler methods |
| **DPO** | Preference **pairs** (chosen vs rejected) | reference model | medium | Default offline alignment; trains like SFT, no RL loop |
| **SimPO** | Preference pairs | none (reference-free) | med-low | DPO without a reference model (avg-logprob implicit reward) |
| **KTO** | **Binary** thumbs up/down (no pairs) | none | low | Cheap / noisy feedback; only unpaired signal available |
| **ORPO** | Instruction demos only | none | low | Merge SFT + preference tuning into one stage |
| **GRPO** | Prompts only (sample a **group** of responses, group-relative advantage) | none (no critic) | medium | RL without a value net; reasoning / math (DeepSeek) |
| **RLVR** | Tasks with **verifiable** reward (unit tests, math answer, valid JSON) | automated verifier | medium | Code / math / tool-use where correctness is checkable |

Key talking points: DPO collapses the reward model + RL loop into a single supervised loss on pairs; **GRPO drops the critic** (group-normalized rewards → lower memory than PPO); **GRPO / RLVR power reasoning models**; algorithm rankings are **scale-dependent** (online RL can win at ~1.5B while SimPO can win at ~7B). Also know DAPO (stabilizing long chain-of-thought RL) and RLAIF (AI feedback in place of human labels).

### Parameter-efficient fine-tuning (PEFT)
- Full fine-tune vs PEFT tradeoffs (compute, storage, catastrophic forgetting)
- **LoRA / QLoRA** (low-rank adapters; QLoRA fine-tunes on a quantized base), adapters, prefix / prompt tuning
- Key hyperparameters: LoRA rank / α, learning rate, epochs, batch size

### Inference & serving optimization
- **Quantization**: PTQ vs QAT; INT8 / INT4 (GPTQ, AWQ), **FP8**, **KV-cache quantization** (KV can exceed weights at long context)
- **Paged attention / continuous batching** (vLLM), prefix caching
- **Speculative decoding** (draft + verify), distillation, **MoE** for sparse compute
- Parallelism: tensor / pipeline / sequence; prefill vs decode phases

### Decoding & in-context learning
- Decoding: greedy, beam, **temperature, top-k, top-p (nucleus)**, repetition penalty; structured / constrained decoding (JSON / grammar)
- In-context learning: zero / few-shot, **chain-of-thought**, self-consistency; **test-time compute / inference-time scaling** (reasoning models)
- Adding knowledge: **RAG vs long-context vs fine-tuning** (tradeoffs)

### Generative model evaluation (2026: *"eval is the new system design"*)
- Why classic metrics fail for open-ended generation; **hallucination** & calibration
- **RAG triad** (RAGAS): faithfulness, answer relevance, context relevance; retrieval metrics (recall@k, MRR, nDCG)
- **LLM-as-judge**, pairwise win-rate / Arena (Elo), golden sets & regression testing
- Agent metrics: tool-selection quality, task / step success, trajectory adherence
- Benchmarks (MMLU, GPQA, SWE-bench, …); safety / red-teaming, jailbreak robustness

## 6. Multimodal & Generative AI

### Multimodal foundation models (FMs)
- Idea: a shared representation across modalities (text, image, audio, video, action)
- **Fusion approaches**: contrastive dual-encoder (**CLIP / SigLIP**); projection / adapter into LLM token space (**LLaVA**); cross-attention (Flamingo); early vs late fusion; native / "omni" any-to-any (GPT-4o, Gemini)
- Unified understanding **and** generation; image / audio tokenizers (VQ-VAE) for autoregressive generation

### Vision-Language Models (VLMs)
- Architecture: **vision encoder (ViT / SigLIP / DINOv2) → projector → LLM**
- Tasks: VQA, captioning, **OCR / document understanding**, grounding / detection, chart / UI understanding
- Examples: GPT-4o, Gemini, Claude, Qwen-VL, **LLaVA**, **PaliGemma**, InternVL
- Training: image-text pretraining + visual instruction tuning

### Vision-Language-Action models (VLAs)
- VLMs extended to **embodied / robotic control** — perceive → understand instruction → output actions, often in a single forward pass
- Action representation: **discrete action tokens** (RT-2, OpenVLA) vs **continuous via diffusion / flow-matching action heads** (π0)
- Examples:
  - **RT-2** (Google DeepMind) — built on PaLI-X / PaLM-E VLMs; transfers web knowledge + chain-of-thought to robot control
  - **OpenVLA** — 7B, open; DINOv2 + SigLIP vision + Llama-2; trained on 970k real demos; beats RT-2-X (55B) with ~7× fewer params
  - **π0 (Pi-Zero)** — PaliGemma VLM + **flow-matching** action expert; ~50 Hz high-frequency dexterous control
- Use cases: robot manipulation, humanoids, generalist robot policies; data from teleop demos + Open X-Embodiment

### Diffusion vs autoregressive generation
| | **Autoregressive (AR)** | **Diffusion** |
|---|---|---|
| How | predict next token sequentially | iterative denoising from noise |
| Likelihood | exact | variational / score-based |
| Strength | discrete sequences, variable length, reasoning | continuous high-dim (image / video / audio), high fidelity |
| Speed | 1 forward / token (KV-cache helps) | many denoising steps (cut via distillation / consistency / **flow matching**) |
| Examples | text (GPT), image tokens (Parti, VAR), audio (AudioLM) | image (Stable Diffusion, Imagen), **video (Sora, Veo — DiT)**, audio / music (Stable Audio), robot actions (π0) |

- **Diffusion deep-dive**: forward / reverse process (DDPM), score / noise prediction, **latent diffusion** (Stable Diffusion), **DiT** (diffusion transformers, used for video), **classifier-free guidance**, fast samplers (DDIM), **flow matching / rectified flow** (SD3, π0)
- **AR deep-dive**: discretize a modality into tokens → next-token prediction; enables unified token-based multimodal models; image AR (next-scale prediction / VAR), audio (AudioLM / MusicGen)
- Emerging: **text diffusion / masked diffusion LMs**; consistency & few-step models; unified AR + diffusion stacks

# 3. ML Fundamentals Sample Questions 
- What is machine learning and how does it differ from traditional programming?
- What are different types of machine learning techniques?
- What is the difference between supervised and unsupervised learning?
- What is semi-supervised learning?
- What are stages of building machine learning models?
- Can you explain the bias-variance trade-off in machine learning?
- What is overfitting and how do you prevent it?
- Why and how do you split data into train, test, and validation set?
- What is cross-validation and why is it important?
- Can you explain the concept of regularization and its types (L1, L2, etc.)? 
- How Do You Handle Missing or Corrupted Data in a Dataset
- What is a decision tree and how does it work?
- Can you explain logistic regression?
- Can you explain the K-Nearest Neighbors (KNN) algorithm?
- Compare K-means and KNN algorithms.
- Explain decision-tree based algorithms (random forest, GBDT)
- What is gradient descent and how does it work?
- Can you explain the support vector machine (SVM) algorithm? what is Kernel SVM?
- Can you explain neural networks and how they work?
- What is deep learning and how does it differ from traditional machine learning?
- Can you explain the backpropagation algorithm and its role in training neural networks?
- What is a convolutional neural network (CNN) and how does it work?
- What is transfer learning and how is it used in practice?
* [45 ML interview questions](https://www.simplilearn.com/tutorials/machine-learning-tutorial/machine-learning-interview-questions)

### LLM / GenAI / Multimodal sample questions (2026)
- Walk through a transformer block. Why divide attention scores by √dₖ?
- What is the KV cache and why does it matter for inference? How does its memory scale?
- Compare MHA, MQA, and GQA. Why did Llama adopt GQA?
- What is RoPE and why are rotary embeddings preferred over learned positional embeddings?
- When would you choose RAG vs fine-tuning vs long-context? 
- Compare SFT, DPO, GRPO, and RLVR — data requirements, what extra models each needs, and when to use each.
- Why does GRPO drop the critic network, and how does it estimate advantage?
- What is LoRA / QLoRA and when would you use PEFT over full fine-tuning?
- Explain quantization (INT8/INT4, FP8) and KV-cache quantization. What's the accuracy/latency tradeoff?
- How does speculative decoding speed up generation?
- How do you evaluate an LLM / RAG system? What is the RAG triad and LLM-as-judge?
- How does a VLM combine a vision encoder with an LLM? (e.g., LLaVA / PaliGemma)
- What is a VLA model? Contrast discrete action tokens (RT-2, OpenVLA) vs flow-matching action heads (π0).
- Diffusion vs autoregressive generation — how do they differ and when is each used (image/video/audio/text)?
- What is classifier-free guidance? What is flow matching / rectified flow?
- What is a Mixture-of-Experts model? Distinguish active vs total parameters.
