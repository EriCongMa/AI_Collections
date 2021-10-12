# Daily arXiv: Machine Translation - October, 2021

# Index


- [2021-10-12](#2021-10-12)

  - [1. CLIP-Adapter: Better Vision-Language Models with Feature Adapters](#2021-10-12-1)
  - [2. The Inductive Bias of In-Context Learning: Rethinking Pretraining Example Design](#2021-10-12-2)
  - [3. Yuan 1.0: Large-Scale Pre-trained Language Model in Zero-Shot and Few-Shot Learning](#2021-10-12-3)
  - [4. WeTS: A Benchmark for Translation Suggestion](#2021-10-12-4)
  - [5. It is Not as Good as You Think! Evaluating Simultaneous Machine Translation on Interpretation Data](#2021-10-12-5)
  - [6. Unsupervised Neural Machine Translation with Generative Language Models Only](#2021-10-12-6)
- [2021-10-11](#2021-10-11)
  - [1. Speeding up Deep Model Training by Sharing Weights and Then Unsharing](#2021-10-11-1)
  - [2. QTN-VQC: An End-to-End Learning framework for Quantum Neural Networks](#2021-10-11-2)
  - [3. M6-10T: A Sharing-Delinking Paradigm for Efficient Multi-Trillion Parameter Pretraining](#2021-10-11-3)
  - [4. Iterative Decoding for Compositional Generalization in Transformers](#2021-10-11-4)
  - [5. Machine Translation Verbosity Control for Automatic Dubbing](#2021-10-11-5)
  - [6. Text analysis and deep learning: A network approach](#2021-10-11-6)
  - [7. Contrastive String Representation Learning using Synthetic Data](#2021-10-11-7)
  - [8. Local and Global Context-Based Pairwise Models for Sentence Ordering](#2021-10-11-8)
- [2021-10-08](#2021-10-08)

  - [1. Unsupervised Multimodal Language Representations using Convolutional Autoencoders](#2021-10-08-1)
  - [2. The Low-Resource Double Bind: An Empirical Study of Pruning for Low-Resource Machine Translation](#2021-10-08-2)
  - [3. On Neurons Invariant to Sentence Structural Changes in Neural Machine Translation](#2021-10-08-3)
  - [4. Towards Continual Knowledge Learning of Language Models](#2021-10-08-4)
- [2021-10-07](#2021-10-07)

  - [1. Sequential Reptile: Inter-Task Gradient Alignment for Multilingual Learning](#2021-10-07-1)
  - [2. How BPE Affects Memorization in Transformers](#2021-10-07-2)
  - [3. Sequence-to-Sequence Lexical Normalization with Multilingual Transformers](#2021-10-07-3)
  - [4. Using Optimal Transport as Alignment Objective for fine-tuning Multilingual Contextualized Embeddings](#2021-10-07-4)
- [2021-10-06](#2021-10-06)
  - [1. OPAD: An Optimized Policy-based Active Learning Framework for Document Content Analysis](#2021-10-06-1)
  - [2. Rerunning OCR -- A Machine Learning Approach to Quality Assessment and Enhancement Prediction](#2021-10-06-2)
  - [3. On the Complementarity between Pre-Training and Back-Translation for Neural Machine Translation](#2021-10-06-3)
  - [4. Data Augmentation Approaches in Natural Language Processing: A Survey](#2021-10-06-4)
  - [5. Sicilian Translator: A Recipe for Low-Resource NMT](#2021-10-06-5)
  - [6. Transfer Learning for Multi-lingual Tasks -- a Survey](#2021-10-06-6)
  - [7. Structured Prediction in NLP -- A survey](#2021-10-06-7)
  - [8. Interactively Generating Explanations for Transformer-based Language Models](#2021-10-06-8)
- [2021-10-05](#2021-10-05)

  - [1. Improving Zero-shot Multilingual Neural Machine Translation for Low-Resource Languages](#2021-10-05-1)
- [2021-10-04](#2021-10-04)

  - [1. Improving Punctuation Restoration for Speech Transcripts via External Data](#2021-10-04-1)
  - [2. A Survey of Knowledge Enhanced Pre-trained Models](#2021-10-04-2)
  - [3. Attention based Sequence to Sequence Learning for Machine Translation of Low Resourced Indic Languages -- A case of Sanskrit to Hindi](#2021-10-04-3)
- [2021-10-01](#2021-10-01)
  - [1. Phonetic Word Embeddings](#2021-10-01-1)
  - [2. Improved statistical machine translation using monolingual paraphrases](#2021-10-01-2)
- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MT-Daily_arXiv-index.md)



# 2021-10-12

[Return to Index](#Index)



<h2 id="2021-10-12-1">1. CLIP-Adapter: Better Vision-Language Models with Feature Adapters
</h2>

Title: [CLIP-Adapter: Better Vision-Language Models with Feature Adapters](https://arxiv.org/abs/2110.04544)

Authors: [Peng Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+P), [Shijie Geng](https://arxiv.org/search/cs?searchtype=author&query=Geng%2C+S), [Renrui Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+R), [Teli Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+T), [Rongyao Fang](https://arxiv.org/search/cs?searchtype=author&query=Fang%2C+R), [Yongfeng Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y), [Hongsheng Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+H), [Yu Qiao](https://arxiv.org/search/cs?searchtype=author&query=Qiao%2C+Y)

> Large-scale contrastive vision-language pre-training has shown significant progress in visual representation learning. Unlike traditional visual systems trained by a fixed set of discrete labels, a new paradigm was introduced in \cite{radford2021learning} to directly learn to align images with raw texts in an open-vocabulary setting. On downstream tasks, a carefully chosen text prompt is employed to make zero-shot predictions.~To avoid non-trivial prompt engineering, context optimization \cite{zhou2021coop} has been proposed to learn continuous vectors as task-specific prompts with few-shot training examples.~In this paper, we show that there is an alternative path to achieve better vision-language models other than prompt tuning.~While prompt tuning is for the textual inputs, we propose CLIP-Adapter to conduct fine-tuning with feature adapters on either visual or language branch. Specifically, CLIP-Adapter adopts an additional bottleneck layer to learn new features and performs residual-style feature blending with the original pre-trained features.~As a consequence, CLIP-Adapter is able to outperform context optimization while maintains a simple design. Experiments and extensive ablation studies on various visual classification tasks demonstrate the effectiveness of our approach.

| Comments: | Technical Report                                             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2110.04544](https://arxiv.org/abs/2110.04544) [cs.CV]** |
|           | (or **[arXiv:2110.04544v1](https://arxiv.org/abs/2110.04544v1) [cs.CV]** for this version) |





<h2 id="2021-10-12-2">2. The Inductive Bias of In-Context Learning: Rethinking Pretraining Example Design
</h2>

Title: [The Inductive Bias of In-Context Learning: Rethinking Pretraining Example Design](https://arxiv.org/abs/2110.04541)

Authors: [Yoav Levine](https://arxiv.org/search/cs?searchtype=author&query=Levine%2C+Y), [Noam Wies](https://arxiv.org/search/cs?searchtype=author&query=Wies%2C+N), [Daniel Jannai](https://arxiv.org/search/cs?searchtype=author&query=Jannai%2C+D), [Dan Navon](https://arxiv.org/search/cs?searchtype=author&query=Navon%2C+D), [Yedid Hoshen](https://arxiv.org/search/cs?searchtype=author&query=Hoshen%2C+Y), [Amnon Shashua](https://arxiv.org/search/cs?searchtype=author&query=Shashua%2C+A)

> Pretraining Neural Language Models (NLMs) over a large corpus involves chunking the text into training examples, which are contiguous text segments of sizes processable by the neural architecture. We highlight a bias introduced by this common practice: we prove that the pretrained NLM can model much stronger dependencies between text segments that appeared in the same training example, than it can between text segments that appeared in different training examples. This intuitive result has a twofold role. First, it formalizes the motivation behind a broad line of recent successful NLM training heuristics, proposed for the pretraining and fine-tuning stages, which do not necessarily appear related at first glance. Second, our result clearly indicates further improvements to be made in NLM pretraining for the benefit of Natural Language Understanding tasks. As an example, we propose "kNN-Pretraining": we show that including semantically related non-neighboring sentences in the same pretraining example yields improved sentence representations and open domain question answering abilities. This theoretically motivated degree of freedom for "pretraining example design" indicates new training schemes for self-improving representations.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2110.04541](https://arxiv.org/abs/2110.04541) [cs.CL]** |
|           | (or **[arXiv:2110.04541v1](https://arxiv.org/abs/2110.04541v1) [cs.CL]** for this version) |





<h2 id="2021-10-12-3">3. Yuan 1.0: Large-Scale Pre-trained Language Model in Zero-Shot and Few-Shot Learning
</h2>

Title: [Yuan 1.0: Large-Scale Pre-trained Language Model in Zero-Shot and Few-Shot Learning](https://arxiv.org/abs/2110.04725)

Authors: [Shaohua Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+S), [Xudong Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+X), [Tong Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+T), [Rongguo Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+R), [Chong Shen](https://arxiv.org/search/cs?searchtype=author&query=Shen%2C+C), [Hongli Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+H), [Feng Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+F), [Hong Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+H), [Jiangang Luo](https://arxiv.org/search/cs?searchtype=author&query=Luo%2C+J), [Liang Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+L), [Xuanwei Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+X), [Jun Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+J)

> Recent work like GPT-3 has demonstrated excellent performance of Zero-Shot and Few-Shot learning on many natural language processing (NLP) tasks by scaling up model size, dataset size and the amount of computation. However, training a model like GPT-3 requires huge amount of computational resources which makes it challengeable to researchers. In this work, we propose a method that incorporates large-scale distributed training performance into model architecture design. With this method, Yuan 1.0, the current largest singleton language model with 245B parameters, achieves excellent performance on thousands GPUs during training, and the state-of-the-art results on NLP tasks. A data processing method is designed to efficiently filter massive amount of raw data. The current largest high-quality Chinese corpus with 5TB high quality texts is built based on this method. In addition, a calibration and label expansion method is proposed to improve the Zero-Shot and Few-Shot performance, and steady improvement is observed on the accuracy of various tasks. Yuan 1.0 presents strong capacity of natural language generation, and the generated articles are difficult to distinguish from the human-written ones.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2110.04725](https://arxiv.org/abs/2110.04725) [cs.CL]** |
|           | (or **[arXiv:2110.04725v1](https://arxiv.org/abs/2110.04725v1) [cs.CL]** for this version) |





<h2 id="2021-10-12-4">4. WeTS: A Benchmark for Translation Suggestion
</h2>

Title: [WeTS: A Benchmark for Translation Suggestion](https://arxiv.org/abs/2110.05151)

Authors: [Zhen Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Z), [Yingxue Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y), [Ernan Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+E), [Fandong Meng](https://arxiv.org/search/cs?searchtype=author&query=Meng%2C+F), [Jie Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+J)

> Translation Suggestion (TS), which provides alternatives for specific words or phrases given the entire documents translated by machine translation (MT) \cite{lee2021intellicat}, has been proven to play a significant role in post editing (PE). However, there is still no publicly available data set to support in-depth research for this problem, and no reproducible experimental results can be followed by researchers in this community. To break this limitation, we create a benchmark data set for TS, called \emph{WeTS}, which contains golden corpus annotated by expert translators on four translation directions. Apart from the human-annotated golden corpus, we also propose several novel methods to generate synthetic corpus which can substantially improve the performance of TS. With the corpus we construct, we introduce the Transformer-based model for TS, and experimental results show that our model achieves State-Of-The-Art (SOTA) results on all four translation directions, including English-to-German, German-to-English, Chinese-to-English and English-to-Chinese. Codes and corpus can be found at \url{[this https URL](https://github.com/ZhenYangIACAS/WeTS.git)}.

| Comments: | Translation suggestion, Transformer                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2110.05151](https://arxiv.org/abs/2110.05151) [cs.CL]** |
|           | (or **[arXiv:2110.05151v1](https://arxiv.org/abs/2110.05151v1) [cs.CL]** for this version) |





<h2 id="2021-10-12-5">5. It is Not as Good as You Think! Evaluating Simultaneous Machine Translation on Interpretation Data
</h2>

Title: [It is Not as Good as You Think! Evaluating Simultaneous Machine Translation on Interpretation Data](https://arxiv.org/abs/2110.05213)

Authors: [Jinming Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+J), [Philip Arthur](https://arxiv.org/search/cs?searchtype=author&query=Arthur%2C+P), [Gholamreza Haffari](https://arxiv.org/search/cs?searchtype=author&query=Haffari%2C+G), [Trevor Cohn](https://arxiv.org/search/cs?searchtype=author&query=Cohn%2C+T), [Ehsan Shareghi](https://arxiv.org/search/cs?searchtype=author&query=Shareghi%2C+E)

> Most existing simultaneous machine translation (SiMT) systems are trained and evaluated on offline translation corpora. We argue that SiMT systems should be trained and tested on real interpretation data. To illustrate this argument, we propose an interpretation test set and conduct a realistic evaluation of SiMT trained on offline translations. Our results, on our test set along with 3 existing smaller scale language pairs, highlight the difference of up-to 13.83 BLEU score when SiMT models are evaluated on translation vs interpretation data. In the absence of interpretation training data, we propose a translation-to-interpretation (T2I) style transfer method which allows converting existing offline translations into interpretation-style data, leading to up-to 2.8 BLEU improvement. However, the evaluation gap remains notable, calling for constructing large-scale interpretation corpora better suited for evaluating and developing SiMT systems.

| Comments: | EMNLP2021                                                    |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2110.05213](https://arxiv.org/abs/2110.05213) [cs.CL]** |
|           | (or **[arXiv:2110.05213v1](https://arxiv.org/abs/2110.05213v1) [cs.CL]** for this version) |





<h2 id="2021-10-12-6">6. Unsupervised Neural Machine Translation with Generative Language Models Only
</h2>

Title: [Unsupervised Neural Machine Translation with Generative Language Models Only](https://arxiv.org/abs/2110.05448)

Authors: [Jesse Michael Han](https://arxiv.org/search/cs?searchtype=author&query=Han%2C+J+M), [Igor Babuschkin](https://arxiv.org/search/cs?searchtype=author&query=Babuschkin%2C+I), [Harrison Edwards](https://arxiv.org/search/cs?searchtype=author&query=Edwards%2C+H), [Arvind Neelakantan](https://arxiv.org/search/cs?searchtype=author&query=Neelakantan%2C+A), [Tao Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+T), [Stanislas Polu](https://arxiv.org/search/cs?searchtype=author&query=Polu%2C+S), [Alex Ray](https://arxiv.org/search/cs?searchtype=author&query=Ray%2C+A), [Pranav Shyam](https://arxiv.org/search/cs?searchtype=author&query=Shyam%2C+P), [Aditya Ramesh](https://arxiv.org/search/cs?searchtype=author&query=Ramesh%2C+A), [Alec Radford](https://arxiv.org/search/cs?searchtype=author&query=Radford%2C+A), [Ilya Sutskever](https://arxiv.org/search/cs?searchtype=author&query=Sutskever%2C+I)

> We show how to derive state-of-the-art unsupervised neural machine translation systems from generatively pre-trained language models. Our method consists of three steps: few-shot amplification, distillation, and backtranslation. We first use the zero-shot translation ability of large pre-trained language models to generate translations for a small set of unlabeled sentences. We then amplify these zero-shot translations by using them as few-shot demonstrations for sampling a larger synthetic dataset. This dataset is distilled by discarding the few-shot demonstrations and then fine-tuning. During backtranslation, we repeatedly generate translations for a set of inputs and then fine-tune a single language model on both directions of the translation task at once, ensuring cycle-consistency by swapping the roles of gold monotext and generated translations when fine-tuning. By using our method to leverage GPT-3's zero-shot translation capability, we achieve a new state-of-the-art in unsupervised translation on the WMT14 English-French benchmark, attaining a BLEU score of 42.1.

| Comments: | 10 pages                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2110.05448](https://arxiv.org/abs/2110.05448) [cs.CL]** |
|           | (or **[arXiv:2110.05448v1](https://arxiv.org/abs/2110.05448v1) [cs.CL]** for this version) |






# 2021-10-11

[Return to Index](#Index)



<h2 id="2021-10-11-1">1. Speeding up Deep Model Training by Sharing Weights and Then Unsharing
</h2>

Title: [Speeding up Deep Model Training by Sharing Weights and Then Unsharing](https://arxiv.org/abs/2110.03848)

Authors: [Shuo Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+S), [Le Hou](https://arxiv.org/search/cs?searchtype=author&query=Hou%2C+L), [Xiaodan Song](https://arxiv.org/search/cs?searchtype=author&query=Song%2C+X), [Qiang Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Q), [Denny Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+D)

> We propose a simple and efficient approach for training the BERT model. Our approach exploits the special structure of BERT that contains a stack of repeated modules (i.e., transformer encoders). Our proposed approach first trains BERT with the weights shared across all the repeated modules till some point. This is for learning the commonly shared component of weights across all repeated layers. We then stop weight sharing and continue training until convergence. We present theoretic insights for training by sharing weights then unsharing with analysis for simplified models. Empirical experiments on the BERT model show that our method yields better performance of trained models, and significantly reduces the number of training iterations.

| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2110.03848](https://arxiv.org/abs/2110.03848) [cs.LG]** |
|           | (or **[arXiv:2110.03848v1](https://arxiv.org/abs/2110.03848v1) [cs.LG]** for this version) |





<h2 id="2021-10-11-2">2. QTN-VQC: An End-to-End Learning framework for Quantum Neural Networks
</h2>

Title: [QTN-VQC: An End-to-End Learning framework for Quantum Neural Networks](https://arxiv.org/abs/2110.03861)

Authors: [Jun Qi](https://arxiv.org/search/quant-ph?searchtype=author&query=Qi%2C+J), [Chao-Han Huck Yang](https://arxiv.org/search/quant-ph?searchtype=author&query=Yang%2C+C+H), [Pin-Yu Chen](https://arxiv.org/search/quant-ph?searchtype=author&query=Chen%2C+P)

> The advent of noisy intermediate-scale quantum (NISQ) computers raises a crucial challenge to design quantum neural networks for fully quantum learning tasks. To bridge the gap, this work proposes an end-to-end learning framework named QTN-VQC, by introducing a trainable quantum tensor network (QTN) for quantum embedding on a variational quantum circuit (VQC). The architecture of QTN is composed of a parametric tensor-train network for feature extraction and a tensor product encoding for quantum encoding. We highlight the QTN for quantum embedding in terms of two perspectives: (1) we theoretically characterize QTN by analyzing its representation power of input features; (2) QTN enables an end-to-end parametric model pipeline, namely QTN-VQC, from the generation of quantum embedding to the output measurement. Our experiments on the MNIST dataset demonstrate the advantages of QTN for quantum embedding over other quantum embedding approaches.

| Subjects: | **Quantum Physics (quant-ph)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG); Neural and Evolutionary Computing (cs.NE) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2110.03861](https://arxiv.org/abs/2110.03861) [quant-ph]** |
|           | (or **[arXiv:2110.03861v1](https://arxiv.org/abs/2110.03861v1) [quant-ph]** for this version) |





<h2 id="2021-10-11-3">3. M6-10T: A Sharing-Delinking Paradigm for Efficient Multi-Trillion Parameter Pretraining
</h2>

Title: [M6-10T: A Sharing-Delinking Paradigm for Efficient Multi-Trillion Parameter Pretraining](https://arxiv.org/abs/2110.03888)

Authors: [Junyang Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+J), [An Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+A), [Jinze Bai](https://arxiv.org/search/cs?searchtype=author&query=Bai%2C+J), [Chang Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+C), [Le Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+L), [Xianyan Jia](https://arxiv.org/search/cs?searchtype=author&query=Jia%2C+X), [Ang Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+A), [Jie Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+J), [Yong Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Y), [Wei Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+W), [Jingren Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+J), [Hongxia Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+H)

> Recent expeditious developments in deep learning algorithms, distributed training, and even hardware design for large models have enabled training extreme-scale models, say GPT-3 and Switch Transformer possessing hundreds of billions or even trillions of parameters. However, under limited resources, extreme-scale model training that requires enormous amounts of computes and memory footprint suffers from frustratingly low efficiency in model convergence. In this paper, we propose a simple training strategy called "Pseudo-to-Real" for high-memory-footprint-required large models. Pseudo-to-Real is compatible with large models with architecture of sequential layers. We demonstrate a practice of pretraining unprecedented 10-trillion-parameter model, an order of magnitude larger than the state-of-the-art, on solely 512 GPUs within 10 days. Besides demonstrating the application of Pseudo-to-Real, we also provide a technique, Granular CPU offloading, to manage CPU memory for training large model and maintain high GPU utilities. Fast training of extreme-scale models on a decent amount of resources can bring much smaller carbon footprint and contribute to greener AI.

| Comments: | 14 pages, 4 figures                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2110.03888](https://arxiv.org/abs/2110.03888) [cs.LG]** |
|           | (or **[arXiv:2110.03888v1](https://arxiv.org/abs/2110.03888v1) [cs.LG]** for this version) |





<h2 id="2021-10-11-4">4. Iterative Decoding for Compositional Generalization in Transformers
</h2>

Title: [Iterative Decoding for Compositional Generalization in Transformers](https://arxiv.org/abs/2110.04169)

Authors: [Luana Ruiz](https://arxiv.org/search/cs?searchtype=author&query=Ruiz%2C+L), [Joshua Ainslie](https://arxiv.org/search/cs?searchtype=author&query=Ainslie%2C+J), [Santiago Ontañón](https://arxiv.org/search/cs?searchtype=author&query=Ontañón%2C+S)

> Deep learning models do well at generalizing to in-distribution data but struggle to generalize compositionally, i.e., to combine a set of learned primitives to solve more complex tasks. In particular, in sequence-to-sequence (seq2seq) learning, transformers are often unable to predict correct outputs for even marginally longer examples than those seen during training. This paper introduces iterative decoding, an alternative to seq2seq learning that (i) improves transformer compositional generalization and (ii) evidences that, in general, seq2seq transformers do not learn iterations that are not unrolled. Inspired by the idea of compositionality -- that complex tasks can be solved by composing basic primitives -- training examples are broken down into a sequence of intermediate steps that the transformer then learns iteratively. At inference time, the intermediate outputs are fed back to the transformer as intermediate inputs until an end-of-iteration token is predicted. Through numerical experiments, we show that transfomers trained via iterative decoding outperform their seq2seq counterparts on the PCFG dataset, and solve the problem of calculating Cartesian products between vectors longer than those seen during training with 100% accuracy, a task at which seq2seq models have been shown to fail. We also illustrate a limitation of iterative decoding, specifically, that it can make sorting harder to learn on the CFQ dataset.

| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2110.04169](https://arxiv.org/abs/2110.04169) [cs.LG]** |
|           | (or **[arXiv:2110.04169v1](https://arxiv.org/abs/2110.04169v1) [cs.LG]** for this version) |





<h2 id="2021-10-11-5">5. Machine Translation Verbosity Control for Automatic Dubbing
</h2>

Title: [Machine Translation Verbosity Control for Automatic Dubbing](https://arxiv.org/abs/2110.03847)

Authors: [Surafel M. Lakew](https://arxiv.org/search/cs?searchtype=author&query=Lakew%2C+S+M), [Marcello Federico](https://arxiv.org/search/cs?searchtype=author&query=Federico%2C+M), [Yue Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Y), [Cuong Hoang](https://arxiv.org/search/cs?searchtype=author&query=Hoang%2C+C), [Yogesh Virkar](https://arxiv.org/search/cs?searchtype=author&query=Virkar%2C+Y), [Roberto Barra-Chicote](https://arxiv.org/search/cs?searchtype=author&query=Barra-Chicote%2C+R), [Robert Enyedi](https://arxiv.org/search/cs?searchtype=author&query=Enyedi%2C+R)

> Automatic dubbing aims at seamlessly replacing the speech in a video document with synthetic speech in a different language. The task implies many challenges, one of which is generating translations that not only convey the original content, but also match the duration of the corresponding utterances. In this paper, we focus on the problem of controlling the verbosity of machine translation output, so that subsequent steps of our automatic dubbing pipeline can generate dubs of better quality. We propose new methods to control the verbosity of MT output and compare them against the state of the art with both intrinsic and extrinsic evaluations. For our experiments we use a public data set to dub English speeches into French, Italian, German and Spanish. Finally, we report extensive subjective tests that measure the impact of MT verbosity control on the final quality of dubbed video clips.

| Comments: | Accepted at IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) 2021 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2110.03847](https://arxiv.org/abs/2110.03847) [cs.CL]** |
|           | (or **[arXiv:2110.03847v1](https://arxiv.org/abs/2110.03847v1) [cs.CL]** for this version) |





<h2 id="2021-10-11-6">6. Text analysis and deep learning: A network approach
</h2>

Title: [Text analysis and deep learning: A network approach](https://arxiv.org/abs/2110.04151)

Authors: [Ingo Marquart](https://arxiv.org/search/cs?searchtype=author&query=Marquart%2C+I)

> Much information available to applied researchers is contained within written language or spoken text. Deep language models such as BERT have achieved unprecedented success in many applications of computational linguistics. However, much less is known about how these models can be used to analyze existing text. We propose a novel method that combines transformer models with network analysis to form a self-referential representation of language use within a corpus of interest. Our approach produces linguistic relations strongly consistent with the underlying model as well as mathematically well-defined operations on them, while reducing the amount of discretionary choices of representation and distance measures. It represents, to the best of our knowledge, the first unsupervised method to extract semantic networks directly from deep language models. We illustrate our approach in a semantic analysis of the term "founder". Using the entire corpus of Harvard Business Review from 1980 to 2020, we find that ties in our network track the semantics of discourse over time, and across contexts, identifying and relating clusters of semantic and syntactic relations. Finally, we discuss how this method can also complement and inform analyses of the behavior of deep learning models.

| Subjects:    | **Computation and Language (cs.CL)**; Social and Information Networks (cs.SI) |
| ------------ | ------------------------------------------------------------ |
| ACM classes: | I.2.7; I.5.4; J.4                                            |
| Cite as:     | **[arXiv:2110.04151](https://arxiv.org/abs/2110.04151) [cs.CL]** |
|              | (or **[arXiv:2110.04151v1](https://arxiv.org/abs/2110.04151v1) [cs.CL]** for this version) |





<h2 id="2021-10-11-7">7. Contrastive String Representation Learning using Synthetic Data
</h2>

Title: [Contrastive String Representation Learning using Synthetic Data](https://arxiv.org/abs/2110.04217)

Authors: [Urchade Zaratiana](https://arxiv.org/search/cs?searchtype=author&query=Zaratiana%2C+U)

> String representation Learning (SRL) is an important task in the field of Natural Language Processing, but it remains under-explored. The goal of SRL is to learn dense and low-dimensional vectors (or embeddings) for encoding character sequences. The learned representation from this task can be used in many downstream application tasks such as string similarity matching or lexical normalization. In this paper, we propose a new method for to train a SRL model by only using synthetic data. Our approach makes use of Contrastive Learning in order to maximize similarity between related strings while minimizing it for unrelated strings. We demonstrate the effectiveness of our approach by evaluating the learned representation on the task of string similarity matching. Codes, data and pretrained models will be made publicly available.

| Subjects: | **Computation and Language (cs.CL)**; Information Retrieval (cs.IR) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2110.04217](https://arxiv.org/abs/2110.04217) [cs.CL]** |
|           | (or **[arXiv:2110.04217v1](https://arxiv.org/abs/2110.04217v1) [cs.CL]** for this version) |





<h2 id="2021-10-11-8">8. Local and Global Context-Based Pairwise Models for Sentence Ordering
</h2>

Title: [Local and Global Context-Based Pairwise Models for Sentence Ordering](https://arxiv.org/abs/2110.04291)

Authors: [Ruskin Raj Manku](https://arxiv.org/search/cs?searchtype=author&query=Manku%2C+R+R), [Aditya Jyoti Paul](https://arxiv.org/search/cs?searchtype=author&query=Paul%2C+A+J)

> Sentence Ordering refers to the task of rearranging a set of sentences into the appropriate coherent order. For this task, most previous approaches have explored global context-based end-to-end methods using Sequence Generation techniques. In this paper, we put forward a set of robust local and global context-based pairwise ordering strategies, leveraging which our prediction strategies outperform all previous works in this domain. Our proposed encoding method utilizes the paragraph's rich global contextual information to predict the pairwise order using novel transformer architectures. Analysis of the two proposed decoding strategies helps better explain error propagation in pairwise models. This approach is the most accurate pure pairwise model and our encoding strategy also significantly improves the performance of other recent approaches that use pairwise models, including the previous state-of-the-art, demonstrating the research novelty and generalizability of this work. Additionally, we show how the pre-training task for ALBERT helps it to significantly outperform BERT, despite having considerably lesser parameters. The extensive experimental results, architectural analysis and ablation studies demonstrate the effectiveness and superiority of the proposed models compared to the previous state-of-the-art, besides providing a much better understanding of the functioning of pairwise models.

| Comments: | Under review by Knowledge-Based Systems                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Information Retrieval (cs.IR); Machine Learning (cs.LG); Logic in Computer Science (cs.LO) |
| Cite as:  | **[arXiv:2110.04291](https://arxiv.org/abs/2110.04291) [cs.CL]** |
|           | (or **[arXiv:2110.04291v1](https://arxiv.org/abs/2110.04291v1) [cs.CL]** for this version) |





# 2021-10-08

[Return to Index](#Index)



<h2 id="2021-10-08-1">1. Unsupervised Multimodal Language Representations using Convolutional Autoencoders
</h2>

Title: [Unsupervised Multimodal Language Representations using Convolutional Autoencoders](https://arxiv.org/abs/2110.03007)

Authors: [Panagiotis Koromilas](https://arxiv.org/search/cs?searchtype=author&query=Koromilas%2C+P), [Theodoros Giannakopoulos](https://arxiv.org/search/cs?searchtype=author&query=Giannakopoulos%2C+T)

> Multimodal Language Analysis is a demanding area of research, since it is associated with two requirements: combining different modalities and capturing temporal information. During the last years, several works have been proposed in the area, mostly centered around supervised learning in downstream tasks. In this paper we propose extracting unsupervised Multimodal Language representations that are universal and can be applied to different tasks. Towards this end, we map the word-level aligned multimodal sequences to 2-D matrices and then use Convolutional Autoencoders to learn embeddings by combining multiple datasets. Extensive experimentation on Sentiment Analysis (MOSEI) and Emotion Recognition (IEMOCAP) indicate that the learned representations can achieve near-state-of-the-art performance with just the use of a Logistic Regression algorithm for downstream classification. It is also shown that our method is extremely lightweight and can be easily generalized to other tasks and unseen data with small performance drop and almost the same number of parameters. The proposed multimodal representation models are open-sourced and will help grow the applicability of Multimodal Language.

| Comments: | 5 pages                                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG); Multimedia (cs.MM) |
| Cite as:  | **[arXiv:2110.03007](https://arxiv.org/abs/2110.03007) [cs.CL]** |
|           | (or **[arXiv:2110.03007v1](https://arxiv.org/abs/2110.03007v1) [cs.CL]** for this version) |







<h2 id="2021-10-08-2">2. The Low-Resource Double Bind: An Empirical Study of Pruning for Low-Resource Machine Translation
</h2>

Title: [The Low-Resource Double Bind: An Empirical Study of Pruning for Low-Resource Machine Translation](https://arxiv.org/abs/2110.03036)

Authors: [Orevaoghene Ahia](https://arxiv.org/search/cs?searchtype=author&query=Ahia%2C+O), [Julia Kreutzer](https://arxiv.org/search/cs?searchtype=author&query=Kreutzer%2C+J), [Sara Hooker](https://arxiv.org/search/cs?searchtype=author&query=Hooker%2C+S)

> A "bigger is better" explosion in the number of parameters in deep neural networks has made it increasingly challenging to make state-of-the-art networks accessible in compute-restricted environments. Compression techniques have taken on renewed importance as a way to bridge the gap. However, evaluation of the trade-offs incurred by popular compression techniques has been centered on high-resource datasets. In this work, we instead consider the impact of compression in a data-limited regime. We introduce the term low-resource double bind to refer to the co-occurrence of data limitations and compute resource constraints. This is a common setting for NLP for low-resource languages, yet the trade-offs in performance are poorly studied. Our work offers surprising insights into the relationship between capacity and generalization in data-limited regimes for the task of machine translation. Our experiments on magnitude pruning for translations from English into Yoruba, Hausa, Igbo and German show that in low-resource regimes, sparsity preserves performance on frequent sentences but has a disparate impact on infrequent ones. However, it improves robustness to out-of-distribution shifts, especially for datasets that are very distinct from the training distribution. Our findings suggest that sparsity can play a beneficial role at curbing memorization of low frequency attributes, and therefore offers a promising solution to the low-resource double bind.

| Comments: | Accepted to Findings of EMNLP 2021                           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2110.03036](https://arxiv.org/abs/2110.03036) [cs.CL]** |
|           | (or **[arXiv:2110.03036v1](https://arxiv.org/abs/2110.03036v1) [cs.CL]** for this version) |







<h2 id="2021-10-08-3">3. On Neurons Invariant to Sentence Structural Changes in Neural Machine Translation
</h2>

Title: [On Neurons Invariant to Sentence Structural Changes in Neural Machine Translation](https://arxiv.org/abs/2110.03067)

Authors: [Gal Patel](https://arxiv.org/search/cs?searchtype=author&query=Patel%2C+G), [Leshem Choshen](https://arxiv.org/search/cs?searchtype=author&query=Choshen%2C+L), [Omri Abend](https://arxiv.org/search/cs?searchtype=author&query=Abend%2C+O)

> To gain insight into the role neurons play, we study the activation patterns corresponding to meaning-preserving paraphrases (e.g., active-passive). We compile a dataset of controlled syntactic paraphrases in English with their reference German translations and demonstrate our model-agnostic approach with the Transformer translation model. First, we identify neurons that correlate across paraphrases and dissect the observed correlation into possible confounds. Although lower-level components are found as the cause of similar activations, no sentence-level semantics or syntax are detected locally. Later, we manipulate neuron activations to influence translation towards a particular syntactic form. We find that a simple value shift is effective, and more so when many neurons are modified. These suggest that complex syntactic constructions are indeed encoded in the model. We conclude by discussing how to better manipulate it using the correlations we first obtained.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2110.03067](https://arxiv.org/abs/2110.03067) [cs.CL]** |
|           | (or **[arXiv:2110.03067v1](https://arxiv.org/abs/2110.03067v1) [cs.CL]** for this version) |







<h2 id="2021-10-08-4">4. Towards Continual Knowledge Learning of Language Models
</h2>

Title: [Towards Continual Knowledge Learning of Language Models](https://arxiv.org/abs/2110.03215)

Authors: [Joel Jang](https://arxiv.org/search/cs?searchtype=author&query=Jang%2C+J), [Seonghyeon Ye](https://arxiv.org/search/cs?searchtype=author&query=Ye%2C+S), [Sohee Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+S), [Joongbo Shin](https://arxiv.org/search/cs?searchtype=author&query=Shin%2C+J), [Janghoon Han](https://arxiv.org/search/cs?searchtype=author&query=Han%2C+J), [Gyeonghun Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+G), [Stanley Jungkyu Choi](https://arxiv.org/search/cs?searchtype=author&query=Choi%2C+S+J), [Minjoon Seo](https://arxiv.org/search/cs?searchtype=author&query=Seo%2C+M)

> Large Language Models (LMs) are known to encode world knowledge in their parameters as they pretrain on a vast amount of web corpus, which is often utilized for performing knowledge-dependent downstream tasks such as question answering, fact-checking, and open dialogue. In real-world scenarios, the world knowledge stored in the LMs can quickly become outdated as the world changes, but it is non-trivial to avoid catastrophic forgetting and reliably acquire new knowledge while preserving invariant knowledge. To push the community towards better maintenance of ever-changing LMs, we formulate a new continual learning (CL) problem called Continual Knowledge Learning (CKL). We construct a new benchmark and metric to quantify the retention of time-invariant world knowledge, the update of outdated knowledge, and the acquisition of new knowledge. We adopt applicable recent methods from literature to create several strong baselines. Through extensive experiments, we find that CKL exhibits unique challenges that are not addressed in previous CL setups, where parameter expansion is necessary to reliably retain and learn knowledge simultaneously. By highlighting the critical causes of knowledge forgetting, we show that CKL is a challenging and important problem that helps us better understand and train ever-changing LMs.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2110.03215](https://arxiv.org/abs/2110.03215) [cs.CL]** |
|           | (or **[arXiv:2110.03215v1](https://arxiv.org/abs/2110.03215v1) [cs.CL]** for this version) |





# 2021-10-07

[Return to Index](#Index)



<h2 id="2021-10-07-1">1. Sequential Reptile: Inter-Task Gradient Alignment for Multilingual Learning
</h2>

Title: [Sequential Reptile: Inter-Task Gradient Alignment for Multilingual Learning](https://arxiv.org/abs/2110.02600)

Authors: [Seanie Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+S), [Hae Beom Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+H+B), [Juho Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+J), [Sung Ju Hwang](https://arxiv.org/search/cs?searchtype=author&query=Hwang%2C+S+J)

> Multilingual models jointly pretrained on multiple languages have achieved remarkable performance on various multilingual downstream tasks. Moreover, models finetuned on a single monolingual downstream task have shown to generalize to unseen languages. In this paper, we first show that it is crucial for those tasks to align gradients between them in order to maximize knowledge transfer while minimizing negative transfer. Despite its importance, the existing methods for gradient alignment either have a completely different purpose, ignore inter-task alignment, or aim to solve continual learning problems in rather inefficient ways. As a result of the misaligned gradients between tasks, the model suffers from severe negative transfer in the form of catastrophic forgetting of the knowledge acquired from the pretraining. To overcome the limitations, we propose a simple yet effective method that can efficiently align gradients between tasks. Specifically, we perform each inner-optimization by sequentially sampling batches from all the tasks, followed by a Reptile outer update. Thanks to the gradients aligned between tasks by our method, the model becomes less vulnerable to negative transfer and catastrophic forgetting. We extensively validate our method on various multi-task learning and zero-shot cross-lingual transfer tasks, where our method largely outperforms all the relevant baselines we consider.

| Comments: | preprint                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2110.02600](https://arxiv.org/abs/2110.02600) [cs.CL]** |
|           | (or **[arXiv:2110.02600v1](https://arxiv.org/abs/2110.02600v1) [cs.CL]** for this version) |





<h2 id="2021-10-07-2">2. How BPE Affects Memorization in Transformers
</h2>

Title: [How BPE Affects Memorization in Transformers](https://arxiv.org/abs/2110.02782)

Authors: [Eugene Kharitonov](https://arxiv.org/search/cs?searchtype=author&query=Kharitonov%2C+E), [Marco Baroni](https://arxiv.org/search/cs?searchtype=author&query=Baroni%2C+M), [Dieuwke Hupkes](https://arxiv.org/search/cs?searchtype=author&query=Hupkes%2C+D)

> Training data memorization in NLP can both be beneficial (e.g., closed-book QA) and undesirable (personal data extraction). In any case, successful model training requires a non-trivial amount of memorization to store word spellings, various linguistic idiosyncrasies and common knowledge. However, little is known about what affects the memorization behavior of NLP models, as the field tends to focus on the equally important question of generalization. In this work, we demonstrate that the size of the subword vocabulary learned by Byte-Pair Encoding (BPE) greatly affects both ability and tendency of standard Transformer models to memorize training data, even when we control for the number of learned parameters. We find that with a large subword vocabulary size, Transformer models fit random mappings more easily and are more vulnerable to membership inference attacks. Similarly, given a prompt, Transformer-based language models with large subword vocabularies reproduce the training data more often. We conjecture this effect is caused by reduction in the sequences' length that happens as the BPE vocabulary grows. Our findings can allow a more informed choice of hyper-parameters, that is better tailored for a particular use-case.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2110.02782](https://arxiv.org/abs/2110.02782) [cs.CL]** |
|           | (or **[arXiv:2110.02782v1](https://arxiv.org/abs/2110.02782v1) [cs.CL]** for this version) |





<h2 id="2021-10-07-3">3. Sequence-to-Sequence Lexical Normalization with Multilingual Transformers
</h2>

Title: [Sequence-to-Sequence Lexical Normalization with Multilingual Transformers](https://arxiv.org/abs/2110.02869)

Authors: [Ana-Maria Bucur](https://arxiv.org/search/cs?searchtype=author&query=Bucur%2C+A), [Adrian Cosma](https://arxiv.org/search/cs?searchtype=author&query=Cosma%2C+A), [Liviu P. Dinu](https://arxiv.org/search/cs?searchtype=author&query=Dinu%2C+L+P)

> Current benchmark tasks for natural language processing contain text that is qualitatively different from the text used in informal day to day digital communication. This discrepancy has led to severe performance degradation of state-of-the-art NLP models when fine-tuned on real-world data. One way to resolve this issue is through lexical normalization, which is the process of transforming non-standard text, usually from social media, into a more standardized form. In this work, we propose a sentence-level sequence-to-sequence model based on mBART, which frames the problem as a machine translation problem. As the noisy text is a pervasive problem across languages, not just English, we leverage the multi-lingual pre-training of mBART to fine-tune it to our data. While current approaches mainly operate at the word or subword level, we argue that this approach is straightforward from a technical standpoint and builds upon existing pre-trained transformer networks. Our results show that while word-level, intrinsic, performance evaluation is behind other methods, our model improves performance on extrinsic, downstream tasks through normalization compared to models operating on raw, unprocessed, social media text.

| Comments: | Accepted to Proceedings of the 7th Workshop on Noisy User-generated Text (WNUT 2021), EMNLP 2021 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2110.02869](https://arxiv.org/abs/2110.02869) [cs.CL]** |
|           | (or **[arXiv:2110.02869v2](https://arxiv.org/abs/2110.02869v2) [cs.CL]** for this version) |





<h2 id="2021-10-07-4">4. Using Optimal Transport as Alignment Objective for fine-tuning Multilingual Contextualized Embeddings
</h2>

Title: [Using Optimal Transport as Alignment Objective for fine-tuning Multilingual Contextualized Embeddings](https://arxiv.org/abs/2110.02887)

Authors: [Sawsan Alqahtani](https://arxiv.org/search/cs?searchtype=author&query=Alqahtani%2C+S), [Garima Lalwani](https://arxiv.org/search/cs?searchtype=author&query=Lalwani%2C+G), [Yi Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y), [Salvatore Romeo](https://arxiv.org/search/cs?searchtype=author&query=Romeo%2C+S), [Saab Mansour](https://arxiv.org/search/cs?searchtype=author&query=Mansour%2C+S)

> Recent studies have proposed different methods to improve multilingual word representations in contextualized settings including techniques that align between source and target embedding spaces. For contextualized embeddings, alignment becomes more complex as we additionally take context into consideration. In this work, we propose using Optimal Transport (OT) as an alignment objective during fine-tuning to further improve multilingual contextualized representations for downstream cross-lingual transfer. This approach does not require word-alignment pairs prior to fine-tuning that may lead to sub-optimal matching and instead learns the word alignments within context in an unsupervised manner. It also allows different types of mappings due to soft matching between source and target sentences. We benchmark our proposed method on two tasks (XNLI and XQuAD) and achieve improvements over baselines as well as competitive results compared to similar recent works.

| Subjects:          | **Computation and Language (cs.CL)**                         |
| ------------------ | ------------------------------------------------------------ |
| Journal reference: | EMNLP 2021                                                   |
| Cite as:           | **[arXiv:2110.02887](https://arxiv.org/abs/2110.02887) [cs.CL]** |
|                    | (or **[arXiv:2110.02887v1](https://arxiv.org/abs/2110.02887v1) [cs.CL]** for this version) |








# 2021-10-06

[Return to Index](#Index)



<h2 id="2021-10-06-1">1. OPAD: An Optimized Policy-based Active Learning Framework for Document Content Analysis
</h2>

Title: [OPAD: An Optimized Policy-based Active Learning Framework for Document Content Analysis](https://arxiv.org/abs/2110.02069)

Authors: [Sumit Shekhar](https://arxiv.org/search/cs?searchtype=author&query=Shekhar%2C+S), [Bhanu Prakash Reddy Guda](https://arxiv.org/search/cs?searchtype=author&query=Guda%2C+B+P+R), [Ashutosh Chaubey](https://arxiv.org/search/cs?searchtype=author&query=Chaubey%2C+A), [Ishan Jindal](https://arxiv.org/search/cs?searchtype=author&query=Jindal%2C+I), [Avanish Jain](https://arxiv.org/search/cs?searchtype=author&query=Jain%2C+A)

> Documents are central to many business systems, and include forms, reports, contracts, invoices or purchase orders. The information in documents is typically in natural language, but can be organized in various layouts and formats. There have been recent spurt of interest in understanding document content with novel deep learning architectures. However, document understanding tasks need dense information annotations, which are costly to scale and generalize. Several active learning techniques have been proposed to reduce the overall budget of annotation while maintaining the performance of the underlying deep learning model. However, most of these techniques work only for classification problems. But content detection is a more complex task, and has been scarcely explored in active learning literature. In this paper, we propose \textit{OPAD}, a novel framework using reinforcement policy for active learning in content detection tasks for documents. The proposed framework learns the acquisition function to decide the samples to be selected while optimizing performance metrics that the tasks typically have. Furthermore, we extend to weak labelling scenarios to further reduce the cost of annotation significantly. We propose novel rewards to account for class imbalance and user feedback in the annotation interface, to improve the active learning method. We show superior performance of the proposed \textit{OPAD} framework for active learning for various tasks related to document understanding like layout parsing, object detection and named entity recognition. Ablation studies for human feedback and class imbalance rewards are presented, along with a comparison of annotation times for different approaches.

| Subjects: | **Information Retrieval (cs.IR)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2110.02069](https://arxiv.org/abs/2110.02069) [cs.IR]** |
|           | (or **[arXiv:2110.02069v1](https://arxiv.org/abs/2110.02069v1) [cs.IR]** for this version) |





<h2 id="2021-10-06-2">2. Rerunning OCR -- A Machine Learning Approach to Quality Assessment and Enhancement Prediction
</h2>

Title: [Rerunning OCR -- A Machine Learning Approach to Quality Assessment and Enhancement Prediction](https://arxiv.org/abs/2110.01661)

Authors: [Pit Schneider](https://arxiv.org/search/cs?searchtype=author&query=Schneider%2C+P)

> Iterating with new and improved OCR solutions enforces decisions to be taken when it comes to targeting the right reprocessing candidates. This especially applies when the underlying data collection is of considerable size and rather diverse in terms of fonts, languages, periods of publication and consequently OCR quality. This article captures the efforts of the National Library of Luxembourg to support those exact decisions. They are crucial in order to guarantee low computational overhead and reduced quality degradation risks, combined with a more quantifiable OCR improvement. In particular, this work explains the methodology of the library with respect to text block level quality assessment. As an extension of this technique, another contribution comes in the form of a regression model that takes the enhancement potential of a new OCR engine into account. They both mark promising approaches, especially for cultural institutions dealing with historic data of lower quality.

| Comments:    | Journal of Data Mining and Digital Humanities                |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| ACM classes: | I.2.7                                                        |
| Cite as:     | **[arXiv:2110.01661](https://arxiv.org/abs/2110.01661) [cs.CL]** |
|              | (or **[arXiv:2110.01661v1](https://arxiv.org/abs/2110.01661v1) [cs.CL]** for this version) |





<h2 id="2021-10-06-3">3. On the Complementarity between Pre-Training and Back-Translation for Neural Machine Translation
</h2>

Title: [On the Complementarity between Pre-Training and Back-Translation for Neural Machine Translation](https://arxiv.org/abs/2110.01811)

Authors: [Xuebo Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+X), [Longyue Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+L), [Derek F. Wong](https://arxiv.org/search/cs?searchtype=author&query=Wong%2C+D+F), [Liang Ding](https://arxiv.org/search/cs?searchtype=author&query=Ding%2C+L), [Lidia S. Chao](https://arxiv.org/search/cs?searchtype=author&query=Chao%2C+L+S), [Shuming Shi](https://arxiv.org/search/cs?searchtype=author&query=Shi%2C+S), [Zhaopeng Tu](https://arxiv.org/search/cs?searchtype=author&query=Tu%2C+Z)

> Pre-training (PT) and back-translation (BT) are two simple and powerful methods to utilize monolingual data for improving the model performance of neural machine translation (NMT). This paper takes the first step to investigate the complementarity between PT and BT. We introduce two probing tasks for PT and BT respectively and find that PT mainly contributes to the encoder module while BT brings more benefits to the decoder. Experimental results show that PT and BT are nicely complementary to each other, establishing state-of-the-art performances on the WMT16 English-Romanian and English-Russian benchmarks. Through extensive analyses on sentence originality and word frequency, we also demonstrate that combining Tagged BT with PT is more helpful to their complementarity, leading to better translation quality. Source code is freely available at [this https URL](https://github.com/SunbowLiu/PTvsBT).

| Comments: | Accepted to Findings of EMNLP 2021                           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2110.01811](https://arxiv.org/abs/2110.01811) [cs.CL]** |
|           | (or **[arXiv:2110.01811v1](https://arxiv.org/abs/2110.01811v1) [cs.CL]** for this version) |





<h2 id="2021-10-06-4">4. Data Augmentation Approaches in Natural Language Processing: A Survey
</h2>

Title: [Data Augmentation Approaches in Natural Language Processing: A Survey](https://arxiv.org/abs/2110.01852)

Authors: [Bohan Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+B), [Yutai Hou](https://arxiv.org/search/cs?searchtype=author&query=Hou%2C+Y), [Wanxiang Che](https://arxiv.org/search/cs?searchtype=author&query=Che%2C+W)

> As an effective strategy, data augmentation (DA) alleviates data scarcity scenarios where deep learning techniques may fail. It is widely applied in computer vision then introduced to natural language processing and achieves improvements in many tasks. One of the main focuses of the DA methods is to improve the diversity of training data, thereby helping the model to better generalize to unseen testing data. In this survey, we frame DA methods into three categories based on the diversity of augmented data, including paraphrasing, noising, and sampling. Our paper sets out to analyze DA methods in detail according to the above categories. Further, we also introduce their applications in NLP tasks as well as the challenges.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2110.01852](https://arxiv.org/abs/2110.01852) [cs.CL]** |
|           | (or **[arXiv:2110.01852v1](https://arxiv.org/abs/2110.01852v1) [cs.CL]** for this version) |





<h2 id="2021-10-06-5">5. Sicilian Translator: A Recipe for Low-Resource NMT
</h2>

Title: [Sicilian Translator: A Recipe for Low-Resource NMT](https://arxiv.org/abs/2110.01938)

Authors: [Eryk Wdowiak](https://arxiv.org/search/cs?searchtype=author&query=Wdowiak%2C+E)

> With 17,000 pairs of Sicilian-English translated sentences, Arba Sicula developed the first neural machine translator for the Sicilian language. Using small subword vocabularies, we trained small Transformer models with high dropout parameters and achieved BLEU scores in the upper 20s. Then we supplemented our dataset with backtranslation and multilingual translation and pushed our scores into the mid 30s. We also attribute our success to incorporating theoretical information in our dataset. Prior to training, we biased the subword vocabulary towards the desinences one finds in a textbook. And we included textbook exercises in our dataset.

| Comments:    | 7 pages, 2 tables                                            |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Computation and Language (cs.CL)**                         |
| ACM classes: | I.2.7                                                        |
| Cite as:     | **[arXiv:2110.01938](https://arxiv.org/abs/2110.01938) [cs.CL]** |
|              | (or **[arXiv:2110.01938v1](https://arxiv.org/abs/2110.01938v1) [cs.CL]** for this version) |





<h2 id="2021-10-06-6">6. Transfer Learning for Multi-lingual Tasks -- a Survey
</h2>

Title: [Transfer Learning for Multi-lingual Tasks -- a Survey](https://arxiv.org/abs/2110.02052)

Authors: [Amir Reza Jafari](https://arxiv.org/search/cs?searchtype=author&query=Jafari%2C+A+R), [Behnam Heidary](https://arxiv.org/search/cs?searchtype=author&query=Heidary%2C+B), [Reza Farahbakhsh](https://arxiv.org/search/cs?searchtype=author&query=Farahbakhsh%2C+R), [Mostafa Salehi](https://arxiv.org/search/cs?searchtype=author&query=Salehi%2C+M), [Mahdi Jalili](https://arxiv.org/search/cs?searchtype=author&query=Jalili%2C+M)

> These days different platforms such as social media provide their clients from different backgrounds and languages the possibility to connect and exchange information. It is not surprising anymore to see comments from different languages in posts published by international celebrities or data providers. In this era, understanding cross languages content and multilingualism in natural language processing (NLP) are hot topics, and multiple efforts have tried to leverage existing technologies in NLP to tackle this challenging research problem. In this survey, we provide a comprehensive overview of the existing literature with a focus on transfer learning techniques in multilingual tasks. We also identify potential opportunities for further research in this domain.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2110.02052](https://arxiv.org/abs/2110.02052) [cs.CL]** |
|           | (or **[arXiv:2110.02052v1](https://arxiv.org/abs/2110.02052v1) [cs.CL]** for this version) |





<h2 id="2021-10-06-7">7. Structured Prediction in NLP -- A survey
</h2>

Title: [Structured Prediction in NLP -- A survey](https://arxiv.org/abs/2110.02057)

Authors: [Chauhan Dev](https://arxiv.org/search/cs?searchtype=author&query=Dev%2C+C), [Naman Biyani](https://arxiv.org/search/cs?searchtype=author&query=Biyani%2C+N), [Nirmal P. Suthar](https://arxiv.org/search/cs?searchtype=author&query=Suthar%2C+N+P), [Prashant Kumar](https://arxiv.org/search/cs?searchtype=author&query=Kumar%2C+P), [Priyanshu Agarwal](https://arxiv.org/search/cs?searchtype=author&query=Agarwal%2C+P)

> Over the last several years, the field of Structured prediction in NLP has had seen huge advancements with sophisticated probabilistic graphical models, energy-based networks, and its combination with deep learning-based approaches. This survey provides a brief of major techniques in structured prediction and its applications in the NLP domains like parsing, sequence labeling, text generation, and sequence to sequence tasks. We also deep-dived into energy-based and attention-based techniques in structured prediction, identified some relevant open issues and gaps in the current state-of-the-art research, and have come up with some detailed ideas for future research in these fields.

| Comments: | 6 pages, 0 figures                                           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2110.02057](https://arxiv.org/abs/2110.02057) [cs.CL]** |
|           | (or **[arXiv:2110.02057v1](https://arxiv.org/abs/2110.02057v1) [cs.CL]** for this version) |





<h2 id="2021-10-06-8">8. Interactively Generating Explanations for Transformer-based Language Models
</h2>

Title: [Interactively Generating Explanations for Transformer-based Language Models](https://arxiv.org/abs/2110.02058)

Authors: [Patrick Schramowski](https://arxiv.org/search/cs?searchtype=author&query=Schramowski%2C+P), [Felix Friedrich](https://arxiv.org/search/cs?searchtype=author&query=Friedrich%2C+F), [Christopher Tauchmann](https://arxiv.org/search/cs?searchtype=author&query=Tauchmann%2C+C), [Kristian Kersting](https://arxiv.org/search/cs?searchtype=author&query=Kersting%2C+K)

> Transformer language models are state-of-the-art in a multitude of NLP tasks. Despite these successes, their opaqueness remains problematic. Recent methods aiming to provide interpretability and explainability to black-box models primarily focus on post-hoc explanations of (sometimes spurious) input-output correlations. Instead, we emphasize using prototype networks directly incorporated into the model architecture and hence explain the reasoning process behind the network's decisions. Moreover, while our architecture performs on par with several language models, it enables one to learn from user interactions. This not only offers a better understanding of language models, but uses human capabilities to incorporate knowledge outside of the rigid range of purely data-driven approaches.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2110.02058](https://arxiv.org/abs/2110.02058) [cs.CL]** |
|           | (or **[arXiv:2110.02058v1](https://arxiv.org/abs/2110.02058v1) [cs.CL]** for this version) |









# 2021-10-05

[Return to Index](#Index)



<h2 id="2021-10-05-1">1. Improving Zero-shot Multilingual Neural Machine Translation for Low-Resource Languages
</h2>

Title: [Improving Zero-shot Multilingual Neural Machine Translation for Low-Resource Languages](https://arxiv.org/abs/2110.00712)

Authors: [Chenyang Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+C), [Gongxu Luo](https://arxiv.org/search/cs?searchtype=author&query=Luo%2C+G)

> Although the multilingual Neural Machine Translation(NMT), which extends Google's multilingual NMT, has ability to perform zero-shot translation and the iterative self-learning algorithm can improve the quality of zero-shot translation, it confronts with two problems: the multilingual NMT model is prone to generate wrong target language when implementing zero-shot translation; the self-learning algorithm, which uses beam search to generate synthetic parallel data, demolishes the diversity of the generated source language and amplifies the impact of the same noise during the iterative learning process. In this paper, we propose the tagged-multilingual NMT model and improve the self-learning algorithm to handle these two problems. Firstly, we extend the Google's multilingual NMT model and add target tokens to the target languages, which associates the start tag with the target language to ensure that the source language can be translated to the required target language. Secondly, we improve the self-learning algorithm by replacing beam search with random sample to increases the diversity of the generated data and makes it properly cover the true data distribution. Experimental results on IWSLT show that the adjusted tagged-multilingual NMT separately obtains 9.41 and 7.85 BLEU scores over the multilingual NMT on 2010 and 2017 Romanian-Italian test sets. Similarly, it obtains 9.08 and 7.99 BLEU scores on Italian-Romanian zero-shot translation. Furthermore, the improved self-learning algorithm shows its superiorities over the conventional self-learning algorithm on zero-shot translations.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2110.00712](https://arxiv.org/abs/2110.00712) [cs.CL]** |
|           | (or **[arXiv:2110.00712v1](https://arxiv.org/abs/2110.00712v1) [cs.CL]** for this version) |









# 2021-10-04

[Return to Index](#Index)



<h2 id="2021-10-04-1">1. Improving Punctuation Restoration for Speech Transcripts via External Data
</h2>

Title: [Improving Punctuation Restoration for Speech Transcripts via External Data](https://arxiv.org/abs/2110.00560)

Authors:[Xue-Yong Fu](https://arxiv.org/search/cs?searchtype=author&query=Fu%2C+X), [Cheng Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+C), [Md Tahmid Rahman Laskar](https://arxiv.org/search/cs?searchtype=author&query=Laskar%2C+M+T+R), [Shashi Bhushan TN](https://arxiv.org/search/cs?searchtype=author&query=TN%2C+S+B), [Simon Corston-Oliver](https://arxiv.org/search/cs?searchtype=author&query=Corston-Oliver%2C+S)

> Automatic Speech Recognition (ASR) systems generally do not produce punctuated transcripts. To make transcripts more readable and follow the expected input format for downstream language models, it is necessary to add punctuation marks. In this paper, we tackle the punctuation restoration problem specifically for the noisy text (e.g., phone conversation scenarios). To leverage the available written text datasets, we introduce a data sampling technique based on an n-gram language model to sample more training data that are similar to our in-domain data. Moreover, we propose a two-stage fine-tuning approach that utilizes the sampled external data as well as our in-domain dataset for models based on BERT. Extensive experiments show that the proposed approach outperforms the baseline with an improvement of 1:12% F1 score.

| Comments: | Accepted by W-NUT at EMNLP 2021                              |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2110.00560](https://arxiv.org/abs/2110.00560) [cs.CL]** |
|           | (or **[arXiv:2110.00560v1](https://arxiv.org/abs/2110.00560v1) [cs.CL]** for this version) |





<h2 id="2021-10-04-2">2. A Survey of Knowledge Enhanced Pre-trained Models
</h2>

Title: [A Survey of Knowledge Enhanced Pre-trained Models](https://arxiv.org/abs/2110.00269)

Authors:[Jian Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+J), [Gang Xiao](https://arxiv.org/search/cs?searchtype=author&query=Xiao%2C+G), [Yulong Shen](https://arxiv.org/search/cs?searchtype=author&query=Shen%2C+Y), [Wei Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+W), [Xinyu Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu%2C+X), [Ying Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y), [Jinghui Peng](https://arxiv.org/search/cs?searchtype=author&query=Peng%2C+J)

> Pre-trained models learn contextualized word representations on large-scale text corpus through a self-supervised learning method, which has achieved promising performance after fine-tuning. These models, however, suffer from poor robustness and lack of interpretability. Pre-trained models with knowledge injection, which we call knowledge enhanced pre-trained models (KEPTMs), possess deep understanding and logical reasoning and introduce interpretability to some extent. In this survey, we provide a comprehensive overview of KEPTMs for natural language processing. We first introduce the progress of pre-trained models and knowledge representation learning. Then we systematically categorize existing KEPTMs from three different perspectives. Finally, we outline some potential directions of KEPTMs for future research.

| Comments: | 16 pages, 11 figures                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2110.00269](https://arxiv.org/abs/2110.00269) [cs.CL]** |
|           | (or **[arXiv:2110.00269v1](https://arxiv.org/abs/2110.00269v1) [cs.CL]** for this version) |





<h2 id="2021-10-04-3">3. Attention based Sequence to Sequence Learning for Machine Translation of Low Resourced Indic Languages -- A case of Sanskrit to Hindi
</h2>

Title: [Attention based Sequence to Sequence Learning for Machine Translation of Low Resourced Indic Languages -- A case of Sanskrit to Hindi](https://arxiv.org/abs/2110.00435)

Authors:[Vishvajit Bakarola](https://arxiv.org/search/cs?searchtype=author&query=Bakarola%2C+V), [Jitendra Nasriwala](https://arxiv.org/search/cs?searchtype=author&query=Nasriwala%2C+J)

> Deep Learning techniques are powerful in mimicking humans in a particular set of problems. They have achieved a remarkable performance in complex learning tasks. Deep learning inspired Neural Machine Translation (NMT) is a proficient technique that outperforms traditional machine translation. Performing machine-aided translation on Indic languages has always been a challenging task considering their rich and diverse grammar. The neural machine translation has shown quality results compared to the traditional machine translation approaches. The fully automatic machine translation becomes problematic when it comes to low-resourced languages, especially with Sanskrit. This paper presents attention mechanism based neural machine translation by selectively focusing on a particular part of language sentences during translation. The work shows the construction of Sanskrit to Hindi bilingual parallel corpus with nearly 10K samples and having 178,000 tokens. The neural translation model equipped with an attention mechanism has been trained on Sanskrit to Hindi parallel corpus. The approach has shown the significance of attention mechanisms to overcome long-term dependencies, primarily associated with low resources Indic languages. The paper shows the attention plots on testing data to demonstrate the alignment between source and translated words. For the evaluation of the translated sentences, manual score based human evaluation and automatic evaluation metric based techniques have been adopted. The attention mechanism based neural translation has achieved 88% accuracy in human evaluation and a BLEU score of 0.92 on Sanskrit to Hindi translation.

| Comments:          | Published with International Journal of Engineering Trends and Technology (IJETT) |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Journal reference: | International Journal of Engineering Trends and Technology (IJETT) 69.9(2021):230-235 |
| DOI:               | [10.14445/22315381/IJETT-V69I9P227](https://arxiv.org/ct?url=https%3A%2F%2Fdx.doi.org%2F10.14445%2F22315381%2FIJETT-V69I9P227&v=cf20c5d7) |
| Cite as:           | **[arXiv:2110.00435](https://arxiv.org/abs/2110.00435) [cs.CL]** |
|                    | (or **[arXiv:2110.00435v1](https://arxiv.org/abs/2110.00435v1) [cs.CL]** for this version) |





# 2021-10-01

[Return to Index](#Index)



<h2 id="2021-10-01-1">1. Phonetic Word Embeddings
</h2>

Title: [Phonetic Word Embeddings](https://arxiv.org/abs/2109.14796)

Authors: [Rahul Sharma](https://arxiv.org/search/cs?searchtype=author&query=Sharma%2C+R), [Kunal Dhawan](https://arxiv.org/search/cs?searchtype=author&query=Dhawan%2C+K), [Balakrishna Pailla](https://arxiv.org/search/cs?searchtype=author&query=Pailla%2C+B)

> This work presents a novel methodology for calculating the phonetic similarity between words taking motivation from the human perception of sounds. This metric is employed to learn a continuous vector embedding space that groups similar sounding words together and can be used for various downstream computational phonology tasks. The efficacy of the method is presented for two different languages (English, Hindi) and performance gains over previous reported works are discussed on established tests for predicting phonetic similarity. To address limited benchmarking mechanisms in this field, we also introduce a heterographic pun dataset based evaluation methodology to compare the effectiveness of acoustic similarity algorithms. Further, a visualization of the embedding space is presented with a discussion on the various possible use-cases of this novel algorithm. An open-source implementation is also shared to aid reproducibility and enable adoption in related tasks.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2109.14796](https://arxiv.org/abs/2109.14796) [cs.CL]** |
|           | (or **[arXiv:2109.14796v1](https://arxiv.org/abs/2109.14796v1) [cs.CL]** for this version) |





<h2 id="2021-10-01-2">2. Improved statistical machine translation using monolingual paraphrases
</h2>

Title: [Improved statistical machine translation using monolingual paraphrases](https://arxiv.org/abs/2109.15119)

Authors: [Preslav Nakov](https://arxiv.org/search/cs?searchtype=author&query=Nakov%2C+P)

> We propose a novel monolingual sentence paraphrasing method for augmenting the training data for statistical machine translation systems "for free" -- by creating it from data that is already available rather than having to create more aligned data. Starting with a syntactic tree, we recursively generate new sentence variants where noun compounds are paraphrased using suitable prepositions, and vice-versa -- preposition-containing noun phrases are turned into noun compounds. The evaluation shows an improvement equivalent to 33%-50% of that of doubling the amount of training data.

| Comments:          | machine translation, SMT, paraphrasing, data augmentation. arXiv admin note: substantial text overlap with [arXiv:1912.01113](https://arxiv.org/abs/1912.01113) |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| MSC classes:       | 68T50                                                        |
| ACM classes:       | F.2.2; I.2.7                                                 |
| Journal reference: | ECAI-2008                                                    |
| Cite as:           | **[arXiv:2109.15119](https://arxiv.org/abs/2109.15119) [cs.CL]** |
|                    | (or **[arXiv:2109.15119v1](https://arxiv.org/abs/2109.15119v1) [cs.CL]** for this version) |

