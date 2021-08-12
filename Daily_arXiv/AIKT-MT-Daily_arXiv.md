# Daily arXiv: Machine Translation - August, 2021

# Index


- [2021-08-12](#2021-08-12)

  - [1. Embodied BERT: A Transformer Model for Embodied, Language-guided Visual Task Completion](#2021-08-12-1)
  - [2. Post-hoc Interpretability for Neural NLP: A Survey](#2021-08-12-2)
  - [3. A Transformer-based Math Language Model for Handwritten Math Expression Recognition](#2021-08-12-3)
- [2021-08-11](#2021-08-11)

  - [1. FairyTailor: A Multimodal Generative Framework for Storytelling](#2021-08-11-1)
  - [2. BROS: A Layout-Aware Pre-trained Language Model for Understanding Documents](#2021-08-11-2)
  - [3. CLSEBERT: Contrastive Learning for Syntax Enhanced Code Pre-Trained Model](#2021-08-11-3)
  - [4. Differentiable Subset Pruning of Transformer Heads](#2021-08-11-4)
  - [5. How Commonsense Knowledge Helps with Natural Language Tasks: A Survey of Recent Resources and Methodologies](#2021-08-11-5)
  - [6. Sampling-Based Minimum Bayes Risk Decoding for Neural Machine Translation](#2021-08-11-6)
- [2021-08-10](#2021-08-10)

  - [1. Improving Similar Language Translation With Transfer Learning](#2021-08-10-1)
  - [2. Image Retrieval on Real-life Images with Pre-trained Vision-and-Language Models](#2021-08-10-2)
  - [3. Facebook AI WMT21 News Translation Task Submission](#2021-08-10-3)
  - [4. Towards Zero-shot Language Modeling](#2021-08-10-4)
  - [5. Generating Personalized Dialogue via Multi-Task Meta-Learning](#2021-08-10-5)
  - [6. Language Model Evaluation in Open-ended Text Generation](#2021-08-10-6)
  - [7. Machine Translation of Low-Resource Indo-European Languages](#2021-08-10-7)
  - [8. The HW-TSC's Offline Speech Translation Systems for IWSLT 2021 Evaluation](#2021-08-10-8)
  - [9. Multi-modal Retrieval of Tables and Texts Using Tri-encoder Models](#2021-08-10-9)
- [2021-08-09](#2021-08-09)

  - [1. Sentence Semantic Regression for Text Generation](#2021-08-09-1)
  - [2. Lights, Camera, Action! A Framework to Improve NLP Accuracy over OCR documents](#2021-08-09-2)
  - [3. StrucTexT: Structured Text Understanding with Multi-Modal Transformers](#2021-08-09-3)
- [2021-08-06](#2021-08-06)
  - [1. Sentence-level Online Handwritten Chinese Character Recognition](#2021-08-06-1)
  - [2. Evaluation of Audio-Visual Alignments in Visually Grounded Speech Models](#2021-08-06-2)
  - [3. WeChat Neural Machine Translation Systems for WMT21](#2021-08-06-3)
  - [4. Finetuning Pretrained Transformers into Variational Autoencoders](#2021-08-06-4)
  - [5. VisualTextRank: Unsupervised Graph-based Content Extraction for Automating Ad Text to Image Search](#2021-08-06-5)
- [2021-08-05](#2021-08-05)

  - [1. Improving Distinction between ASR Errors and Speech Disfluencies with Feature Space Interpolation](#2021-08-05-1)
  - [2. PARADISE: Exploiting Parallel Data for Multilingual Sequence-to-Sequence Pretraining](#2021-08-05-2)
  - [3. How to Query Language Models?](#2021-08-05-3)
  - [4. Curriculum learning for language modeling](#2021-08-05-4)
- [2021-08-04](#2021-08-04)

  - [1. Knowledge-intensive Language Understanding for Explainable AI](#2021-08-04-1)
  - [2. Underreporting of errors in NLG output, and what to do about it](#2021-08-04-2)
  - [3. A Dynamic Head Importance Computation Mechanism for Neural Machine Translation](#2021-08-04-3)
- [2021-08-03](#2021-08-03)

  - [1. Word2Pix: Word to Pixel Cross Attention Transformer in Visual Grounding](#2021-08-03-1)
  - [2. StyleGAN-NADA: CLIP-Guided Domain Adaptation of Image Generators](#2021-08-03-2)
  - [3. Structural Guidance for Transformer Language Models](#2021-08-03-3)
  - [4. LICHEE: Improving Language Model Pre-training with Multi-grained Tokenization](#2021-08-03-4)
- [2021-08-02](#2021-08-02)

  - [1. Difficulty-Aware Machine Translation Evaluation](#2021-08-02-1)
  - [2. Residual Tree Aggregation of Layers for Neural Machine Translation](#2021-08-02-2)
  - [3. Neural Variational Learning for Grounded Language Acquisition](#2021-08-02-3)
  - [4. Multi-stage Pre-training over Simplified Multimodal Pre-training Models](#2021-08-02-4)
  - [5. MDQE: A More Accurate Direct Pretraining for Machine Translation Quality Estimation](#2021-08-02-5)
  - [6. Towards Universality in Multilingual Text Rewriting](#2021-08-02-6)
  - [7. ChrEnTranslate: Cherokee-English Machine Translation Demo with Quality Estimation and Corrective Feedback](#2021-08-02-7)
- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MT-Daily_arXiv-index.md)



# 2021-08-12

[Return to Index](#Index)



<h2 id="2021-08-12-1">1. Embodied BERT: A Transformer Model for Embodied, Language-guided Visual Task Completion
</h2>

Title: [Embodied BERT: A Transformer Model for Embodied, Language-guided Visual Task Completion](https://arxiv.org/abs/2108.04927)

Authors: [Alessandro Suglia](https://arxiv.org/search/cs?searchtype=author&query=Suglia%2C+A), [Qiaozi Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+Q), [Jesse Thomason](https://arxiv.org/search/cs?searchtype=author&query=Thomason%2C+J), [Govind Thattai](https://arxiv.org/search/cs?searchtype=author&query=Thattai%2C+G), [Gaurav Sukhatme](https://arxiv.org/search/cs?searchtype=author&query=Sukhatme%2C+G)

> Language-guided robots performing home and office tasks must navigate in and interact with the world. Grounding language instructions against visual observations and actions to take in an environment is an open challenge. We present Embodied BERT (EmBERT), a transformer-based model which can attend to high-dimensional, multi-modal inputs across long temporal horizons for language-conditioned task completion. Additionally, we bridge the gap between successful object-centric navigation models used for non-interactive agents and the language-guided visual task completion benchmark, ALFRED, by introducing object navigation targets for EmBERT training. We achieve competitive performance on the ALFRED benchmark, and EmBERT marks the first transformer-based model to successfully handle the long-horizon, dense, multi-modal histories of ALFRED, and the first ALFRED model to utilize object-centric navigation targets.

| Comments: | [this https URL](https://github.com/amazon-research/embert)  |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2108.04927](https://arxiv.org/abs/2108.04927) [cs.CV]** |
|           | (or **[arXiv:2108.04927v1](https://arxiv.org/abs/2108.04927v1) [cs.CV]** for this version) |





<h2 id="2021-08-12-2">2. Post-hoc Interpretability for Neural NLP: A Survey
</h2>

Title: [Post-hoc Interpretability for Neural NLP: A Survey](https://arxiv.org/abs/2108.04840)

Authors: [Andreas Madsen](https://arxiv.org/search/cs?searchtype=author&query=Madsen%2C+A), [Siva Reddy](https://arxiv.org/search/cs?searchtype=author&query=Reddy%2C+S), [Sarath Chandar](https://arxiv.org/search/cs?searchtype=author&query=Chandar%2C+S)

> Natural Language Processing (NLP) models have become increasingly more complex and widespread. With recent developments in neural networks, a growing concern is whether it is responsible to use these models. Concerns such as safety and ethics can be partially addressed by providing explanations. Furthermore, when models do fail, providing explanations is paramount for accountability purposes. To this end, interpretability serves to provide these explanations in terms that are understandable to humans. Central to what is understandable is how explanations are communicated. Therefore, this survey provides a categorization of how recent interpretability methods communicate explanations and discusses the methods in depth. Furthermore, the survey focuses on post-hoc methods, which provide explanations after a model is learned and generally model-agnostic. A common concern for this class of methods is whether they accurately reflect the model. Hence, how these post-hoc methods are evaluated is discussed throughout the paper.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG); Neural and Evolutionary Computing (cs.NE) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2108.04840](https://arxiv.org/abs/2108.04840) [cs.CL]** |
|           | (or **[arXiv:2108.04840v1](https://arxiv.org/abs/2108.04840v1) [cs.CL]** for this version) |





<h2 id="2021-08-12-3">3. A Transformer-based Math Language Model for Handwritten Math Expression Recognition
</h2>

Title: [A Transformer-based Math Language Model for Handwritten Math Expression Recognition](https://arxiv.org/abs/2108.05002)

Authors: [Huy Quang Ung](https://arxiv.org/search/cs?searchtype=author&query=Ung%2C+H+Q), [Cuong Tuan Nguyen](https://arxiv.org/search/cs?searchtype=author&query=Nguyen%2C+C+T), [Hung Tuan Nguyen](https://arxiv.org/search/cs?searchtype=author&query=Nguyen%2C+H+T), [Thanh-Nghia Truong](https://arxiv.org/search/cs?searchtype=author&query=Truong%2C+T), [Masaki Nakagawa](https://arxiv.org/search/cs?searchtype=author&query=Nakagawa%2C+M)

> Handwritten mathematical expressions (HMEs) contain ambiguities in their interpretations, even for humans sometimes. Several math symbols are very similar in the writing style, such as dot and comma or 0, O, and o, which is a challenge for HME recognition systems to handle without using contextual information. To address this problem, this paper presents a Transformer-based Math Language Model (TMLM). Based on the self-attention mechanism, the high-level representation of an input token in a sequence of tokens is computed by how it is related to the previous tokens. Thus, TMLM can capture long dependencies and correlations among symbols and relations in a mathematical expression (ME). We trained the proposed language model using a corpus of approximately 70,000 LaTeX sequences provided in CROHME 2016. TMLM achieved the perplexity of 4.42, which outperformed the previous math language models, i.e., the N-gram and recurrent neural network-based language models. In addition, we combine TMLM into a stochastic context-free grammar-based HME recognition system using a weighting parameter to re-rank the top-10 best candidates. The expression rates on the testing sets of CROHME 2016 and CROHME 2019 were improved by 2.97 and 0.83 percentage points, respectively.

| Comments: | 14 pages, accepted in ICDAR-DIL 2021                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Computer Vision and Pattern Recognition (cs.CV) |
| Cite as:  | **[arXiv:2108.05002](https://arxiv.org/abs/2108.05002) [cs.CL]** |
|           | (or **[arXiv:2108.05002v1](https://arxiv.org/abs/2108.05002v1) [cs.CL]** for this version) |





# 2021-08-11

[Return to Index](#Index)



<h2 id="2021-08-11-1">1. FairyTailor: A Multimodal Generative Framework for Storytelling
</h2>

Title: [FairyTailor: A Multimodal Generative Framework for Storytelling](https://arxiv.org/abs/2108.04324)

Authors: [Eden Bensaid](https://arxiv.org/search/cs?searchtype=author&query=Bensaid%2C+E), [Mauro Martino](https://arxiv.org/search/cs?searchtype=author&query=Martino%2C+M), [Benjamin Hoover](https://arxiv.org/search/cs?searchtype=author&query=Hoover%2C+B), [Jacob Andreas](https://arxiv.org/search/cs?searchtype=author&query=Andreas%2C+J), [Hendrik Strobelt](https://arxiv.org/search/cs?searchtype=author&query=Strobelt%2C+H)

> Storytelling is an open-ended task that entails creative thinking and requires a constant flow of ideas. Natural language generation (NLG) for storytelling is especially challenging because it requires the generated text to follow an overall theme while remaining creative and diverse to engage the reader. In this work, we introduce a system and a web-based demo, FairyTailor, for human-in-the-loop visual story co-creation. Users can create a cohesive children's fairytale by weaving generated texts and retrieved images with their input. FairyTailor adds another modality and modifies the text generation process to produce a coherent and creative sequence of text and images. To our knowledge, this is the first dynamic tool for multimodal story generation that allows interactive co-formation of both texts and images. It allows users to give feedback on co-created stories and share their results.

| Comments: | visit [this https URL](https://fairytailor.org/) and [this https URL](https://github.com/EdenBD/MultiModalStory-demo) for web demo and source code |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV) |
| Cite as:  | **[arXiv:2108.04324](https://arxiv.org/abs/2108.04324) [cs.CL]** |
|           | (or **[arXiv:2108.04324v1](https://arxiv.org/abs/2108.04324v1) [cs.CL]** for this version) |





<h2 id="2021-08-11-2">2. BROS: A Layout-Aware Pre-trained Language Model for Understanding Documents
</h2>

Title: [BROS: A Layout-Aware Pre-trained Language Model for Understanding Documents](https://arxiv.org/abs/2108.04539)

Authors: [Teakgyu Hong](https://arxiv.org/search/cs?searchtype=author&query=Hong%2C+T), [Donghyun Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+D), [Mingi Ji](https://arxiv.org/search/cs?searchtype=author&query=Ji%2C+M), [Wonseok Hwang](https://arxiv.org/search/cs?searchtype=author&query=Hwang%2C+W), [Daehyun Nam](https://arxiv.org/search/cs?searchtype=author&query=Nam%2C+D), [Sungrae Park](https://arxiv.org/search/cs?searchtype=author&query=Park%2C+S)

> Understanding documents from their visual snapshots is an emerging problem that requires both advanced computer vision and NLP methods. The recent advance in OCR enables the accurate recognition of text blocks, yet it is still challenging to extract key information from documents due to the diversity of their layouts. Although recent studies on pre-trained language models show the importance of incorporating layout information on this task, the conjugation of texts and their layouts still follows the style of BERT optimized for understanding the 1D text. This implies there is room for further improvement considering the 2D nature of text layouts. This paper introduces a pre-trained language model, BERT Relying On Spatiality (BROS), which effectively utilizes the information included in individual text blocks and their layouts. Specifically, BROS encodes spatial information by utilizing relative positions and learns spatial dependencies between OCR blocks with a novel area-masking strategy. These two novel approaches lead to an efficient encoding of spatial layout information highlighted by the robust performance of BROS under low-resource environments. We also introduce a general-purpose parser that can be combined with BROS to extract key information even when there is no order information between text blocks. BROS shows its superiority on four public benchmarks---FUNSD, SROIE*, CORD, and SciTSR---and its robustness in practical cases where order information of text blocks is not available. Further experiments with a varying number of training examples demonstrate the high training efficiency of our approach. Our code will be open to the public.

| Comments: | 11 pages, 6 figures                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2108.04539](https://arxiv.org/abs/2108.04539) [cs.CL]** |
|           | (or **[arXiv:2108.04539v1](https://arxiv.org/abs/2108.04539v1) [cs.CL]** for this version) |





<h2 id="2021-08-11-3">3. CLSEBERT: Contrastive Learning for Syntax Enhanced Code Pre-Trained Model
</h2>

Title: [CLSEBERT: Contrastive Learning for Syntax Enhanced Code Pre-Trained Model](https://arxiv.org/abs/2108.04556)

Authors: [Xin Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Yasheng Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Y), [Pingyi Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+P), [Meng Xiao](https://arxiv.org/search/cs?searchtype=author&query=Xiao%2C+M), [Yadao Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Y), [Li Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+L), [Xiao Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+X), [Hao Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+H), [Jin Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+J), [Xin Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+X)

> Pre-trained models for programming languages have proven their significant values in various code-related tasks, such as code search, code clone detection, and code translation. Currently, most pre-trained models treat a code snippet as a sequence of tokens or only focus on the data flow between code identifiers. However, rich code syntax and hierarchy are ignored which can provide important structure information and semantic rules of codes to help enhance code representations. In addition, although the BERT-based code pre-trained models achieve high performance on many downstream tasks, the native derived sequence representations of BERT are proven to be of low-quality, it performs poorly on code matching and similarity tasks. To address these problems, we propose CLSEBERT, a Constrastive Learning Framework for Syntax Enhanced Code Pre-Trained Model, to deal with various code intelligence tasks. In the pre-training stage, we consider the code syntax and hierarchy contained in the Abstract Syntax Tree (AST) and leverage the constrastive learning to learn noise-invariant code representations. Besides the masked language modeling (MLM), we also introduce two novel pre-training objectives. One is to predict the edges between nodes in the abstract syntax tree, and the other is to predict the types of code tokens. Through extensive experiments on four code intelligence tasks, we successfully show the effectiveness of our proposed model.

| Comments: | 10 pages, 3 figures                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Programming Languages (cs.PL) |
| Cite as:  | **[arXiv:2108.04556](https://arxiv.org/abs/2108.04556) [cs.CL]** |
|           | (or **[arXiv:2108.04556v1](https://arxiv.org/abs/2108.04556v1) [cs.CL]** for this version) |





<h2 id="2021-08-11-4">4. Differentiable Subset Pruning of Transformer Heads
</h2>

Title: [Differentiable Subset Pruning of Transformer Heads](https://arxiv.org/abs/2108.04657)

Authors: [Jiaoda Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+J), [Ryan Cotterell](https://arxiv.org/search/cs?searchtype=author&query=Cotterell%2C+R), [Mrinmaya Sachan](https://arxiv.org/search/cs?searchtype=author&query=Sachan%2C+M)

> Multi-head attention, a collection of several attention mechanisms that independently attend to different parts of the input, is the key ingredient in the Transformer (Vaswaniet al., 2017). Recent work has shown, however, that a large proportion of the heads in a Transformer's multi-head attention mechanism can be safely pruned away without significantly harming the performance of the model; such pruning leads to models that are noticeably smaller and faster in practice. Our work introduces a new head pruning technique that we term differentiable subset pruning. Intuitively, our method learns per-head importance variables and then enforces a user-specified hard constraint on the number of unpruned heads. The importance variables are learned via stochastic gradient descent. We conduct experiments on natural language inference and machine translation; we show that differentiable subset pruning performs comparably or better than Voita et al. (2019) while offering the same exact control over the number of heads as Michel et al. (2019).

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2108.04657](https://arxiv.org/abs/2108.04657) [cs.CL]** |
|           | (or **[arXiv:2108.04657v1](https://arxiv.org/abs/2108.04657v1) [cs.CL]** for this version) |





<h2 id="2021-08-11-5">5. How Commonsense Knowledge Helps with Natural Language Tasks: A Survey of Recent Resources and Methodologies
</h2>

Title: [How Commonsense Knowledge Helps with Natural Language Tasks: A Survey of Recent Resources and Methodologies](https://arxiv.org/abs/2108.04674)

Authors: [Yubo Xie](https://arxiv.org/search/cs?searchtype=author&query=Xie%2C+Y), [Pearl Pu](https://arxiv.org/search/cs?searchtype=author&query=Pu%2C+P)

> In this paper, we give an overview of commonsense reasoning in natural language processing, which requires a deeper understanding of the contexts and usually involves inference over implicit external knowledge. We first review some popular commonsense knowledge bases and commonsense reasoning benchmarks, but give more emphasis on the methodologies, including recent approaches that aim at solving some general natural language problems that take advantage of external knowledge bases. Finally, we discuss some future directions in pushing the boundary of commonsense reasoning in natural language processing.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2108.04674](https://arxiv.org/abs/2108.04674) [cs.CL]** |
|           | (or **[arXiv:2108.04674v1](https://arxiv.org/abs/2108.04674v1) [cs.CL]** for this version) |





<h2 id="2021-08-11-6">6. Sampling-Based Minimum Bayes Risk Decoding for Neural Machine Translation
</h2>

Title: [Sampling-Based Minimum Bayes Risk Decoding for Neural Machine Translation](https://arxiv.org/abs/2108.04718)

Authors: [Bryan Eikema](https://arxiv.org/search/cs?searchtype=author&query=Eikema%2C+B), [Wilker Aziz](https://arxiv.org/search/cs?searchtype=author&query=Aziz%2C+W)

> In neural machine translation (NMT), we search for the mode of the model distribution to form predictions. The mode as well as other high probability translations found by beam search have been shown to often be inadequate in a number of ways. This prevents practitioners from improving translation quality through better search, as these idiosyncratic translations end up being selected by the decoding algorithm, a problem known as the beam search curse. Recently, a sampling-based approximation to minimum Bayes risk (MBR) decoding has been proposed as an alternative decision rule for NMT that would likely not suffer from the same problems. We analyse this approximation and establish that it has no equivalent to the beam search curse, i.e. better search always leads to better translations. We also design different approximations aimed at decoupling the cost of exploration from the cost of robust estimation of expected utility. This allows for exploration of much larger hypothesis spaces, which we show to be beneficial. We also show that it can be beneficial to make use of strategies like beam search and nucleus sampling to construct hypothesis spaces efficiently. We show on three language pairs (English into and from German, Romanian, and Nepali) that MBR can improve upon beam search with moderate computation.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2108.04718](https://arxiv.org/abs/2108.04718) [cs.CL]** |
|           | (or **[arXiv:2108.04718v1](https://arxiv.org/abs/2108.04718v1) [cs.CL]** for this version) |









# 2021-08-10

[Return to Index](#Index)



<h2 id="2021-08-10-1">1. Improving Similar Language Translation With Transfer Learning
</h2>

Title: [Improving Similar Language Translation With Transfer Learning](https://arxiv.org/abs/2108.03533)

Authors: [Ife Adebara](https://arxiv.org/search/cs?searchtype=author&query=Adebara%2C+I), [Muhammad Abdul-Mageed](https://arxiv.org/search/cs?searchtype=author&query=Abdul-Mageed%2C+M)

> We investigate transfer learning based on pre-trained neural machine translation models to translate between (low-resource) similar languages. This work is part of our contribution to the WMT 2021 Similar Languages Translation Shared Task where we submitted models for different language pairs, including French-Bambara, Spanish-Catalan, and Spanish-Portuguese in both directions. Our models for Catalan-Spanish (82.79 BLEU) and Portuguese-Spanish (87.11 BLEU) rank top 1 in the official shared task evaluation, and we are the only team to submit models for the French-Bambara pairs.

| Comments: | Submitted to WMT 2021 Similar Language Task                  |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Artificial Intelligence (cs.AI)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2108.03533](https://arxiv.org/abs/2108.03533) [cs.AI]** |
|           | (or **[arXiv:2108.03533v1](https://arxiv.org/abs/2108.03533v1) [cs.AI]** for this version) |





<h2 id="2021-08-10-2">2. Image Retrieval on Real-life Images with Pre-trained Vision-and-Language Models
</h2>

Title: [Image Retrieval on Real-life Images with Pre-trained Vision-and-Language Models](https://arxiv.org/abs/2108.04024)

Authors: [Zheyuan Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Z), [Cristian Rodriguez-Opazo](https://arxiv.org/search/cs?searchtype=author&query=Rodriguez-Opazo%2C+C), [Damien Teney](https://arxiv.org/search/cs?searchtype=author&query=Teney%2C+D), [Stephen Gould](https://arxiv.org/search/cs?searchtype=author&query=Gould%2C+S)

> We extend the task of composed image retrieval, where an input query consists of an image and short textual description of how to modify the image. Existing methods have only been applied to non-complex images within narrow domains, such as fashion products, thereby limiting the scope of study on in-depth visual reasoning in rich image and language contexts. To address this issue, we collect the Compose Image Retrieval on Real-life images (CIRR) dataset, which consists of over 36,000 pairs of crowd-sourced, open-domain images with human-generated modifying text. To extend current methods to the open-domain, we propose CIRPLANT, a transformer based model that leverages rich pre-trained vision-and-language (V&L) knowledge for modifying visual features conditioned on natural language. Retrieval is then done by nearest neighbor lookup on the modified features. We demonstrate that with a relatively simple architecture, CIRPLANT outperforms existing methods on open-domain images, while matching state-of-the-art accuracy on the existing narrow datasets, such as fashion. Together with the release of CIRR, we believe this work will inspire further research on composed image retrieval.

| Comments: | ICCV 2021. Dataset, code, and pre-trained models are released at [this https URL](https://cuberick-orion.github.io/CIRR/) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Information Retrieval (cs.IR) |
| Cite as:  | **[arXiv:2108.04024](https://arxiv.org/abs/2108.04024) [cs.CV]** |
|           | (or **[arXiv:2108.04024v1](https://arxiv.org/abs/2108.04024v1) [cs.CV]** for this version) |





<h2 id="2021-08-10-3">3. Facebook AI WMT21 News Translation Task Submission
</h2>

Title: [Facebook AI WMT21 News Translation Task Submission](https://arxiv.org/abs/2108.03265)

Authors: [Chau Tran](https://arxiv.org/search/cs?searchtype=author&query=Tran%2C+C), [Shruti Bhosale](https://arxiv.org/search/cs?searchtype=author&query=Bhosale%2C+S), [James Cross](https://arxiv.org/search/cs?searchtype=author&query=Cross%2C+J), [Philipp Koehn](https://arxiv.org/search/cs?searchtype=author&query=Koehn%2C+P), [Sergey Edunov](https://arxiv.org/search/cs?searchtype=author&query=Edunov%2C+S), [Angela Fan](https://arxiv.org/search/cs?searchtype=author&query=Fan%2C+A)

> We describe Facebook's multilingual model submission to the WMT2021 shared task on news translation. We participate in 14 language directions: English to and from Czech, German, Hausa, Icelandic, Japanese, Russian, and Chinese. To develop systems covering all these directions, we focus on multilingual models. We utilize data from all available sources --- WMT, large-scale data mining, and in-domain backtranslation --- to create high quality bilingual and multilingual baselines. Subsequently, we investigate strategies for scaling multilingual model size, such that one system has sufficient capacity for high quality representations of all eight languages. Our final submission is an ensemble of dense and sparse Mixture-of-Expert multilingual translation models, followed by finetuning on in-domain news data and noisy channel reranking. Compared to previous year's winning submissions, our multilingual system improved the translation quality on all language directions, with an average improvement of 2.0 BLEU. In the WMT2021 task, our system ranks first in 10 directions based on automatic evaluation.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2108.03265](https://arxiv.org/abs/2108.03265) [cs.CL]** |
|           | (or **[arXiv:2108.03265v1](https://arxiv.org/abs/2108.03265v1) [cs.CL]** for this version) |





<h2 id="2021-08-10-4">4. Towards Zero-shot Language Modeling
</h2>

Title: [Towards Zero-shot Language Modeling](https://arxiv.org/abs/2108.03334)

Authors: [Edoardo Maria Ponti](https://arxiv.org/search/cs?searchtype=author&query=Ponti%2C+E+M), [Ivan Vulić](https://arxiv.org/search/cs?searchtype=author&query=Vulić%2C+I), [Ryan Cotterell](https://arxiv.org/search/cs?searchtype=author&query=Cotterell%2C+R), [Roi Reichart](https://arxiv.org/search/cs?searchtype=author&query=Reichart%2C+R), [Anna Korhonen](https://arxiv.org/search/cs?searchtype=author&query=Korhonen%2C+A)

> Can we construct a neural model that is inductively biased towards learning human languages? Motivated by this question, we aim at constructing an informative prior over neural weights, in order to adapt quickly to held-out languages in the task of character-level language modeling. We infer this distribution from a sample of typologically diverse training languages via Laplace approximation. The use of such a prior outperforms baseline models with an uninformative prior (so-called "fine-tuning") in both zero-shot and few-shot settings. This shows that the prior is imbued with universal phonological knowledge. Moreover, we harness additional language-specific side information as distant supervision for held-out languages. Specifically, we condition language models on features from typological databases, by concatenating them to hidden states or generating weights with hyper-networks. These features appear beneficial in the few-shot setting, but not in the zero-shot setting. Since the paucity of digital texts affects the majority of the world's languages, we hope that these findings will help broaden the scope of applications for language technology.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2108.03334](https://arxiv.org/abs/2108.03334) [cs.CL]** |
|           | (or **[arXiv:2108.03334v1](https://arxiv.org/abs/2108.03334v1) [cs.CL]** for this version) |





<h2 id="2021-08-10-5">5. Generating Personalized Dialogue via Multi-Task Meta-Learning
</h2>

Title: [Generating Personalized Dialogue via Multi-Task Meta-Learning](https://arxiv.org/abs/2108.03377)

Authors: [Jing Yang Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+J+Y), [Kong Aik Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+K+A), [Woon Seng Gan](https://arxiv.org/search/cs?searchtype=author&query=Gan%2C+W+S)

> Conventional approaches to personalized dialogue generation typically require a large corpus, as well as predefined persona information. However, in a real-world setting, neither a large corpus of training data nor persona information are readily available. To address these practical limitations, we propose a novel multi-task meta-learning approach which involves training a model to adapt to new personas without relying on a large corpus, or on any predefined persona information. Instead, the model is tasked with generating personalized responses based on only the dialogue context. Unlike prior work, our approach leverages on the provided persona information only during training via the introduction of an auxiliary persona reconstruction task. In this paper, we introduce 2 frameworks that adopt the proposed multi-task meta-learning approach: the Multi-Task Meta-Learning (MTML) framework, and the Alternating Multi-Task Meta-Learning (AMTML) framework. Experimental results show that utilizing MTML and AMTML results in dialogue responses with greater persona consistency.

| Comments: | Accepted at SemDial 2021 (PotsDial 2021)                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2108.03377](https://arxiv.org/abs/2108.03377) [cs.CL]** |
|           | (or **[arXiv:2108.03377v1](https://arxiv.org/abs/2108.03377v1) [cs.CL]** for this version) |





<h2 id="2021-08-10-6">6. Language Model Evaluation in Open-ended Text Generation
</h2>

Title: [Language Model Evaluation in Open-ended Text Generation](https://arxiv.org/abs/2108.03578)

Authors: [An Nguyen](https://arxiv.org/search/cs?searchtype=author&query=Nguyen%2C+A)

> Although current state-of-the-art language models have achieved impressive results in numerous natural language processing tasks, still they could not solve the problem of producing repetitive, dull and sometimes inconsistent text in open-ended text generation. Studies often attribute this problem to the maximum likelihood training objective, and propose alternative approaches by using stochastic decoding methods or altering the training objective. However, there is still a lack of consistent evaluation metrics to directly compare the efficacy of these solutions. In this work, we study different evaluation metrics that have been proposed to evaluate quality, diversity and consistency of machine-generated text. From there, we propose a practical pipeline to evaluate language models in open-ended generation task, and research on how to improve the model's performance in all dimensions by leveraging different auxiliary training objectives.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2108.03578](https://arxiv.org/abs/2108.03578) [cs.CL]** |
|           | (or **[arXiv:2108.03578v1](https://arxiv.org/abs/2108.03578v1) [cs.CL]** for this version) |





<h2 id="2021-08-10-7">7. Machine Translation of Low-Resource Indo-European Languages
</h2>

Title: [Machine Translation of Low-Resource Indo-European Languages](https://arxiv.org/abs/2108.03739)

Authors: [Wei-Rui Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+W), [Muhammad Abdul-Mageed](https://arxiv.org/search/cs?searchtype=author&query=Abdul-Mageed%2C+M)

> Transfer learning has been an important technique for low-resource neural machine translation. In this work, we build two systems to study how relatedness can benefit the translation performance. The primary system adopts machine translation model pre-trained on related language pair and the contrastive system adopts that pre-trained on unrelated language pair. We show that relatedness is not required for transfer learning to work but does benefit the performance.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2108.03739](https://arxiv.org/abs/2108.03739) [cs.CL]** |
|           | (or **[arXiv:2108.03739v1](https://arxiv.org/abs/2108.03739v1) [cs.CL]** for this version) |





<h2 id="2021-08-10-8">8. The HW-TSC's Offline Speech Translation Systems for IWSLT 2021 Evaluation
</h2>

Title: [The HW-TSC's Offline Speech Translation Systems for IWSLT 2021 Evaluation](https://arxiv.org/abs/2108.03845)

Authors: [Minghan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+M), [Yuxia Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Y), [Chang Su](https://arxiv.org/search/cs?searchtype=author&query=Su%2C+C), [Jiaxin Guo](https://arxiv.org/search/cs?searchtype=author&query=Guo%2C+J), [Yingtao Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y), [Yujia Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Y), [Min Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+M), [Shimin Tao](https://arxiv.org/search/cs?searchtype=author&query=Tao%2C+S), [Xingshan Zeng](https://arxiv.org/search/cs?searchtype=author&query=Zeng%2C+X), [Liangyou Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+L), [Hao Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+H), [Ying Qin](https://arxiv.org/search/cs?searchtype=author&query=Qin%2C+Y)

> This paper describes our work in participation of the IWSLT-2021 offline speech translation task. Our system was built in a cascade form, including a speaker diarization module, an Automatic Speech Recognition (ASR) module and a Machine Translation (MT) module. We directly use the LIUM SpkDiarization tool as the diarization module. The ASR module is trained with three ASR datasets from different sources, by multi-source training, using a modified Transformer encoder. The MT module is pretrained on the large-scale WMT news translation dataset and fine-tuned on the TED corpus. Our method achieves 24.6 BLEU score on the 2021 test set.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2108.03845](https://arxiv.org/abs/2108.03845) [cs.CL]** |
|           | (or **[arXiv:2108.03845v1](https://arxiv.org/abs/2108.03845v1) [cs.CL]** for this version) |





<h2 id="2021-08-10-9">9. Multi-modal Retrieval of Tables and Texts Using Tri-encoder Models
</h2>

Title: [Multi-modal Retrieval of Tables and Texts Using Tri-encoder Models](https://arxiv.org/abs/2108.04049)

Authors: [Bogdan Kostić](https://arxiv.org/search/cs?searchtype=author&query=Kostić%2C+B), [Julian Risch](https://arxiv.org/search/cs?searchtype=author&query=Risch%2C+J), [Timo Möller](https://arxiv.org/search/cs?searchtype=author&query=Möller%2C+T)

> Open-domain extractive question answering works well on textual data by first retrieving candidate texts and then extracting the answer from those candidates. However, some questions cannot be answered by text alone but require information stored in tables. In this paper, we present an approach for retrieving both texts and tables relevant to a question by jointly encoding texts, tables and questions into a single vector space. To this end, we create a new multi-modal dataset based on text and table datasets from related work and compare the retrieval performance of different encoding schemata. We find that dense vector embeddings of transformer models outperform sparse embeddings on four out of six evaluation datasets. Comparing different dense embedding models, tri-encoders, with one encoder for each question, text and table, increase retrieval performance compared to bi-encoders with one encoder for the question and one for both text and tables. We release the newly created multi-modal dataset to the community so that it can be used for training and evaluation.

| Subjects: | **Computation and Language (cs.CL)**; Information Retrieval (cs.IR) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2108.04049](https://arxiv.org/abs/2108.04049) [cs.CL]** |
|           | (or **[arXiv:2108.04049v1](https://arxiv.org/abs/2108.04049v1) [cs.CL]** for this version) |





# 2021-08-09

[Return to Index](#Index)



<h2 id="2021-08-09-1">1. Sentence Semantic Regression for Text Generation
</h2>

Title: [Sentence Semantic Regression for Text Generation](https://arxiv.org/abs/2108.02984)

Authors: [Wei Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+W), [Piji Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+P), [Hai-Tao Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+H)

> Recall the classical text generation works, the generation framework can be briefly divided into two phases: \textbf{idea reasoning} and \textbf{surface realization}. The target of idea reasoning is to figure out the main idea which will be presented in the following talking/writing periods. Surface realization aims to arrange the most appropriate sentence to depict and convey the information distilled from the main idea. However, the current popular token-by-token text generation methods ignore this crucial process and suffer from many serious issues, such as idea/topic drift. To tackle the problems and realize this two-phase paradigm, we propose a new framework named Sentence Semantic Regression (\textbf{SSR}) based on sentence-level language modeling. For idea reasoning, two architectures \textbf{SSR-AR} and \textbf{SSR-NonAR} are designed to conduct sentence semantic regression autoregressively (like GPT2/3) and bidirectionally (like BERT). In the phase of surface realization, a mixed-granularity sentence decoder is designed to generate text with better consistency by jointly incorporating the predicted sentence-level main idea as well as the preceding contextual token-level information. We conduct experiments on four tasks of story ending prediction, story ending generation, dialogue generation, and sentence infilling. The results show that SSR can obtain better performance in terms of automatic metrics and human evaluation.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2108.02984](https://arxiv.org/abs/2108.02984) [cs.CL]** |
|           | (or **[arXiv:2108.02984v1](https://arxiv.org/abs/2108.02984v1) [cs.CL]** for this version) |





<h2 id="2021-08-09-2">2. Lights, Camera, Action! A Framework to Improve NLP Accuracy over OCR documents
</h2>

Title: [Lights, Camera, Action! A Framework to Improve NLP Accuracy over OCR documents](https://arxiv.org/abs/2108.02899)

Authors: [Amit Gupte](https://arxiv.org/search/cs?searchtype=author&query=Gupte%2C+A), [Alexey Romanov](https://arxiv.org/search/cs?searchtype=author&query=Romanov%2C+A), [Sahitya Mantravadi](https://arxiv.org/search/cs?searchtype=author&query=Mantravadi%2C+S), [Dalitso Banda](https://arxiv.org/search/cs?searchtype=author&query=Banda%2C+D), [Jianjie Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+J), [Raza Khan](https://arxiv.org/search/cs?searchtype=author&query=Khan%2C+R), [Lakshmanan Ramu Meenal](https://arxiv.org/search/cs?searchtype=author&query=Meenal%2C+L+R), [Benjamin Han](https://arxiv.org/search/cs?searchtype=author&query=Han%2C+B), [Soundar Srinivasan](https://arxiv.org/search/cs?searchtype=author&query=Srinivasan%2C+S)

> Document digitization is essential for the digital transformation of our societies, yet a crucial step in the process, Optical Character Recognition (OCR), is still not perfect. Even commercial OCR systems can produce questionable output depending on the fidelity of the scanned documents. In this paper, we demonstrate an effective framework for mitigating OCR errors for any downstream NLP task, using Named Entity Recognition (NER) as an example. We first address the data scarcity problem for model training by constructing a document synthesis pipeline, generating realistic but degraded data with NER labels. We measure the NER accuracy drop at various degradation levels and show that a text restoration model, trained on the degraded data, significantly closes the NER accuracy gaps caused by OCR errors, including on an out-of-domain dataset. For the benefit of the community, we have made the document synthesis pipeline available as an open-source project.

| Comments: | Accepted to the Document Intelligence Workshop at KDD 2021. The source code of Genalog is available at [this https URL](https://github.com/microsoft/genalog) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2108.02899](https://arxiv.org/abs/2108.02899) [cs.CL]** |
|           | (or **[arXiv:2108.02899v1](https://arxiv.org/abs/2108.02899v1) [cs.CL]** for this version) |





<h2 id="2021-08-09-3">3. StrucTexT: Structured Text Understanding with Multi-Modal Transformers
</h2>

Title: [StrucTexT: Structured Text Understanding with Multi-Modal Transformers](https://arxiv.org/abs/2108.02923)

Authors: [Yulin Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Y), [Yuxi Qian](https://arxiv.org/search/cs?searchtype=author&query=Qian%2C+Y), [Yuchen Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+Y), [Xiameng Qin](https://arxiv.org/search/cs?searchtype=author&query=Qin%2C+X), [Chengquan Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+C), [Yan Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Y), [Kun Yao](https://arxiv.org/search/cs?searchtype=author&query=Yao%2C+K), [Junyu Han](https://arxiv.org/search/cs?searchtype=author&query=Han%2C+J), [Jingtuo Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+J), [Errui Ding](https://arxiv.org/search/cs?searchtype=author&query=Ding%2C+E)

> Structured text understanding on Visually Rich Documents (VRDs) is a crucial part of Document Intelligence. Due to the complexity of content and layout in VRDs, structured text understanding has been a challenging task. Most existing studies decoupled this problem into two sub-tasks: entity labeling and entity linking, which require an entire understanding of the context of documents at both token and segment levels. However, little work has been concerned with the solutions that efficiently extract the structured data from different levels. This paper proposes a unified framework named StrucTexT, which is flexible and effective for handling both sub-tasks. Specifically, based on the transformer, we introduce a segment-token aligned encoder to deal with the entity labeling and entity linking tasks at different levels of granularity. Moreover, we design a novel pre-training strategy with three self-supervised tasks to learn a richer representation. StrucTexT uses the existing Masked Visual Language Modeling task and the new Sentence Length Prediction and Paired Boxes Direction tasks to incorporate the multi-modal information across text, image, and layout. We evaluate our method for structured text understanding at segment-level and token-level and show it outperforms the state-of-the-art counterparts with significantly superior performance on the FUNSD, SROIE, and EPHOIE datasets.

| Comments: | 9 pages                                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2108.02923](https://arxiv.org/abs/2108.02923) [cs.CV]** |
|           | (or **[arXiv:2108.02923v1](https://arxiv.org/abs/2108.02923v1) [cs.CV]** for this version) |








# 2021-08-06

[Return to Index](#Index)



<h2 id="2021-08-06-1">1. Sentence-level Online Handwritten Chinese Character Recognition
</h2>

Title: [Sentence-level Online Handwritten Chinese Character Recognition](https://arxiv.org/abs/2108.02561)

Authors: [Yunxin Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Y), [Qian Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Q), [Qingcai Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Q), [Lin Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+L), [Baotian Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu%2C+B), [Xiaolong Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Yuxin Ding](https://arxiv.org/search/cs?searchtype=author&query=Ding%2C+Y)

> Single online handwritten Chinese character recognition~(single OLHCCR) has achieved prominent performance. However, in real application scenarios, users always write multiple Chinese characters to form one complete sentence and the contextual information within these characters holds the significant potential to improve the accuracy, robustness and efficiency of sentence-level OLHCCR. In this work, we first propose a simple and straightforward end-to-end network, namely vanilla compositional network~(VCN) to tackle the sentence-level OLHCCR. It couples convolutional neural network with sequence modeling architecture to exploit the handwritten character's previous contextual information. Although VCN performs much better than the state-of-the-art single OLHCCR model, it exposes high fragility when confronting with not well written characters such as sloppy writing, missing or broken strokes. To improve the robustness of sentence-level OLHCCR, we further propose a novel deep spatial-temporal fusion network~(DSTFN). It utilizes a pre-trained autoregresssive framework as the backbone component, which projects each Chinese character into word embeddings, and integrates the spatial glyph features of handwritten characters and their contextual information multiple times at multi-layer fusion module. We also construct a large-scale sentence-level handwriting dataset, named as CSOHD to evaluate models. Extensive experiment results demonstrate that DSTFN achieves the state-of-the-art performance, which presents strong robustness compared with VCN and exiting single OLHCCR models. The in-depth empirical analysis and case studies indicate that DSTFN can significantly improve the efficiency of handwriting input, with the handwritten Chinese character with incomplete strokes being recognized precisely.

| Comments: | 10 pages, 10 figures                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2108.02561](https://arxiv.org/abs/2108.02561) [cs.CV]** |
|           | (or **[arXiv:2108.02561v1](https://arxiv.org/abs/2108.02561v1) [cs.CV]** for this version) |





<h2 id="2021-08-06-2">2. Evaluation of Audio-Visual Alignments in Visually Grounded Speech Models
</h2>

Title: [Evaluation of Audio-Visual Alignments in Visually Grounded Speech Models](https://arxiv.org/abs/2108.02562)

Authors: [Khazar Khorrami](https://arxiv.org/search/cs?searchtype=author&query=Khorrami%2C+K), [Okko Räsänen](https://arxiv.org/search/cs?searchtype=author&query=Räsänen%2C+O)

> Systems that can find correspondences between multiple modalities, such as between speech and images, have great potential to solve different recognition and data analysis tasks in an unsupervised manner. This work studies multimodal learning in the context of visually grounded speech (VGS) models, and focuses on their recently demonstrated capability to extract spatiotemporal alignments between spoken words and the corresponding visual objects without ever been explicitly trained for object localization or word recognition. As the main contributions, we formalize the alignment problem in terms of an audiovisual alignment tensor that is based on earlier VGS work, introduce systematic metrics for evaluating model performance in aligning visual objects and spoken words, and propose a new VGS model variant for the alignment task utilizing cross-modal attention layer. We test our model and a previously proposed model in the alignment task using SPEECH-COCO captions coupled with MSCOCO images. We compare the alignment performance using our proposed evaluation metrics to the semantic retrieval task commonly used to evaluate VGS models. We show that cross-modal attention layer not only helps the model to achieve higher semantic cross-modal retrieval performance, but also leads to substantial improvements in the alignment performance between image object and spoken words.

| Comments: | To be published in Proc. Interspeech-2021, Brno, Czech Republic |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2108.02562](https://arxiv.org/abs/2108.02562) [cs.CV]** |
|           | (or **[arXiv:2108.02562v1](https://arxiv.org/abs/2108.02562v1) [cs.CV]** for this version) |





<h2 id="2021-08-06-3">3. WeChat Neural Machine Translation Systems for WMT21
</h2>

Title: [WeChat Neural Machine Translation Systems for WMT21](https://arxiv.org/abs/2108.02401)

Authors: [Xianfeng Zeng](https://arxiv.org/search/cs?searchtype=author&query=Zeng%2C+X), [Yijin Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Y), [Ernan Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+E), [Qiu Ran](https://arxiv.org/search/cs?searchtype=author&query=Ran%2C+Q), [Fandong Meng](https://arxiv.org/search/cs?searchtype=author&query=Meng%2C+F), [Peng Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+P), [Jinan Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+J), [Jie Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+J)

> This paper introduces WeChat AI's participation in WMT 2021 shared news translation task on English->Chinese, English->Japanese, Japanese->English and English->German. Our systems are based on the Transformer (Vaswani et al., 2017) with several novel and effective variants. In our experiments, we employ data filtering, large-scale synthetic data generation (i.e., back-translation, knowledge distillation, forward-translation, iterative in-domain knowledge transfer), advanced finetuning approaches, and boosted Self-BLEU based model ensemble. Our constrained systems achieve 36.9, 46.9, 27.8 and 31.3 case-sensitive BLEU scores on English->Chinese, English->Japanese, Japanese->English and English->German, respectively. The BLEU scores of English->Chinese, English->Japanese and Japanese->English are the highest among all submissions, and that of English->German is the highest among all constrained submissions.

| Comments: | Submitted to WMT 2021 as a system paper                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2108.02401](https://arxiv.org/abs/2108.02401) [cs.CL]** |
|           | (or **[arXiv:2108.02401v1](https://arxiv.org/abs/2108.02401v1) [cs.CL]** for this version) |





<h2 id="2021-08-06-4">4. Finetuning Pretrained Transformers into Variational Autoencoders
</h2>

Title: [Finetuning Pretrained Transformers into Variational Autoencoders](https://arxiv.org/abs/2108.02446)

Authors: [Seongmin Park](https://arxiv.org/search/cs?searchtype=author&query=Park%2C+S), [Jihwa Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+J)

> Text variational autoencoders (VAEs) are notorious for posterior collapse, a phenomenon where the model's decoder learns to ignore signals from the encoder. Because posterior collapse is known to be exacerbated by expressive decoders, Transformers have seen limited adoption as components of text VAEs. Existing studies that incorporate Transformers into text VAEs (Li et al., 2020; Fang et al., 2021) mitigate posterior collapse using massive pretraining, a technique unavailable to most of the research community without extensive computing resources. We present a simple two-phase training scheme to convert a sequence-to-sequence Transformer into a VAE with just finetuning. The resulting language model is competitive with massively pretrained Transformer-based VAEs in some internal metrics while falling short on others. To facilitate training we comprehensively explore the impact of common posterior collapse alleviation techniques in the literature. We release our code for reproducability.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2108.02446](https://arxiv.org/abs/2108.02446) [cs.CL]** |
|           | (or **[arXiv:2108.02446v1](https://arxiv.org/abs/2108.02446v1) [cs.CL]** for this version) |





<h2 id="2021-08-06-5">5. VisualTextRank: Unsupervised Graph-based Content Extraction for Automating Ad Text to Image Search
</h2>

Title: [VisualTextRank: Unsupervised Graph-based Content Extraction for Automating Ad Text to Image Search](https://arxiv.org/abs/2108.02725)

Authors: [Shaunak Mishra](https://arxiv.org/search/cs?searchtype=author&query=Mishra%2C+S), [Mikhail Kuznetsov](https://arxiv.org/search/cs?searchtype=author&query=Kuznetsov%2C+M), [Gaurav Srivastava](https://arxiv.org/search/cs?searchtype=author&query=Srivastava%2C+G), [Maxim Sviridenko](https://arxiv.org/search/cs?searchtype=author&query=Sviridenko%2C+M)

> Numerous online stock image libraries offer high quality yet copyright free images for use in marketing campaigns. To assist advertisers in navigating such third party libraries, we study the problem of automatically fetching relevant ad images given the ad text (via a short textual query for images). Motivated by our observations in logged data on ad image search queries (given ad text), we formulate a keyword extraction problem, where a keyword extracted from the ad text (or its augmented version) serves as the ad image query. In this context, we propose VisualTextRank: an unsupervised method to (i) augment input ad text using semantically similar ads, and (ii) extract the image query from the augmented ad text. VisualTextRank builds on prior work on graph based context extraction (biased TextRank in particular) by leveraging both the text and image of similar ads for better keyword extraction, and using advertiser category specific biasing with sentence-BERT embeddings. Using data collected from the Verizon Media Native (Yahoo Gemini) ad platform's stock image search feature for onboarding advertisers, we demonstrate the superiority of VisualTextRank compared to competitive keyword extraction baselines (including an 11% accuracy lift over biased TextRank). For the case when the stock image library is restricted to English queries, we show the effectiveness of VisualTextRank on multilingual ads (translated to English) while leveraging semantically similar English ads. Online tests with a simplified version of VisualTextRank led to a 28.7% increase in the usage of stock image search, and a 41.6% increase in the advertiser onboarding rate in the Verizon Media Native ad platform.

| Comments: | Accepted for publication at KDD 2021                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Computer Vision and Pattern Recognition (cs.CV) |
| DOI:      | [10.1145/1122445.1122456](https://arxiv.org/ct?url=https%3A%2F%2Fdx.doi.org%2F10.1145%2F1122445.1122456&v=a56ae4b4) |
| Cite as:  | **[arXiv:2108.02725](https://arxiv.org/abs/2108.02725) [cs.CL]** |
|           | (or **[arXiv:2108.02725v1](https://arxiv.org/abs/2108.02725v1) [cs.CL]** for this version) |





# 2021-08-05

[Return to Index](#Index)



<h2 id="2021-08-05-1">1. Improving Distinction between ASR Errors and Speech Disfluencies with Feature Space Interpolation
</h2>

Title: [Improving Distinction between ASR Errors and Speech Disfluencies with Feature Space Interpolation](https://arxiv.org/abs/2108.01812)

Authors: [Seongmin Park](https://arxiv.org/search/cs?searchtype=author&query=Park%2C+S), [Dongchan Shin](https://arxiv.org/search/cs?searchtype=author&query=Shin%2C+D), [Sangyoun Paik](https://arxiv.org/search/cs?searchtype=author&query=Paik%2C+S), [Subong Choi](https://arxiv.org/search/cs?searchtype=author&query=Choi%2C+S), [Alena Kazakova](https://arxiv.org/search/cs?searchtype=author&query=Kazakova%2C+A), [Jihwa Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+J)

> Fine-tuning pretrained language models (LMs) is a popular approach to automatic speech recognition (ASR) error detection during post-processing. While error detection systems often take advantage of statistical language archetypes captured by LMs, at times the pretrained knowledge can hinder error detection performance. For instance, presence of speech disfluencies might confuse the post-processing system into tagging disfluent but accurate transcriptions as ASR errors. Such confusion occurs because both error detection and disfluency detection tasks attempt to identify tokens at statistically unlikely positions. This paper proposes a scheme to improve existing LM-based ASR error detection systems, both in terms of detection scores and resilience to such distracting auxiliary tasks. Our approach adopts the popular mixup method in text feature space and can be utilized with any black-box ASR output. To demonstrate the effectiveness of our method, we conduct post-processing experiments with both traditional and end-to-end ASR systems (both for English and Korean languages) with 5 different speech corpora. We find that our method improves both ASR error detection F 1 scores and reduces the number of correctly transcribed disfluencies wrongly detected as ASR errors. Finally, we suggest methods to utilize resulting LMs directly in semi-supervised ASR training.

| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2108.01812](https://arxiv.org/abs/2108.01812) [cs.CL]** |
|           | (or **[arXiv:2108.01812v1](https://arxiv.org/abs/2108.01812v1) [cs.CL]** for this version) |





<h2 id="2021-08-05-2">2. PARADISE: Exploiting Parallel Data for Multilingual Sequence-to-Sequence Pretraining
</h2>

Title: [PARADISE: Exploiting Parallel Data for Multilingual Sequence-to-Sequence Pretraining](https://arxiv.org/abs/2108.01887)

Authors: [Machel Reid](https://arxiv.org/search/cs?searchtype=author&query=Reid%2C+M), [Mikel Artetxe](https://arxiv.org/search/cs?searchtype=author&query=Artetxe%2C+M)

> Despite the success of multilingual sequence-to-sequence pretraining, most existing approaches rely on monolingual corpora, and do not make use of the strong cross-lingual signal contained in parallel data. In this paper, we present PARADISE (PARAllel & Denoising Integration in SEquence-to-sequence models), which extends the conventional denoising objective used to train these models by (i) replacing words in the noised sequence according to a multilingual dictionary, and (ii) predicting the reference translation according to a parallel corpus instead of recovering the original sequence. Our experiments on machine translation and cross-lingual natural language inference show an average improvement of 2.0 BLEU points and 6.7 accuracy points from integrating parallel data into pretraining, respectively, obtaining results that are competitive with several popular models at a fraction of their computational cost.

| Comments: | Preprint                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2108.01887](https://arxiv.org/abs/2108.01887) [cs.CL]** |
|           | (or **[arXiv:2108.01887v1](https://arxiv.org/abs/2108.01887v1) [cs.CL]** for this version) |



<h2 id="2021-08-05-3">3. How to Query Language Models?
</h2>

Title: [How to Query Language Models?](https://arxiv.org/abs/2108.01928)

Authors: [Leonard Adolphs](https://arxiv.org/search/cs?searchtype=author&query=Adolphs%2C+L), [Shehzaad Dhuliawala](https://arxiv.org/search/cs?searchtype=author&query=Dhuliawala%2C+S), [Thomas Hofmann](https://arxiv.org/search/cs?searchtype=author&query=Hofmann%2C+T)

> Large pre-trained language models (LMs) are capable of not only recovering linguistic but also factual and commonsense knowledge. To access the knowledge stored in mask-based LMs, we can use cloze-style questions and let the model fill in the blank. The flexibility advantage over structured knowledge bases comes with the drawback of finding the right query for a certain information need. Inspired by human behavior to disambiguate a question, we propose to query LMs by example. To clarify the ambivalent question "Who does Neuer play for?", a successful strategy is to demonstrate the relation using another subject, e.g., "Ronaldo plays for Portugal. Who does Neuer play for?". We apply this approach of querying by example to the LAMA probe and obtain substantial improvements of up to 37.8% for BERT-large on the T-REx data when providing only 10 demonstrations--even outperforming a baseline that queries the model with up to 40 paraphrases of the question. The examples are provided through the model's context and thus require neither fine-tuning nor an additional forward pass. This suggests that LMs contain more factual and commonsense knowledge than previously assumed--if we query the model in the right way.

| Subjects: | **Computation and Language (cs.CL)**; Information Retrieval (cs.IR); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2108.01928](https://arxiv.org/abs/2108.01928) [cs.CL]** |
|           | (or **[arXiv:2108.01928v1](https://arxiv.org/abs/2108.01928v1) [cs.CL]** for this version) |





<h2 id="2021-08-05-4">4. Curriculum learning for language modeling
</h2>

Title: [Curriculum learning for language modeling](https://arxiv.org/abs/2108.02170)

Authors: [Daniel Campos](https://arxiv.org/search/cs?searchtype=author&query=Campos%2C+D)

> Language Models like ELMo and BERT have provided robust representations of natural language, which serve as the language understanding component for a diverse range of downstream tasks.Curriculum learning is a method that employs a structured training regime instead, which has been leveraged in computer vision and machine translation to improve model training speed and model performance. While language models have proven transformational for the natural language processing community, these models have proven expensive, energy-intensive, and challenging to train. In this work, we explore the effect of curriculum learning on language model pretraining using various linguistically motivated curricula and evaluate transfer performance on the GLUE Benchmark. Despite a broad variety of training methodologies and experiments we do not find compelling evidence that curriculum learning methods improve language model training.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2108.02170](https://arxiv.org/abs/2108.02170) [cs.CL]** |
|           | (or **[arXiv:2108.02170v1](https://arxiv.org/abs/2108.02170v1) [cs.CL]** for this version) |







# 2021-08-04

[Return to Index](#Index)



<h2 id="2021-08-04-1">1. Knowledge-intensive Language Understanding for Explainable AI
</h2>

Title: [Knowledge-intensive Language Understanding for Explainable AI](https://arxiv.org/abs/2108.01174)

Authors: [Amit Sheth](https://arxiv.org/search/cs?searchtype=author&query=Sheth%2C+A), [Manas Gaur](https://arxiv.org/search/cs?searchtype=author&query=Gaur%2C+M), [Kaushik Roy](https://arxiv.org/search/cs?searchtype=author&query=Roy%2C+K), [Keyur Faldu](https://arxiv.org/search/cs?searchtype=author&query=Faldu%2C+K)

> AI systems have seen significant adoption in various domains. At the same time, further adoption in some domains is hindered by inability to fully trust an AI system that it will not harm a human. Besides the concerns for fairness, privacy, transparency, and explainability are key to developing trusts in AI systems. As stated in describing trustworthy AI "Trust comes through understanding. How AI-led decisions are made and what determining factors were included are crucial to understand." The subarea of explaining AI systems has come to be known as XAI. Multiple aspects of an AI system can be explained; these include biases that the data might have, lack of data points in a particular region of the example space, fairness of gathering the data, feature importances, etc. However, besides these, it is critical to have human-centered explanations that are directly related to decision-making similar to how a domain expert makes decisions based on "domain knowledge," that also include well-established, peer-validated explicit guidelines. To understand and validate an AI system's outcomes (such as classification, recommendations, predictions), that lead to developing trust in the AI system, it is necessary to involve explicit domain knowledge that humans understand and use.

| Comments: | To appear in IEEE Internet Computing, September/October 2021 Issue |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Artificial Intelligence (cs.AI)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2108.01174](https://arxiv.org/abs/2108.01174) [cs.AI]** |
|           | (or **[arXiv:2108.01174v1](https://arxiv.org/abs/2108.01174v1) [cs.AI]** for this version) |





<h2 id="2021-08-04-2">2. Underreporting of errors in NLG output, and what to do about it
</h2>

Title: [Underreporting of errors in NLG output, and what to do about it](https://arxiv.org/abs/2108.01182)

Authors: [Emiel van Miltenburg](https://arxiv.org/search/cs?searchtype=author&query=van+Miltenburg%2C+E), [Miruna-Adriana Clinciu](https://arxiv.org/search/cs?searchtype=author&query=Clinciu%2C+M), [Ondřej Dušek](https://arxiv.org/search/cs?searchtype=author&query=Dušek%2C+O), [Dimitra Gkatzia](https://arxiv.org/search/cs?searchtype=author&query=Gkatzia%2C+D), [Stephanie Inglis](https://arxiv.org/search/cs?searchtype=author&query=Inglis%2C+S), [Leo Leppänen](https://arxiv.org/search/cs?searchtype=author&query=Leppänen%2C+L), [Saad Mahamood](https://arxiv.org/search/cs?searchtype=author&query=Mahamood%2C+S), [Emma Manning](https://arxiv.org/search/cs?searchtype=author&query=Manning%2C+E), [Stephanie Schoch](https://arxiv.org/search/cs?searchtype=author&query=Schoch%2C+S), [Craig Thomson](https://arxiv.org/search/cs?searchtype=author&query=Thomson%2C+C), [Luou Wen](https://arxiv.org/search/cs?searchtype=author&query=Wen%2C+L)

> We observe a severe under-reporting of the different kinds of errors that Natural Language Generation systems make. This is a problem, because mistakes are an important indicator of where systems should still be improved. If authors only report overall performance metrics, the research community is left in the dark about the specific weaknesses that are exhibited by `state-of-the-art' research. Next to quantifying the extent of error under-reporting, this position paper provides recommendations for error identification, analysis and reporting.

| Comments: | Prefinal version, accepted for publication in the Proceedings of the 14th International Conference on Natural Language Generation (INLG 2021, Aberdeen). Comments welcome |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2108.01182](https://arxiv.org/abs/2108.01182) [cs.CL]** |
|           | (or **[arXiv:2108.01182v1](https://arxiv.org/abs/2108.01182v1) [cs.CL]** for this version) |





<h2 id="2021-08-04-3">3. A Dynamic Head Importance Computation Mechanism for Neural Machine Translation
</h2>

Title: [A Dynamic Head Importance Computation Mechanism for Neural Machine Translation](https://arxiv.org/abs/2108.01377)

Authors: [Akshay Goindani](https://arxiv.org/search/cs?searchtype=author&query=Goindani%2C+A), [Manish Shrivastava](https://arxiv.org/search/cs?searchtype=author&query=Shrivastava%2C+M)

> Multiple parallel attention mechanisms that use multiple attention heads facilitate greater performance of the Transformer model for various applications e.g., Neural Machine Translation (NMT), text classification. In multi-head attention mechanism, different heads attend to different parts of the input. However, the limitation is that multiple heads might attend to the same part of the input, resulting in multiple heads being redundant. Thus, the model resources are under-utilized. One approach to avoid this is to prune least important heads based on certain importance score. In this work, we focus on designing a Dynamic Head Importance Computation Mechanism (DHICM) to dynamically calculate the importance of a head with respect to the input. Our insight is to design an additional attention layer together with multi-head attention, and utilize the outputs of the multi-head attention along with the input, to compute the importance for each head. Additionally, we add an extra loss function to prevent the model from assigning same score to all heads, to identify more important heads and improvise performance. We analyzed performance of DHICM for NMT with different languages. Experiments on different datasets show that DHICM outperforms traditional Transformer-based approach by large margin, especially, when less training data is available.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2108.01377](https://arxiv.org/abs/2108.01377) [cs.CL]** |
|           | (or **[arXiv:2108.01377v1](https://arxiv.org/abs/2108.01377v1) [cs.CL]** for this version) |





# 2021-08-03

[Return to Index](#Index)



<h2 id="2021-08-03-1">1. Word2Pix: Word to Pixel Cross Attention Transformer in Visual Grounding
</h2>

Title: [Word2Pix: Word to Pixel Cross Attention Transformer in Visual Grounding](https://arxiv.org/abs/2108.00205)

Authors: [Heng Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+H), [Joey Tianyi Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+J+T), [Yew-Soon Ong](https://arxiv.org/search/cs?searchtype=author&query=Ong%2C+Y)

> Current one-stage methods for visual grounding encode the language query as one holistic sentence embedding before fusion with visual feature. Such a formulation does not treat each word of a query sentence on par when modeling language to visual attention, therefore prone to neglect words which are less important for sentence embedding but critical for visual grounding. In this paper we propose Word2Pix: a one-stage visual grounding network based on encoder-decoder transformer architecture that enables learning for textual to visual feature correspondence via word to pixel attention. The embedding of each word from the query sentence is treated alike by attending to visual pixels individually instead of single holistic sentence embedding. In this way, each word is given equivalent opportunity to adjust the language to vision attention towards the referent target through multiple stacks of transformer decoder layers. We conduct the experiments on RefCOCO, RefCOCO+ and RefCOCOg datasets and the proposed Word2Pix outperforms existing one-stage methods by a notable margin. The results obtained also show that Word2Pix surpasses two-stage visual grounding models, while at the same time keeping the merits of one-stage paradigm namely end-to-end training and real-time inference speed intact.

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2108.00205](https://arxiv.org/abs/2108.00205) [cs.CV]** |
|           | (or **[arXiv:2108.00205v1](https://arxiv.org/abs/2108.00205v1) [cs.CV]** for this version) |





<h2 id="2021-08-03-2">2. StyleGAN-NADA: CLIP-Guided Domain Adaptation of Image Generators
</h2>

Title: [StyleGAN-NADA: CLIP-Guided Domain Adaptation of Image Generators](https://arxiv.org/abs/2108.00946)

Authors: [Rinon Gal](https://arxiv.org/search/cs?searchtype=author&query=Gal%2C+R), [Or Patashnik](https://arxiv.org/search/cs?searchtype=author&query=Patashnik%2C+O), [Haggai Maron](https://arxiv.org/search/cs?searchtype=author&query=Maron%2C+H), [Gal Chechik](https://arxiv.org/search/cs?searchtype=author&query=Chechik%2C+G), [Daniel Cohen-Or](https://arxiv.org/search/cs?searchtype=author&query=Cohen-Or%2C+D)

> Can a generative model be trained to produce images from a specific domain, guided by a text prompt only, without seeing any image? In other words: can an image generator be trained blindly? Leveraging the semantic power of large scale Contrastive-Language-Image-Pre-training (CLIP) models, we present a text-driven method that allows shifting a generative model to new domains, without having to collect even a single image from those domains. We show that through natural language prompts and a few minutes of training, our method can adapt a generator across a multitude of domains characterized by diverse styles and shapes. Notably, many of these modifications would be difficult or outright impossible to reach with existing methods. We conduct an extensive set of experiments and comparisons across a wide range of domains. These demonstrate the effectiveness of our approach and show that our shifted models maintain the latent-space properties that make generative models appealing for downstream tasks.

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Graphics (cs.GR); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2108.00946](https://arxiv.org/abs/2108.00946) [cs.CV]** |
|           | (or **[arXiv:2108.00946v1](https://arxiv.org/abs/2108.00946v1) [cs.CV]** for this version) |





<h2 id="2021-08-03-3">3. Structural Guidance for Transformer Language Models
</h2>

Title: [Structural Guidance for Transformer Language Models](https://arxiv.org/abs/2108.00104)

Authors: [Peng Qian](https://arxiv.org/search/cs?searchtype=author&query=Qian%2C+P), [Tahira Naseem](https://arxiv.org/search/cs?searchtype=author&query=Naseem%2C+T), [Roger Levy](https://arxiv.org/search/cs?searchtype=author&query=Levy%2C+R), [Ramón Fernandez Astudillo](https://arxiv.org/search/cs?searchtype=author&query=Astudillo%2C+R+F)

> Transformer-based language models pre-trained on large amounts of text data have proven remarkably successful in learning generic transferable linguistic representations. Here we study whether structural guidance leads to more human-like systematic linguistic generalization in Transformer language models without resorting to pre-training on very large amounts of data. We explore two general ideas. The "Generative Parsing" idea jointly models the incremental parse and word sequence as part of the same sequence modeling task. The "Structural Scaffold" idea guides the language model's representation via additional structure loss that separately predicts the incremental constituency parse. We train the proposed models along with a vanilla Transformer language model baseline on a 14 million-token and a 46 million-token subset of the BLLIP dataset, and evaluate models' syntactic generalization performances on SG Test Suites and sized BLiMP. Experiment results across two benchmarks suggest converging evidence that generative structural supervisions can induce more robust and humanlike linguistic generalization in Transformer language models without the need for data intensive pre-training.

| Comments: | To be issued as paper revision for ACL 2021                  |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2108.00104](https://arxiv.org/abs/2108.00104) [cs.CL]** |
|           | (or **[arXiv:2108.00104v1](https://arxiv.org/abs/2108.00104v1) [cs.CL]** for this version) |





<h2 id="2021-08-03-4">4. LICHEE: Improving Language Model Pre-training with Multi-grained Tokenization
</h2>

Title: [LICHEE: Improving Language Model Pre-training with Multi-grained Tokenization](https://arxiv.org/abs/2108.00801)

Authors: [Weidong Guo](https://arxiv.org/search/cs?searchtype=author&query=Guo%2C+W), [Mingjun Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+M), [Lusheng Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+L), [Di Niu](https://arxiv.org/search/cs?searchtype=author&query=Niu%2C+D), [Jinwen Luo](https://arxiv.org/search/cs?searchtype=author&query=Luo%2C+J), [Zhenhua Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Z), [Zhenyang Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Z), [Jianbo Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang%2C+J)

> Language model pre-training based on large corpora has achieved tremendous success in terms of constructing enriched contextual representations and has led to significant performance gains on a diverse range of Natural Language Understanding (NLU) tasks. Despite the success, most current pre-trained language models, such as BERT, are trained based on single-grained tokenization, usually with fine-grained characters or sub-words, making it hard for them to learn the precise meaning of coarse-grained words and phrases. In this paper, we propose a simple yet effective pre-training method named LICHEE to efficiently incorporate multi-grained information of input text. Our method can be applied to various pre-trained language models and improve their representation capability. Extensive experiments conducted on CLUE and SuperGLUE demonstrate that our method achieves comprehensive improvements on a wide variety of NLU tasks in both Chinese and English with little extra inference cost incurred, and that our best ensemble model achieves the state-of-the-art performance on CLUE benchmark competition.

| Comments: | Accepted by ACL Findings 2021                                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2108.00801](https://arxiv.org/abs/2108.00801) [cs.CL]** |
|           | (or **[arXiv:2108.00801v1](https://arxiv.org/abs/2108.00801v1) [cs.CL]** for this version) |







# 2021-08-02

[Return to Index](#Index)



<h2 id="2021-08-02-1">1. Difficulty-Aware Machine Translation Evaluation
</h2>

Title: [Difficulty-Aware Machine Translation Evaluation](https://arxiv.org/abs/2107.14402)

Authors: [Runzhe Zhan](https://arxiv.org/search/cs?searchtype=author&query=Zhan%2C+R), [Xuebo Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+X), [Derek F. Wong](https://arxiv.org/search/cs?searchtype=author&query=Wong%2C+D+F), [Lidia S. Chao](https://arxiv.org/search/cs?searchtype=author&query=Chao%2C+L+S)

> The high-quality translation results produced by machine translation (MT) systems still pose a huge challenge for automatic evaluation. Current MT evaluation pays the same attention to each sentence component, while the questions of real-world examinations (e.g., university examinations) have different difficulties and weightings. In this paper, we propose a novel difficulty-aware MT evaluation metric, expanding the evaluation dimension by taking translation difficulty into consideration. A translation that fails to be predicted by most MT systems will be treated as a difficult one and assigned a large weight in the final score function, and conversely. Experimental results on the WMT19 English-German Metrics shared tasks show that our proposed method outperforms commonly used MT metrics in terms of human correlation. In particular, our proposed method performs well even when all the MT systems are very competitive, which is when most existing metrics fail to distinguish between them. The source code is freely available at [this https URL](https://github.com/NLP2CT/Difficulty-Aware-MT-Evaluation).

| Comments: | Accepted to ACL 2021                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2107.14402](https://arxiv.org/abs/2107.14402) [cs.CL]** |
|           | (or **[arXiv:2107.14402v1](https://arxiv.org/abs/2107.14402v1) [cs.CL]** for this version) |





<h2 id="2021-08-02-2">2. Residual Tree Aggregation of Layers for Neural Machine Translation
</h2>

Title: [Residual Tree Aggregation of Layers for Neural Machine Translation](https://arxiv.org/abs/2107.14590)

Authors: [GuoLiang Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+G), [Yiyang Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Y)

> Although attention-based Neural Machine Translation has achieved remarkable progress in recent layers, it still suffers from issue of making insufficient use of the output of each layer. In transformer, it only uses the top layer of encoder and decoder in the subsequent process, which makes it impossible to take advantage of the useful information in other layers. To address this issue, we propose a residual tree aggregation of layers for Transformer(RTAL), which helps to fuse information across layers. Specifically, we try to fuse the information across layers by constructing a post-order binary tree. In additional to the last node, we add the residual connection to the process of generating child nodes. Our model is based on the Neural Machine Translation model Transformer and we conduct our experiments on WMT14 English-to-German and WMT17 English-to-France translation tasks. Experimental results across language pairs show that the proposed approach outperforms the strong baseline model significantly

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2107.14590](https://arxiv.org/abs/2107.14590) [cs.CL]** |
|           | (or **[arXiv:2107.14590v1](https://arxiv.org/abs/2107.14590v1) [cs.CL]** for this version) |





<h2 id="2021-08-02-3">3. Neural Variational Learning for Grounded Language Acquisition
</h2>

Title: [Neural Variational Learning for Grounded Language Acquisition](https://arxiv.org/abs/2107.14593)

Authors: [Nisha Pillai](https://arxiv.org/search/cs?searchtype=author&query=Pillai%2C+N), [Cynthia Matuszek](https://arxiv.org/search/cs?searchtype=author&query=Matuszek%2C+C), [Francis Ferraro](https://arxiv.org/search/cs?searchtype=author&query=Ferraro%2C+F)

> We propose a learning system in which language is grounded in visual percepts without specific pre-defined categories of terms. We present a unified generative method to acquire a shared semantic/visual embedding that enables the learning of language about a wide range of real-world objects. We evaluate the efficacy of this learning by predicting the semantics of objects and comparing the performance with neural and non-neural inputs. We show that this generative approach exhibits promising results in language grounding without pre-specifying visual categories under low resource settings. Our experiments demonstrate that this approach is generalizable to multilingual, highly varied datasets.

| Subjects:          | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Robotics (cs.RO) |
| ------------------ | ------------------------------------------------------------ |
| Journal reference: | 2021 30th IEEE International Conference on Robot and Human Interactive Communication (RO-MAN) |
| Cite as:           | **[arXiv:2107.14593](https://arxiv.org/abs/2107.14593) [cs.CL]** |
|                    | (or **[arXiv:2107.14593v1](https://arxiv.org/abs/2107.14593v1) [cs.CL]** for this version) |





<h2 id="2021-08-02-4">4. Multi-stage Pre-training over Simplified Multimodal Pre-training Models
</h2>

Title: [Multi-stage Pre-training over Simplified Multimodal Pre-training Models](https://arxiv.org/abs/2107.14596)

Authors: [Tongtong Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+T), [Fangxiang Feng](https://arxiv.org/search/cs?searchtype=author&query=Feng%2C+F), [Xiaojie Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X)

> Multimodal pre-training models, such as LXMERT, have achieved excellent results in downstream tasks. However, current pre-trained models require large amounts of training data and have huge model sizes, which make them difficult to apply in low-resource situations. How to obtain similar or even better performance than a larger model under the premise of less pre-training data and smaller model size has become an important problem. In this paper, we propose a new Multi-stage Pre-training (MSP) method, which uses information at different granularities from word, phrase to sentence in both texts and images to pre-train the model in stages. We also design several different pre-training tasks suitable for the information granularity in different stage in order to efficiently capture the diverse knowledge from a limited corpus. We take a Simplified LXMERT (LXMERT- S), which has only 45.9% parameters of the original LXMERT model and 11.76% of the original pre-training data as the testbed of our MSP method. Experimental results show that our method achieves comparable performance to the original LXMERT model in all downstream tasks, and even outperforms the original model in Image-Text Retrieval task.

| Comments: | 10 pages, 4 figures                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2107.14596](https://arxiv.org/abs/2107.14596) [cs.CL]** |
|           | (or **[arXiv:2107.14596v1](https://arxiv.org/abs/2107.14596v1) [cs.CL]** for this version) |





<h2 id="2021-08-02-5">5. MDQE: A More Accurate Direct Pretraining for Machine Translation Quality Estimation
</h2>

Title: [MDQE: A More Accurate Direct Pretraining for Machine Translation Quality Estimation](https://arxiv.org/abs/2107.14600)

Authors: [Lei Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+L)

> It is expensive to evaluate the results of Machine Translation(MT), which usually requires manual translation as a reference. Machine Translation Quality Estimation (QE) is a task of predicting the quality of machine translations without relying on any reference. Recently, the emergence of predictor-estimator framework which trains the predictor as a feature extractor and estimator as a QE predictor, and pre-trained language models(PLM) have achieved promising QE performance. However, we argue that there are still gaps between the predictor and the estimator in both data quality and training objectives, which preclude QE models from benefiting from a large number of parallel corpora more directly. Based on previous related work that have alleviated gaps to some extent, we propose a novel framework that provides a more accurate direct pretraining for QE tasks. In this framework, a generator is trained to produce pseudo data that is closer to the real QE data, and a estimator is pretrained on these data with novel objectives that are the same as the QE task. Experiments on widely used benchmarks show that our proposed framework outperforms existing methods, without using any pretraining models such as BERT.

| Comments: | arXiv admin note: substantial text overlap with [arXiv:2105.07149](https://arxiv.org/abs/2105.07149) by other authors |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2107.14600](https://arxiv.org/abs/2107.14600) [cs.CL]** |
|           | (or **[arXiv:2107.14600v1](https://arxiv.org/abs/2107.14600v1) [cs.CL]** for this version) |





<h2 id="2021-08-02-6">6. Towards Universality in Multilingual Text Rewriting
</h2>

Title: [Towards Universality in Multilingual Text Rewriting](https://arxiv.org/abs/2107.14749)

Authors: [Xavier Garcia](https://arxiv.org/search/cs?searchtype=author&query=Garcia%2C+X), [Noah Constant](https://arxiv.org/search/cs?searchtype=author&query=Constant%2C+N), [Mandy Guo](https://arxiv.org/search/cs?searchtype=author&query=Guo%2C+M), [Orhan Firat](https://arxiv.org/search/cs?searchtype=author&query=Firat%2C+O)

> In this work, we take the first steps towards building a universal rewriter: a model capable of rewriting text in any language to exhibit a wide variety of attributes, including styles and languages, while preserving as much of the original semantics as possible. In addition to obtaining state-of-the-art results on unsupervised translation, we also demonstrate the ability to do zero-shot sentiment transfer in non-English languages using only English exemplars for sentiment. We then show that our model is able to modify multiple attributes at once, for example adjusting both language and sentiment jointly. Finally, we show that our model is capable of performing zero-shot formality-sensitive translation.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2107.14749](https://arxiv.org/abs/2107.14749) [cs.CL]** |
|           | (or **[arXiv:2107.14749v1](https://arxiv.org/abs/2107.14749v1) [cs.CL]** for this version) |





<h2 id="2021-08-02-7">7. ChrEnTranslate: Cherokee-English Machine Translation Demo with Quality Estimation and Corrective Feedback
</h2>

Title: [ChrEnTranslate: Cherokee-English Machine Translation Demo with Quality Estimation and Corrective Feedback](https://arxiv.org/abs/2107.14800)

Authors: [Shiyue Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+S), [Benjamin Frey](https://arxiv.org/search/cs?searchtype=author&query=Frey%2C+B), [Mohit Bansal](https://arxiv.org/search/cs?searchtype=author&query=Bansal%2C+M)

> We introduce ChrEnTranslate, an online machine translation demonstration system for translation between English and an endangered language Cherokee. It supports both statistical and neural translation models as well as provides quality estimation to inform users of reliability, two user feedback interfaces for experts and common users respectively, example inputs to collect human translations for monolingual data, word alignment visualization, and relevant terms from the Cherokee-English dictionary. The quantitative evaluation demonstrates that our backbone translation models achieve state-of-the-art translation performance and our quality estimation well correlates with both BLEU and human judgment. By analyzing 216 pieces of expert feedback, we find that NMT is preferable because it copies less than SMT, and, in general, current models can translate fragments of the source sentence but make major mistakes. When we add these 216 expert-corrected parallel texts into the training set and retrain models, equal or slightly better performance is observed, which demonstrates indicates the potential of human-in-the-loop learning. Our online demo is at [this https URL](https://chren.cs.unc.edu/;) our code is open-sourced at [this https URL](https://github.com/ZhangShiyue/ChrEnTranslate;) and our data is available at [this https URL](https://github.com/ZhangShiyue/ChrEn).

| Comments: | ACL 2021 Demo (8 pages)                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2107.14800](https://arxiv.org/abs/2107.14800) [cs.CL]** |
|           | (or **[arXiv:2107.14800v1](https://arxiv.org/abs/2107.14800v1) [cs.CL]** for this version) |



