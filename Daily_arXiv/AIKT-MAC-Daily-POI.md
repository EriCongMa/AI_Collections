# MA C.'s Daily Paper Of Interest - April, 2022

# Index

- [2022-04-08](#2022-04-08)
  - [1. Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality](#2022-04-08-1)
  
  - [2. Fusing finetuned models for better pretraining](#2022-04-08-2)
  
  - [3. Knowledge Infused Decoding](#2022-04-08-3)
  
  - [4. MAESTRO: Matched Speech Text Representations through Modality Matching](#2022-04-08-4)
  
  - [5. A Survey of Multi-task Learning in Natural Language Processing: Regarding Task Relatedness and Training Methods](#2022-04-08-5)
  
- [2022-04-07](#2022-04-07)
  - [1. Combining Spectral and Self-Supervised Features for Low Resource Speech Recognition and Translation](#2022-04-07-1)

  - [2. Probing Structured Pruning on Multilingual Pre-trained Models: Settings, Algorithms, and Efficiency](#2022-04-07-2)

  - [3. EMMT: A simultaneous eye-tracking, 4-electrode EEG and audio corpus for multi-modal reading and translation scenarios](#2022-04-07-3)

  - [4. Paying More Attention to Self-attention: Improving Pre-trained Language Models via Attention Guiding](#2022-04-07-4)

  - [5. Enhanced Direct Speech-to-Speech Translation Using Self-supervised Pre-training and Data Augmentation](#2022-04-07-5)

- [2022-04-06](#2022-04-06)
  - [1. Multi-View Transformer for 3D Visual Grounding](#2022-04-06-1)

  - [2. latent-GLAT: Glancing at Latent Variables for Parallel Text Generation](#2022-04-06-2)

  - [3. PaLM: Scaling Language Modeling with Pathways](#2022-04-06-3)

- [2022-04-05](#2022-04-05)
  - [1. Moment-based Adversarial Training for Embodied Language Comprehension](#2022-04-05-1)

  - [2. Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](#2022-04-05-2)

  - [3. CipherDAug: Ciphertext based Data Augmentation for Neural Machine Translation](#2022-04-05-3)

  - [4. On Efficiently Acquiring Annotations for Multilingual Models](#2022-04-05-4)

  - [5. PERFECT: Prompt-free and Efficient Few-shot Learning with Language Models](#2022-04-05-5)

  - [6. Aligned Weight Regularizers for Pruning Pretrained Neural Networks](#2022-04-05-6)

  - [7. Estimating the Entropy of Linguistic Distributions](#2022-04-05-7)

- [2022-04-04](#2022-04-04)
  - [1. AdaSpeech 4: Adaptive Text to Speech in Zero-Shot Scenarios](#2022-04-04-1)

  - [2. Unified and Effective Ensemble Knowledge Distillation](#2022-04-04-2)

  - [3. Better Intermediates Improve CTC Inference](#2022-04-04-3)

  - [4. WavFT: Acoustic model finetuning with labelled and unlabelled data](#2022-04-04-4)

  - [5. Uncertainty Determines the Adequacy of the Mode and the Tractability of Decoding in Sequence-to-Sequence Models](#2022-04-04-5)

- [2022-04-01](#2022-04-01)
  - [1. MAE-AST: Masked Autoencoding Audio Spectrogram Transformer](#2022-04-01-1)
  - [2. Exploiting Single-Channel Speech for Multi-Channel End-to-End Speech Recognition: A Comparative Study](#2022-04-01-2)
  - [3. How Does Pre-trained Wav2Vec2.0 Perform on Domain Shifted ASR? An Extensive Benchmark on Air Traffic Control Communications](#2022-04-01-3)
  - [4. Interpretation of Black Box NLP Models: A Survey](#2022-04-01-4)
  - [5. Scaling Up Models and Data with 𝚝𝟻𝚡 and ](#2022-04-01-5)
  - [6. Mixed-Phoneme BERT: Improving BERT with Mixed Phoneme and Sup-Phoneme Representations for Text to Speech](#2022-04-01-6)
  - [7. VL-InterpreT: An Interactive Visualization Tool for Interpreting Vision-Language Transformers](#2022-04-01-7)
  - [8. Is Word Error Rate a good evaluation metric for Speech Recognition in Indic Languages?](#2022-04-01-8)
  - [9. PADA: Pruning Assisted Domain Adaptation for Self-Supervised Speech Representations](#2022-04-01-9)
  - [10. Analyzing the factors affecting usefulness of Self-Supervised Pre-trained Representations for Speech Recognition](#2022-04-01-10)
  - [11. PANGUBOT: Efficient Generative Dialogue Pre-training from Pre-trained Language Model](#2022-04-01-11)

- [2022-03-31](#2022-03-31)
  - [1. WAVPROMPT: Towards Few-Shot Spoken Language Understanding with Frozen Language Models](#2022-03-31-1)
- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MAC-Daily-POI-index.md)



# 2022-04-08

[Return to Index](#Index)



<h2 id="2022-04-08-1">1. Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality
</h2>

Title: [Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality](https://arxiv.org/abs/2204.03162)

Authors:[Tristan Thrush](https://arxiv.org/search/cs?searchtype=author&query=Thrush%2C+T), [Ryan Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+R), [Max Bartolo](https://arxiv.org/search/cs?searchtype=author&query=Bartolo%2C+M), [Amanpreet Singh](https://arxiv.org/search/cs?searchtype=author&query=Singh%2C+A), [Adina Williams](https://arxiv.org/search/cs?searchtype=author&query=Williams%2C+A), [Douwe Kiela](https://arxiv.org/search/cs?searchtype=author&query=Kiela%2C+D), [Candace Ross](https://arxiv.org/search/cs?searchtype=author&query=Ross%2C+C)

> We present a novel task and dataset for evaluating the ability of vision and language models to conduct visio-linguistic compositional reasoning, which we call Winoground. Given two images and two captions, the goal is to match them correctly - but crucially, both captions contain a completely identical set of words, only in a different order. The dataset was carefully hand-curated by expert annotators and is labeled with a rich set of fine-grained tags to assist in analyzing model performance. We probe a diverse range of state-of-the-art vision and language models and find that, surprisingly, none of them do much better than chance. Evidently, these models are not as skilled at visio-linguistic compositional reasoning as we might have hoped. We perform an extensive analysis to obtain insights into how future work might try to mitigate these models' shortcomings. We aim for Winoground to serve as a useful evaluation set for advancing the state of the art and driving further progress in the field. The dataset is available at [this https URL](https://huggingface.co/datasets/facebook/winoground).

| Comments: | CVPR 2022                                                    |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2204.03162](https://arxiv.org/abs/2204.03162) [cs.CV]** |
|           | (or **[arXiv:2204.03162v1](https://arxiv.org/abs/2204.03162v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.03162Focus to learn more |





<h2 id="2022-04-08-2">2. Fusing finetuned models for better pretraining
</h2>

Title: [Fusing finetuned models for better pretraining](https://arxiv.org/abs/2204.03044)

Authors:[Leshem Choshen](https://arxiv.org/search/cs?searchtype=author&query=Choshen%2C+L), [Elad Venezian](https://arxiv.org/search/cs?searchtype=author&query=Venezian%2C+E), [Noam Slonim](https://arxiv.org/search/cs?searchtype=author&query=Slonim%2C+N), [Yoav Katz](https://arxiv.org/search/cs?searchtype=author&query=Katz%2C+Y)

> Pretrained models are the standard starting point for training. This approach consistently outperforms the use of a random initialization. However, pretraining is a costly endeavour that few can undertake. 
> In this paper, we create better base models at hardly any cost, by fusing multiple existing fine tuned models into one. Specifically, we fuse by averaging the weights of these models. We show that the fused model results surpass the pretrained model ones. We also show that fusing is often better than intertraining. 
> We find that fusing is less dependent on the target task. Furthermore, weight decay nullifies intertraining effects but not those of fusing.

| Subjects: | **Computation and Language (cs.CL)**; Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2204.03044](https://arxiv.org/abs/2204.03044) [cs.CL]** |
|           | (or **[arXiv:2204.03044v1](https://arxiv.org/abs/2204.03044v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.03044Focus to learn more |





<h2 id="2022-04-08-3">3. Knowledge Infused Decoding
</h2>

Title: [Knowledge Infused Decoding](https://arxiv.org/abs/2204.03084)

Authors:[Ruibo Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+R), [Guoqing Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+G), [Shashank Gupta](https://arxiv.org/search/cs?searchtype=author&query=Gupta%2C+S), [Radhika Gaonkar](https://arxiv.org/search/cs?searchtype=author&query=Gaonkar%2C+R), [Chongyang Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+C), [Soroush Vosoughi](https://arxiv.org/search/cs?searchtype=author&query=Vosoughi%2C+S), [Milad Shokouhi](https://arxiv.org/search/cs?searchtype=author&query=Shokouhi%2C+M), [Ahmed Hassan Awadallah](https://arxiv.org/search/cs?searchtype=author&query=Awadallah%2C+A+H)

> Pre-trained language models (LMs) have been shown to memorize a substantial amount of knowledge from the pre-training corpora; however, they are still limited in recalling factually correct knowledge given a certain context. Hence, they tend to suffer from counterfactual or hallucinatory generation when used in knowledge-intensive natural language generation (NLG) tasks. Recent remedies to this problem focus on modifying either the pre-training or task fine-tuning objectives to incorporate knowledge, which normally require additional costly training or architecture modification of LMs for practical applications. We present Knowledge Infused Decoding (KID) -- a novel decoding algorithm for generative LMs, which dynamically infuses external knowledge into each step of the LM decoding. Specifically, we maintain a local knowledge memory based on the current context, interacting with a dynamically created external knowledge trie, and continuously update the local memory as a knowledge-aware constraint to guide decoding via reinforcement learning. On six diverse knowledge-intensive NLG tasks, task-agnostic LMs (e.g., GPT-2 and BART) armed with KID outperform many task-optimized state-of-the-art models, and show particularly strong performance in few-shot scenarios over seven related knowledge-infusion techniques. Human evaluation confirms KID's ability to generate more relevant and factual language for the input context when compared with multiple baselines. Finally, KID also alleviates exposure bias and provides stable generation quality when generating longer sequences. Code for KID is available at [this https URL](https://github.com/microsoft/KID).

| Comments: | In ICLR 2022                                                 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2204.03084](https://arxiv.org/abs/2204.03084) [cs.CL]** |
|           | (or **[arXiv:2204.03084v1](https://arxiv.org/abs/2204.03084v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.03084Focus to learn more |





<h2 id="2022-04-08-4">4. MAESTRO: Matched Speech Text Representations through Modality Matching
</h2>

Title: [MAESTRO: Matched Speech Text Representations through Modality Matching](https://arxiv.org/abs/2204.03409)

Authors:[Zhehuai Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Z), [Yu Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Y), [Andrew Rosenberg](https://arxiv.org/search/cs?searchtype=author&query=Rosenberg%2C+A), [Bhuvana Ramabhadran](https://arxiv.org/search/cs?searchtype=author&query=Ramabhadran%2C+B), [Pedro Moreno](https://arxiv.org/search/cs?searchtype=author&query=Moreno%2C+P), [Ankur Bapna](https://arxiv.org/search/cs?searchtype=author&query=Bapna%2C+A), [Heiga Zen](https://arxiv.org/search/cs?searchtype=author&query=Zen%2C+H)

> We present Maestro, a self-supervised training method to unify representations learnt from speech and text modalities. Self-supervised learning from speech signals aims to learn the latent structure inherent in the signal, while self-supervised learning from text attempts to capture lexical information. Learning aligned representations from unpaired speech and text sequences is a challenging task. Previous work either implicitly enforced the representations learnt from these two modalities to be aligned in the latent space through multitasking and parameter sharing or explicitly through conversion of modalities via speech synthesis. While the former suffers from interference between the two modalities, the latter introduces additional complexity. In this paper, we propose Maestro, a novel algorithm to learn unified representations from both these modalities simultaneously that can transfer to diverse downstream tasks such as Automated Speech Recognition (ASR) and Speech Translation (ST). Maestro learns unified representations through sequence alignment, duration prediction and matching embeddings in the learned space through an aligned masked-language model loss. We establish a new state-of-the-art (SOTA) on VoxPopuli multilingual ASR with a 11% relative reduction in Word Error Rate (WER), multidomain SpeechStew ASR (3.7% relative) and 21 languages to English multilingual ST on CoVoST 2 with an improvement of 2.8 BLEU averaged over 21 languages.

| Comments:    | Submitted to Interspeech 2022                                |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| MSC classes: | 68T10                                                        |
| ACM classes: | I.2.7                                                        |
| Cite as:     | **[arXiv:2204.03409](https://arxiv.org/abs/2204.03409) [cs.CL]** |
|              | (or **[arXiv:2204.03409v1](https://arxiv.org/abs/2204.03409v1) [cs.CL]** for this version) |
|              | https://doi.org/10.48550/arXiv.2204.03409Focus to learn more |





<h2 id="2022-04-08-5">5. A Survey of Multi-task Learning in Natural Language Processing: Regarding Task Relatedness and Training Methods
</h2>

Title: [A Survey of Multi-task Learning in Natural Language Processing: Regarding Task Relatedness and Training Methods](https://arxiv.org/abs/2204.03508)

Authors:[Zhihan Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Z), [Wenhao Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+W), [Mengxia Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+M), [Zhichun Guo](https://arxiv.org/search/cs?searchtype=author&query=Guo%2C+Z), [Meng Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+M)

> Multi-task learning (MTL) has become increasingly popular in natural language processing (NLP) because it improves the performance of related tasks by exploiting their commonalities and differences. Nevertheless, it is still not understood very well how multi-task learning can be implemented based on the relatedness of training tasks. In this survey, we review recent advances of multi-task learning methods in NLP, with the aim of summarizing them into two general multi-task training methods based on their task relatedness: (i) joint training and (ii) multi-step training. We present examples in various NLP downstream applications, summarize the task relationships and discuss future directions of this promising topic.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2204.03508](https://arxiv.org/abs/2204.03508) [cs.CL]** |
|           | (or **[arXiv:2204.03508v1](https://arxiv.org/abs/2204.03508v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.03508 Focus to learn morearXiv-issued DOI via DataCite |







# 2022-04-07

[Return to Index](#Index)



<h2 id="2022-04-07-1">1. Combining Spectral and Self-Supervised Features for Low Resource Speech Recognition and Translation
</h2>

Title: [Combining Spectral and Self-Supervised Features for Low Resource Speech Recognition and Translation](https://arxiv.org/abs/2204.02470)

Authors: [Dan Berrebbi](https://arxiv.org/search/cs?searchtype=author&query=Berrebbi%2C+D), [Jiatong Shi](https://arxiv.org/search/cs?searchtype=author&query=Shi%2C+J), [Brian Yan](https://arxiv.org/search/cs?searchtype=author&query=Yan%2C+B), [Osbel Lopez-Francisco](https://arxiv.org/search/cs?searchtype=author&query=Lopez-Francisco%2C+O), [Jonathan D. Amith](https://arxiv.org/search/cs?searchtype=author&query=Amith%2C+J+D), [Shinji Watanabe](https://arxiv.org/search/cs?searchtype=author&query=Watanabe%2C+S)

> Self-Supervised Learning (SSL) models have been successfully applied in various deep learning-based speech tasks, particularly those with a limited amount of data. However, the quality of SSL representations depends highly on the relatedness between the SSL training domain(s) and the target data domain. On the contrary, spectral feature (SF) extractors such as log Mel-filterbanks are hand-crafted non-learnable components, and could be more robust to domain shifts. The present work examines the assumption that combining non-learnable SF extractors to SSL models is an effective approach to low resource speech tasks. We propose a learnable and interpretable framework to combine SF and SSL representations. The proposed framework outperforms significantly both baseline and SSL models on Automatic Speech Recognition (ASR) and Speech Translation (ST) tasks on three low resource datasets. We additionally design a mixture of experts based combination model. This last model reveals that the relative contribution of SSL models over conventional SF extractors is very small in case of domain mismatch between SSL training set and the target language data.

| Comments: | 5 pages, 2 figures, submitted to Interspeech 2022            |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2204.02470](https://arxiv.org/abs/2204.02470) [cs.CL]** |
|           | (or **[arXiv:2204.02470v1](https://arxiv.org/abs/2204.02470v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.02470Focus to learn more |





<h2 id="2022-04-07-2">2. Probing Structured Pruning on Multilingual Pre-trained Models: Settings, Algorithms, and Efficiency
</h2>

Title: [Probing Structured Pruning on Multilingual Pre-trained Models: Settings, Algorithms, and Efficiency](https://arxiv.org/abs/2204.02601)

Authors: [Yanyang Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Y), [Fuli Luo](https://arxiv.org/search/cs?searchtype=author&query=Luo%2C+F), [Runxin Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+R), [Songfang Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+S), [Fei Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+F), [Liwei Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+L)

> Structured pruning has been extensively studied on monolingual pre-trained language models and is yet to be fully evaluated on their multilingual counterparts. This work investigates three aspects of structured pruning on multilingual pre-trained language models: settings, algorithms, and efficiency. Experiments on nine downstream tasks show several counter-intuitive phenomena: for settings, individually pruning for each language does not induce a better result; for algorithms, the simplest method performs the best; for efficiency, a fast model does not imply that it is also small. To facilitate the comparison on all sparsity levels, we present Dynamic Sparsification, a simple approach that allows training the model once and adapting to different model sizes at inference. We hope this work fills the gap in the study of structured pruning on multilingual pre-trained models and sheds light on future research.

| Comments: | ACL 2022 Main Conference, Camera-ready version               |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2204.02601](https://arxiv.org/abs/2204.02601) [cs.CL]** |
|           | (or **[arXiv:2204.02601v1](https://arxiv.org/abs/2204.02601v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.02601Focus to learn more |





<h2 id="2022-04-07-3">3. EMMT: A simultaneous eye-tracking, 4-electrode EEG and audio corpus for multi-modal reading and translation scenarios
</h2>

Title: [EMMT: A simultaneous eye-tracking, 4-electrode EEG and audio corpus for multi-modal reading and translation scenarios](https://arxiv.org/abs/2204.02905)

Authors: [Sunit Bhattacharya](https://arxiv.org/search/cs?searchtype=author&query=Bhattacharya%2C+S), [Věra Kloudová](https://arxiv.org/search/cs?searchtype=author&query=Kloudová%2C+V), [Vilém Zouhar](https://arxiv.org/search/cs?searchtype=author&query=Zouhar%2C+V), [Ondřej Bojar](https://arxiv.org/search/cs?searchtype=author&query=Bojar%2C+O)

> We present the Eyetracked Multi-Modal Translation (EMMT) corpus, a dataset containing monocular eye movement recordings, audio and 4-electrode electroencephalogram (EEG) data of 43 participants. The objective was to collect cognitive signals as responses of participants engaged in a number of language intensive tasks involving different text-image stimuli settings when translating from English to Czech. 
> Each participant was exposed to 32 text-image stimuli pairs and asked to (1) read the English sentence, (2) translate it into Czech, (3) consult the image, (4) translate again, either updating or repeating the previous translation. The text stimuli consisted of 200 unique sentences with 616 unique words coupled with 200 unique images as the visual stimuli. 
> The recordings were collected over a two week period and all the participants included in the study were Czech natives with strong English skills. Due to the nature of the tasks involved in the study and the relatively large number of participants involved, the corpus is well suited for research in Translation Process Studies, Cognitive Sciences among other disciplines.

| Comments: | Submitted to Nature Scientific Data                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Human-Computer Interaction (cs.HC) |
| Cite as:  | **[arXiv:2204.02905](https://arxiv.org/abs/2204.02905) [cs.CL]** |
|           | (or **[arXiv:2204.02905v1](https://arxiv.org/abs/2204.02905v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.02905Focus to learn more |





<h2 id="2022-04-07-4">4. Paying More Attention to Self-attention: Improving Pre-trained Language Models via Attention Guiding
</h2>

Title: [Paying More Attention to Self-attention: Improving Pre-trained Language Models via Attention Guiding](https://arxiv.org/abs/2204.02922)

Authors: [Shanshan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+S), [Zhumin Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Z), [Zhaochun Ren](https://arxiv.org/search/cs?searchtype=author&query=Ren%2C+Z), [Huasheng Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+H), [Qiang Yan](https://arxiv.org/search/cs?searchtype=author&query=Yan%2C+Q), [Pengjie Ren](https://arxiv.org/search/cs?searchtype=author&query=Ren%2C+P)

> Pre-trained language models (PLM) have demonstrated their effectiveness for a broad range of information retrieval and natural language processing tasks. As the core part of PLM, multi-head self-attention is appealing for its ability to jointly attend to information from different positions. However, researchers have found that PLM always exhibits fixed attention patterns regardless of the input (e.g., excessively paying attention to [CLS] or [SEP]), which we argue might neglect important information in the other positions. In this work, we propose a simple yet effective attention guiding mechanism to improve the performance of PLM by encouraging attention towards the established goals. Specifically, we propose two kinds of attention guiding methods, i.e., map discrimination guiding (MDG) and attention pattern decorrelation guiding (PDG). The former definitely encourages the diversity among multiple self-attention heads to jointly attend to information from different representation subspaces, while the latter encourages self-attention to attend to as many different positions of the input as possible. We conduct experiments with multiple general pre-trained models (i.e., BERT, ALBERT, and Roberta) and domain-specific pre-trained models (i.e., BioBERT, ClinicalBERT, BlueBert, and SciBERT) on three benchmark datasets (i.e., MultiNLI, MedNLI, and Cross-genre-IR). Extensive experimental results demonstrate that our proposed MDG and PDG bring stable performance improvements on all datasets with high efficiency and low cost.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2204.02922](https://arxiv.org/abs/2204.02922) [cs.CL]** |
|           | (or **[arXiv:2204.02922v1](https://arxiv.org/abs/2204.02922v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.02922Focus to learn more |





<h2 id="2022-04-07-5">5. Enhanced Direct Speech-to-Speech Translation Using Self-supervised Pre-training and Data Augmentation
</h2>

Title: [Enhanced Direct Speech-to-Speech Translation Using Self-supervised Pre-training and Data Augmentation](https://arxiv.org/abs/2204.02967)

Authors: [Sravya Popuri](https://arxiv.org/search/cs?searchtype=author&query=Popuri%2C+S), [Peng-Jen Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+P), [Changhan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+C), [Juan Pino](https://arxiv.org/search/cs?searchtype=author&query=Pino%2C+J), [Yossi Adi](https://arxiv.org/search/cs?searchtype=author&query=Adi%2C+Y), [Jiatao Gu](https://arxiv.org/search/cs?searchtype=author&query=Gu%2C+J), [Wei-Ning Hsu](https://arxiv.org/search/cs?searchtype=author&query=Hsu%2C+W), [Ann Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+A)

> Direct speech-to-speech translation (S2ST) models suffer from data scarcity issues as there exists little parallel S2ST data, compared to the amount of data available for conventional cascaded systems that consist of automatic speech recognition (ASR), machine translation (MT), and text-to-speech (TTS) synthesis. In this work, we explore self-supervised pre-training with unlabeled speech data and data augmentation to tackle this issue. We take advantage of a recently proposed speech-to-unit translation (S2UT) framework that encodes target speech into discrete representations, and transfer pre-training and efficient partial finetuning techniques that work well for speech-to-text translation (S2T) to the S2UT domain by studying both speech encoder and discrete unit decoder pre-training. Our experiments show that self-supervised pre-training consistently improves model performance compared with multitask learning with a BLEU gain of 4.3-12.0 under various data setups, and it can be further combined with data augmentation techniques that apply MT to create weakly supervised training data. Audio samples are available at: [this https URL](https://facebookresearch.github.io/speech_translation/enhanced_direct_s2st_units/index.html) .

| Comments: | Submitted to Interspeech 2022                                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2204.02967](https://arxiv.org/abs/2204.02967) [cs.CL]** |
|           | (or **[arXiv:2204.02967v1](https://arxiv.org/abs/2204.02967v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.02967Focus to learn more |





# 2022-04-06

[Return to Index](#Index)



<h2 id="2022-04-06-1">1. Multi-View Transformer for 3D Visual Grounding
</h2>

Title: [Multi-View Transformer for 3D Visual Grounding](https://arxiv.org/abs/2204.02174)

Authors: [Shijia Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+S), [Yilun Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Y), [Jiaya Jia](https://arxiv.org/search/cs?searchtype=author&query=Jia%2C+J), [Liwei Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+L)

> The 3D visual grounding task aims to ground a natural language description to the targeted object in a 3D scene, which is usually represented in 3D point clouds. Previous works studied visual grounding under specific views. The vision-language correspondence learned by this way can easily fail once the view changes. In this paper, we propose a Multi-View Transformer (MVT) for 3D visual grounding. We project the 3D scene to a multi-view space, in which the position information of the 3D scene under different views are modeled simultaneously and aggregated together. The multi-view space enables the network to learn a more robust multi-modal representation for 3D visual grounding and eliminates the dependence on specific views. Extensive experiments show that our approach significantly outperforms all state-of-the-art methods. Specifically, on Nr3D and Sr3D datasets, our method outperforms the best competitor by 11.2% and 7.1% and even surpasses recent work with extra 2D assistance by 5.9% and 6.6%. Our code is available at [this https URL](https://github.com/sega-hsj/MVT-3DVG).

| Comments: | cvpr2022                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2204.02174](https://arxiv.org/abs/2204.02174) [cs.CV]** |
|           | (or **[arXiv:2204.02174v1](https://arxiv.org/abs/2204.02174v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.02174Focus to learn more |





<h2 id="2022-04-06-2">2. latent-GLAT: Glancing at Latent Variables for Parallel Text Generation
</h2>

Title: [latent-GLAT: Glancing at Latent Variables for Parallel Text Generation](https://arxiv.org/abs/2204.02030)

Authors: [Yu Bao](https://arxiv.org/search/cs?searchtype=author&query=Bao%2C+Y), [Hao Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+H), [Shujian Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+S), [Dongqi Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+D), [Lihua Qian](https://arxiv.org/search/cs?searchtype=author&query=Qian%2C+L), [Xinyu Dai](https://arxiv.org/search/cs?searchtype=author&query=Dai%2C+X), [Jiajun Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+J), [Lei Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+L)

> Recently, parallel text generation has received widespread attention due to its success in generation efficiency. Although many advanced techniques are proposed to improve its generation quality, they still need the help of an autoregressive model for training to overcome the one-to-many multi-modal phenomenon in the dataset, limiting their applications. In this paper, we propose latent-GLAT, which employs the discrete latent variables to capture word categorical information and invoke an advanced curriculum learning technique, alleviating the multi-modality problem. Experiment results show that our method outperforms strong baselines without the help of an autoregressive model, which further broadens the application scenarios of the parallel decoding paradigm.

| Comments: | 12 pages, 5 figures, 6 tables. Accepted as a long paper in the main conference of ACL-2022 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2204.02030](https://arxiv.org/abs/2204.02030) [cs.CL]** |
|           | (or **[arXiv:2204.02030v1](https://arxiv.org/abs/2204.02030v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.02030Focus to learn more |





<h2 id="2022-04-06-3">3. PaLM: Scaling Language Modeling with Pathways
</h2>

Title: [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)

Authors: [Aakanksha Chowdhery](https://arxiv.org/search/cs?searchtype=author&query=Chowdhery%2C+A), [Sharan Narang](https://arxiv.org/search/cs?searchtype=author&query=Narang%2C+S), [Jacob Devlin](https://arxiv.org/search/cs?searchtype=author&query=Devlin%2C+J), [Maarten Bosma](https://arxiv.org/search/cs?searchtype=author&query=Bosma%2C+M), [Gaurav Mishra](https://arxiv.org/search/cs?searchtype=author&query=Mishra%2C+G), [Adam Roberts](https://arxiv.org/search/cs?searchtype=author&query=Roberts%2C+A), [Paul Barham](https://arxiv.org/search/cs?searchtype=author&query=Barham%2C+P), [Hyung Won Chung](https://arxiv.org/search/cs?searchtype=author&query=Chung%2C+H+W), [Charles Sutton](https://arxiv.org/search/cs?searchtype=author&query=Sutton%2C+C), [Sebastian Gehrmann](https://arxiv.org/search/cs?searchtype=author&query=Gehrmann%2C+S), [Parker Schuh](https://arxiv.org/search/cs?searchtype=author&query=Schuh%2C+P), [Kensen Shi](https://arxiv.org/search/cs?searchtype=author&query=Shi%2C+K), [Sasha Tsvyashchenko](https://arxiv.org/search/cs?searchtype=author&query=Tsvyashchenko%2C+S), [Joshua Maynez](https://arxiv.org/search/cs?searchtype=author&query=Maynez%2C+J), [Abhishek Rao](https://arxiv.org/search/cs?searchtype=author&query=Rao%2C+A), [Parker Barnes](https://arxiv.org/search/cs?searchtype=author&query=Barnes%2C+P), [Yi Tay](https://arxiv.org/search/cs?searchtype=author&query=Tay%2C+Y), [Noam Shazeer](https://arxiv.org/search/cs?searchtype=author&query=Shazeer%2C+N), [Vinodkumar Prabhakaran](https://arxiv.org/search/cs?searchtype=author&query=Prabhakaran%2C+V), [Emily Reif](https://arxiv.org/search/cs?searchtype=author&query=Reif%2C+E), [Nan Du](https://arxiv.org/search/cs?searchtype=author&query=Du%2C+N), [Ben Hutchinson](https://arxiv.org/search/cs?searchtype=author&query=Hutchinson%2C+B), [Reiner Pope](https://arxiv.org/search/cs?searchtype=author&query=Pope%2C+R), [James Bradbury](https://arxiv.org/search/cs?searchtype=author&query=Bradbury%2C+J), [Jacob Austin](https://arxiv.org/search/cs?searchtype=author&query=Austin%2C+J), [Michael Isard](https://arxiv.org/search/cs?searchtype=author&query=Isard%2C+M), [Guy Gur-Ari](https://arxiv.org/search/cs?searchtype=author&query=Gur-Ari%2C+G), [Pengcheng Yin](https://arxiv.org/search/cs?searchtype=author&query=Yin%2C+P), [Toju Duke](https://arxiv.org/search/cs?searchtype=author&query=Duke%2C+T), [Anselm Levskaya](https://arxiv.org/search/cs?searchtype=author&query=Levskaya%2C+A), [Sanjay Ghemawat](https://arxiv.org/search/cs?searchtype=author&query=Ghemawat%2C+S), [Sunipa Dev](https://arxiv.org/search/cs?searchtype=author&query=Dev%2C+S), [Henryk Michalewski](https://arxiv.org/search/cs?searchtype=author&query=Michalewski%2C+H), [Xavier Garcia](https://arxiv.org/search/cs?searchtype=author&query=Garcia%2C+X), [Vedant Misra](https://arxiv.org/search/cs?searchtype=author&query=Misra%2C+V), [Kevin Robinson](https://arxiv.org/search/cs?searchtype=author&query=Robinson%2C+K), [Liam Fedus](https://arxiv.org/search/cs?searchtype=author&query=Fedus%2C+L), [Denny Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+D), [Daphne Ippolito](https://arxiv.org/search/cs?searchtype=author&query=Ippolito%2C+D), [David Luan](https://arxiv.org/search/cs?searchtype=author&query=Luan%2C+D), [Hyeontaek Lim](https://arxiv.org/search/cs?searchtype=author&query=Lim%2C+H), [Barret Zoph](https://arxiv.org/search/cs?searchtype=author&query=Zoph%2C+B), [Alexander Spiridonov](https://arxiv.org/search/cs?searchtype=author&query=Spiridonov%2C+A), [Ryan Sepassi](https://arxiv.org/search/cs?searchtype=author&query=Sepassi%2C+R), [David Dohan](https://arxiv.org/search/cs?searchtype=author&query=Dohan%2C+D), [Shivani Agrawal](https://arxiv.org/search/cs?searchtype=author&query=Agrawal%2C+S), [Mark Omernick](https://arxiv.org/search/cs?searchtype=author&query=Omernick%2C+M), [Andrew M. Dai](https://arxiv.org/search/cs?searchtype=author&query=Dai%2C+A+M), [Thanumalayan Sankaranarayana Pillai](https://arxiv.org/search/cs?searchtype=author&query=Pillai%2C+T+S), [Marie Pellat](https://arxiv.org/search/cs?searchtype=author&query=Pellat%2C+M), [Aitor Lewkowycz](https://arxiv.org/search/cs?searchtype=author&query=Lewkowycz%2C+A), [Erica Moreira](https://arxiv.org/search/cs?searchtype=author&query=Moreira%2C+E), [Rewon Child](https://arxiv.org/search/cs?searchtype=author&query=Child%2C+R), [Oleksandr Polozov](https://arxiv.org/search/cs?searchtype=author&query=Polozov%2C+O), [Katherine Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+K), [Zongwei Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+Z), [Xuezhi Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Brennan Saeta](https://arxiv.org/search/cs?searchtype=author&query=Saeta%2C+B), [Mark Diaz](https://arxiv.org/search/cs?searchtype=author&query=Diaz%2C+M), [Orhan Firat](https://arxiv.org/search/cs?searchtype=author&query=Firat%2C+O), [Michele Catasta](https://arxiv.org/search/cs?searchtype=author&query=Catasta%2C+M), [Jason Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+J), [Kathy Meier-Hellstern](https://arxiv.org/search/cs?searchtype=author&query=Meier-Hellstern%2C+K), [Douglas Eck](https://arxiv.org/search/cs?searchtype=author&query=Eck%2C+D), [Jeff Dean](https://arxiv.org/search/cs?searchtype=author&query=Dean%2C+J), [Slav Petrov](https://arxiv.org/search/cs?searchtype=author&query=Petrov%2C+S), [Noah Fiedel](https://arxiv.org/search/cs?searchtype=author&query=Fiedel%2C+N)

> Large language models have been shown to achieve remarkable performance across a variety of natural language tasks using few-shot learning, which drastically reduces the number of task-specific training examples needed to adapt the model to a particular application. To further our understanding of the impact of scale on few-shot learning, we trained a 540-billion parameter, densely activated, Transformer language model, which we call Pathways Language Model PaLM. We trained PaLM on 6144 TPU v4 chips using Pathways, a new ML system which enables highly efficient training across multiple TPU Pods. We demonstrate continued benefits of scaling by achieving state-of-the-art few-shot learning results on hundreds of language understanding and generation benchmarks. On a number of these tasks, PaLM 540B achieves breakthrough performance, outperforming the finetuned state-of-the-art on a suite of multi-step reasoning tasks, and outperforming average human performance on the recently released BIG-bench benchmark. A significant number of BIG-bench tasks showed discontinuous improvements from model scale, meaning that performance steeply increased as we scaled to our largest model. PaLM also has strong capabilities in multilingual tasks and source code generation, which we demonstrate on a wide array of benchmarks. We additionally provide a comprehensive analysis on bias and toxicity, and study the extent of training data memorization with respect to model scale. Finally, we discuss the ethical considerations related to large language models and discuss potential mitigation strategies.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2204.02311](https://arxiv.org/abs/2204.02311) [cs.CL]** |
|           | (or **[arXiv:2204.02311v1](https://arxiv.org/abs/2204.02311v1) [cs.CL]** for this version) |





# 2022-04-05

[Return to Index](#Index)



<h2 id="2022-04-05-1">1. Moment-based Adversarial Training for Embodied Language Comprehension
</h2>

Title: [Moment-based Adversarial Training for Embodied Language Comprehension](https://arxiv.org/abs/2204.00889)

Authors: [Shintaro Ishikawa](https://arxiv.org/search/cs?searchtype=author&query=Ishikawa%2C+S), [Komei Sugiura](https://arxiv.org/search/cs?searchtype=author&query=Sugiura%2C+K)

> In this paper, we focus on a vision-and-language task in which a robot is instructed to execute household tasks. Given an instruction such as "Rinse off a mug and place it in the coffee maker," the robot is required to locate the mug, wash it, and put it in the coffee maker. This is challenging because the robot needs to break down the instruction sentences into subgoals and execute them in the correct order. On the ALFRED benchmark, the performance of state-of-the-art methods is still far lower than that of humans. This is partially because existing methods sometimes fail to infer subgoals that are not explicitly specified in the instruction sentences. We propose Moment-based Adversarial Training (MAT), which uses two types of moments for perturbation updates in adversarial training. We introduce MAT to the embedding spaces of the instruction, subgoals, and state representations to handle their varieties. We validated our method on the ALFRED benchmark, and the results demonstrated that our method outperformed the baseline method for all the metrics on the benchmark.

| Comments: | Accepted for presentation at ICPR2022                        |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Robotics (cs.RO)**; Computation and Language (cs.CL); Computer Vision and Pattern Recognition (cs.CV) |
| Cite as:  | **[arXiv:2204.00889](https://arxiv.org/abs/2204.00889) [cs.RO]** |
|           | (or **[arXiv:2204.00889v1](https://arxiv.org/abs/2204.00889v1) [cs.RO]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.00889Focus to learn more |





<h2 id="2022-04-05-2">2. Do As I Can, Not As I Say: Grounding Language in Robotic Affordances
</h2>

Title: [Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](https://arxiv.org/abs/2204.01691)

Authors: [Michael Ahn](https://arxiv.org/search/cs?searchtype=author&query=Ahn%2C+M), [Anthony Brohan](https://arxiv.org/search/cs?searchtype=author&query=Brohan%2C+A), [Noah Brown](https://arxiv.org/search/cs?searchtype=author&query=Brown%2C+N), [Yevgen Chebotar](https://arxiv.org/search/cs?searchtype=author&query=Chebotar%2C+Y), [Omar Cortes](https://arxiv.org/search/cs?searchtype=author&query=Cortes%2C+O), [Byron David](https://arxiv.org/search/cs?searchtype=author&query=David%2C+B), [Chelsea Finn](https://arxiv.org/search/cs?searchtype=author&query=Finn%2C+C), [Keerthana Gopalakrishnan](https://arxiv.org/search/cs?searchtype=author&query=Gopalakrishnan%2C+K), [Karol Hausman](https://arxiv.org/search/cs?searchtype=author&query=Hausman%2C+K), [Alex Herzog](https://arxiv.org/search/cs?searchtype=author&query=Herzog%2C+A), [Daniel Ho](https://arxiv.org/search/cs?searchtype=author&query=Ho%2C+D), [Jasmine Hsu](https://arxiv.org/search/cs?searchtype=author&query=Hsu%2C+J), [Julian Ibarz](https://arxiv.org/search/cs?searchtype=author&query=Ibarz%2C+J), [Brian Ichter](https://arxiv.org/search/cs?searchtype=author&query=Ichter%2C+B), [Alex Irpan](https://arxiv.org/search/cs?searchtype=author&query=Irpan%2C+A), [Eric Jang](https://arxiv.org/search/cs?searchtype=author&query=Jang%2C+E), [Rosario Jauregui Ruano](https://arxiv.org/search/cs?searchtype=author&query=Ruano%2C+R+J), [Kyle Jeffrey](https://arxiv.org/search/cs?searchtype=author&query=Jeffrey%2C+K), [Sally Jesmonth](https://arxiv.org/search/cs?searchtype=author&query=Jesmonth%2C+S), [Nikhil J Joshi](https://arxiv.org/search/cs?searchtype=author&query=Joshi%2C+N+J), [Ryan Julian](https://arxiv.org/search/cs?searchtype=author&query=Julian%2C+R), [Dmitry Kalashnikov](https://arxiv.org/search/cs?searchtype=author&query=Kalashnikov%2C+D), [Yuheng Kuang](https://arxiv.org/search/cs?searchtype=author&query=Kuang%2C+Y), [Kuang-Huei Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+K), [Sergey Levine](https://arxiv.org/search/cs?searchtype=author&query=Levine%2C+S), [Yao Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu%2C+Y), [Linda Luu](https://arxiv.org/search/cs?searchtype=author&query=Luu%2C+L), [Carolina Parada](https://arxiv.org/search/cs?searchtype=author&query=Parada%2C+C), [Peter Pastor](https://arxiv.org/search/cs?searchtype=author&query=Pastor%2C+P), [Jornell Quiambao](https://arxiv.org/search/cs?searchtype=author&query=Quiambao%2C+J), [Kanishka Rao](https://arxiv.org/search/cs?searchtype=author&query=Rao%2C+K), [Jarek Rettinghouse](https://arxiv.org/search/cs?searchtype=author&query=Rettinghouse%2C+J), [Diego Reyes](https://arxiv.org/search/cs?searchtype=author&query=Reyes%2C+D), [Pierre Sermanet](https://arxiv.org/search/cs?searchtype=author&query=Sermanet%2C+P), [Nicolas Sievers](https://arxiv.org/search/cs?searchtype=author&query=Sievers%2C+N), [Clayton Tan](https://arxiv.org/search/cs?searchtype=author&query=Tan%2C+C), [Alexander Toshev](https://arxiv.org/search/cs?searchtype=author&query=Toshev%2C+A), [Vincent Vanhoucke](https://arxiv.org/search/cs?searchtype=author&query=Vanhoucke%2C+V), [Fei Xia](https://arxiv.org/search/cs?searchtype=author&query=Xia%2C+F), [Ted Xiao](https://arxiv.org/search/cs?searchtype=author&query=Xiao%2C+T), [Peng Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+P), [Sichun Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+S), [Mengyuan Yan](https://arxiv.org/search/cs?searchtype=author&query=Yan%2C+M)

> Large language models can encode a wealth of semantic knowledge about the world. Such knowledge could be extremely useful to robots aiming to act upon high-level, temporally extended instructions expressed in natural language. However, a significant weakness of language models is that they lack real-world experience, which makes it difficult to leverage them for decision making within a given embodiment. For example, asking a language model to describe how to clean a spill might result in a reasonable narrative, but it may not be applicable to a particular agent, such as a robot, that needs to perform this task in a particular environment. We propose to provide real-world grounding by means of pretrained skills, which are used to constrain the model to propose natural language actions that are both feasible and contextually appropriate. The robot can act as the language model's "hands and eyes," while the language model supplies high-level semantic knowledge about the task. We show how low-level skills can be combined with large language models so that the language model provides high-level knowledge about the procedures for performing complex and temporally-extended instructions, while value functions associated with these skills provide the grounding necessary to connect this knowledge to a particular physical environment. We evaluate our method on a number of real-world robotic tasks, where we show the need for real-world grounding and that this approach is capable of completing long-horizon, abstract, natural language instructions on a mobile manipulator. The project's website and the video can be found at [this https URL](https://say-can.github.io/)

| Comments: | See website at [this https URL](https://say-can.github.io/)  |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Robotics (cs.RO)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2204.01691](https://arxiv.org/abs/2204.01691) [cs.RO]** |
|           | (or **[arXiv:2204.01691v1](https://arxiv.org/abs/2204.01691v1) [cs.RO]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.01691Focus to learn more |





<h2 id="2022-04-05-3">3. CipherDAug: Ciphertext based Data Augmentation for Neural Machine Translation
</h2>

Title: [CipherDAug: Ciphertext based Data Augmentation for Neural Machine Translation](https://arxiv.org/abs/2204.00665)

Authors: [Nishant Kambhatla](https://arxiv.org/search/cs?searchtype=author&query=Kambhatla%2C+N), [Logan Born](https://arxiv.org/search/cs?searchtype=author&query=Born%2C+L), [Anoop Sarkar](https://arxiv.org/search/cs?searchtype=author&query=Sarkar%2C+A)

> We propose a novel data-augmentation technique for neural machine translation based on ROT-k ciphertexts. ROT-k is a simple letter substitution cipher that replaces a letter in the plaintext with the kth letter after it in the alphabet. We first generate multiple ROT-k ciphertexts using different values of k for the plaintext which is the source side of the parallel data. We then leverage this enciphered training data along with the original parallel data via multi-source training to improve neural machine translation. Our method, CipherDAug, uses a co-regularization-inspired training procedure, requires no external data sources other than the original training data, and uses a standard Transformer to outperform strong data augmentation techniques on several datasets by a significant margin. This technique combines easily with existing approaches to data augmentation, and yields particularly strong results in low-resource settings.

| Comments: | ACL 2022 Main Conf. camera ready version                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2204.00665](https://arxiv.org/abs/2204.00665) [cs.CL]** |
|           | (or **[arXiv:2204.00665v1](https://arxiv.org/abs/2204.00665v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.00665Focus to learn more |





<h2 id="2022-04-05-4">4. On Efficiently Acquiring Annotations for Multilingual Models
</h2>

Title: [On Efficiently Acquiring Annotations for Multilingual Models](https://arxiv.org/abs/2204.01016)

Authors: [Joel Ruben Antony Moniz](https://arxiv.org/search/cs?searchtype=author&query=Moniz%2C+J+R+A), [Barun Patra](https://arxiv.org/search/cs?searchtype=author&query=Patra%2C+B), [Matthew R. Gormley](https://arxiv.org/search/cs?searchtype=author&query=Gormley%2C+M+R)

> When tasked with supporting multiple languages for a given problem, two approaches have arisen: training a model for each language with the annotation budget divided equally among them, and training on a high-resource language followed by zero-shot transfer to the remaining languages. In this work, we show that the strategy of joint learning across multiple languages using a single model performs substantially better than the aforementioned alternatives. We also demonstrate that active learning provides additional, complementary benefits. We show that this simple approach enables the model to be data efficient by allowing it to arbitrate its annotation budget to query languages it is less certain on. We illustrate the effectiveness of our proposed method on a diverse set of tasks: a classification task with 4 languages, a sequence tagging task with 4 languages and a dependency parsing task with 5 languages. Our proposed method, whilst simple, substantially outperforms the other viable alternatives for building a model in a multilingual setting under constrained budgets.

| Comments: | ACL 2022 (Short Paper)                                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2204.01016](https://arxiv.org/abs/2204.01016) [cs.CL]** |
|           | (or **[arXiv:2204.01016v1](https://arxiv.org/abs/2204.01016v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.01016Focus to learn more |







<h2 id="2022-04-05-5">5. PERFECT: Prompt-free and Efficient Few-shot Learning with Language Models
</h2>

Title: [PERFECT: Prompt-free and Efficient Few-shot Learning with Language Models](https://arxiv.org/abs/2204.01172)

Authors: [Rabeeh Karimi Mahabadi](https://arxiv.org/search/cs?searchtype=author&query=Mahabadi%2C+R+K), [Luke Zettlemoyer](https://arxiv.org/search/cs?searchtype=author&query=Zettlemoyer%2C+L), [James Henderson](https://arxiv.org/search/cs?searchtype=author&query=Henderson%2C+J), [Marzieh Saeidi](https://arxiv.org/search/cs?searchtype=author&query=Saeidi%2C+M), [Lambert Mathias](https://arxiv.org/search/cs?searchtype=author&query=Mathias%2C+L), [Veselin Stoyanov](https://arxiv.org/search/cs?searchtype=author&query=Stoyanov%2C+V), [Majid Yazdani](https://arxiv.org/search/cs?searchtype=author&query=Yazdani%2C+M)

> Current methods for few-shot fine-tuning of pretrained masked language models (PLMs) require carefully engineered prompts and verbalizers for each new task to convert examples into a cloze-format that the PLM can score. In this work, we propose PERFECT, a simple and efficient method for few-shot fine-tuning of PLMs without relying on any such handcrafting, which is highly effective given as few as 32 data points. PERFECT makes two key design choices: First, we show that manually engineered task prompts can be replaced with task-specific adapters that enable sample-efficient fine-tuning and reduce memory and storage costs by roughly factors of 5 and 100, respectively. Second, instead of using handcrafted verbalizers, we learn new multi-token label embeddings during fine-tuning, which are not tied to the model vocabulary and which allow us to avoid complex auto-regressive decoding. These embeddings are not only learnable from limited data but also enable nearly 100x faster training and inference. Experiments on a wide range of few-shot NLP tasks demonstrate that PERFECT, while being simple and efficient, also outperforms existing state-of-the-art few-shot learning methods. Our code is publicly available at [this https URL](https://github.com/rabeehk/perfect).

| Comments: | ACL, 2022                                                    |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2204.01172](https://arxiv.org/abs/2204.01172) [cs.CL]** |
|           | (or **[arXiv:2204.01172v1](https://arxiv.org/abs/2204.01172v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.01172Focus to learn more |





<h2 id="2022-04-05-6">6. Aligned Weight Regularizers for Pruning Pretrained Neural Networks
</h2>

Title: [Aligned Weight Regularizers for Pruning Pretrained Neural Networks](https://arxiv.org/abs/2204.01385)

Authors: [James O' Neill](https://arxiv.org/search/cs?searchtype=author&query=Neill%2C+J+O), [Sourav Dutta](https://arxiv.org/search/cs?searchtype=author&query=Dutta%2C+S), [Haytham Assem](https://arxiv.org/search/cs?searchtype=author&query=Assem%2C+H)

> While various avenues of research have been explored for iterative pruning, little is known what effect pruning has on zero-shot test performance and its potential implications on the choice of pruning criteria. This pruning setup is particularly important for cross-lingual models that implicitly learn alignment between language representations during pretraining, which if distorted via pruning, not only leads to poorer performance on language data used for retraining but also on zero-shot languages that are evaluated. 
> In this work, we show that there is a clear performance discrepancy in magnitude-based pruning when comparing standard supervised learning to the zero-shot setting. From this finding, we propose two weight regularizers that aim to maximize the alignment between units of pruned and unpruned networks to mitigate alignment distortion in pruned cross-lingual models and perform well for both non zero-shot and zero-shot settings. 
> We provide experimental results on cross-lingual tasks for the zero-shot setting using XLM-RoBERTaBase, where we also find that pruning has varying degrees of representational degradation depending on the language corresponding to the zero-shot test set. This is also the first study that focuses on cross-lingual language model compression.

| Comments: | Accepted to ACL Findings 2022                                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2204.01385](https://arxiv.org/abs/2204.01385) [cs.CL]** |
|           | (or **[arXiv:2204.01385v1](https://arxiv.org/abs/2204.01385v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.01385Focus to learn more |





<h2 id="2022-04-05-7">7. Estimating the Entropy of Linguistic Distributions
</h2>

Title: [Estimating the Entropy of Linguistic Distributions](https://arxiv.org/abs/2204.01469)

Authors: [Aryaman Arora](https://arxiv.org/search/cs?searchtype=author&query=Arora%2C+A), [Clara Meister](https://arxiv.org/search/cs?searchtype=author&query=Meister%2C+C), [Ryan Cotterell](https://arxiv.org/search/cs?searchtype=author&query=Cotterell%2C+R)

> Shannon entropy is often a quantity of interest to linguists studying the communicative capacity of human language. However, entropy must typically be estimated from observed data because researchers do not have access to the underlying probability distribution that gives rise to these data. While entropy estimation is a well-studied problem in other fields, there is not yet a comprehensive exploration of the efficacy of entropy estimators for use with linguistic data. In this work, we fill this void, studying the empirical effectiveness of different entropy estimators for linguistic distributions. In a replication of two recent information-theoretic linguistic studies, we find evidence that the reported effect size is over-estimated due to over-reliance on poor entropy estimators. Finally, we end our paper with concrete recommendations for entropy estimation depending on distribution type and data availability.

| Comments:    | 21 pages (5 pages main text). 4 figures. Accepted to ACL 2022 |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Computation and Language (cs.CL)**                         |
| MSC classes: | 94A17 (Primary) 62B10 (Secondary)                            |
| ACM classes: | I.2.7; E.4                                                   |
| Cite as:     | **[arXiv:2204.01469](https://arxiv.org/abs/2204.01469) [cs.CL]** |
|              | (or **[arXiv:2204.01469v1](https://arxiv.org/abs/2204.01469v1) [cs.CL]** for this version) |
|              | https://doi.org/10.48550/arXiv.2204.01469Focus to learn more |







# 2022-04-04

[Return to Index](#Index)



<h2 id="2022-04-04-1">1. AdaSpeech 4: Adaptive Text to Speech in Zero-Shot Scenarios
</h2>

Title: [AdaSpeech 4: Adaptive Text to Speech in Zero-Shot Scenarios](https://arxiv.org/abs/2204.00436)

Authors: [Yihan Wu](https://arxiv.org/search/eess?searchtype=author&query=Wu%2C+Y), [Xu Tan](https://arxiv.org/search/eess?searchtype=author&query=Tan%2C+X), [Bohan Li](https://arxiv.org/search/eess?searchtype=author&query=Li%2C+B), [Lei He](https://arxiv.org/search/eess?searchtype=author&query=He%2C+L), [Sheng Zhao](https://arxiv.org/search/eess?searchtype=author&query=Zhao%2C+S), [Ruihua Song](https://arxiv.org/search/eess?searchtype=author&query=Song%2C+R), [Tao Qin](https://arxiv.org/search/eess?searchtype=author&query=Qin%2C+T), [Tie-Yan Liu](https://arxiv.org/search/eess?searchtype=author&query=Liu%2C+T)

> Adaptive text to speech (TTS) can synthesize new voices in zero-shot scenarios efficiently, by using a well-trained source TTS model without adapting it on the speech data of new speakers. Considering seen and unseen speakers have diverse characteristics, zero-shot adaptive TTS requires strong generalization ability on speaker characteristics, which brings modeling challenges. In this paper, we develop AdaSpeech 4, a zero-shot adaptive TTS system for high-quality speech synthesis. We model the speaker characteristics systematically to improve the generalization on new speakers. Generally, the modeling of speaker characteristics can be categorized into three steps: extracting speaker representation, taking this speaker representation as condition, and synthesizing speech/mel-spectrogram given this speaker representation. Accordingly, we improve the modeling in three steps: 1) To extract speaker representation with better generalization, we factorize the speaker characteristics into basis vectors and extract speaker representation by weighted combining of these basis vectors through attention. 2) We leverage conditional layer normalization to integrate the extracted speaker representation to TTS model. 3) We propose a novel supervision loss based on the distribution of basis vectors to maintain the corresponding speaker characteristics in generated mel-spectrograms. Without any fine-tuning, AdaSpeech 4 achieves better voice quality and similarity than baselines in multiple datasets.

| Comments: | 5 pages, 2 tables, 2 figure. Submitted to Interspeech 2022   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Audio and Speech Processing (eess.AS)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD) |
| Cite as:  | **[arXiv:2204.00436](https://arxiv.org/abs/2204.00436) [eess.AS]** |
|           | (or **[arXiv:2204.00436v1](https://arxiv.org/abs/2204.00436v1) [eess.AS]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.00436Focus to learn more |





<h2 id="2022-04-04-2">2. Unified and Effective Ensemble Knowledge Distillation
</h2>

Title: [Unified and Effective Ensemble Knowledge Distillation](https://arxiv.org/abs/2204.00548)

Authors: [Chuhan Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+C), [Fangzhao Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+F), [Tao Qi](https://arxiv.org/search/cs?searchtype=author&query=Qi%2C+T), [Yongfeng Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+Y)

> Ensemble knowledge distillation can extract knowledge from multiple teacher models and encode it into a single student model. Many existing methods learn and distill the student model on labeled data only. However, the teacher models are usually learned on the same labeled data, and their predictions have high correlations with groudtruth labels. Thus, they cannot provide sufficient knowledge complementary to task labels for student teaching. Distilling on unseen unlabeled data has the potential to enhance the knowledge transfer from the teachers to the student. In this paper, we propose a unified and effective ensemble knowledge distillation method that distills a single student model from an ensemble of teacher models on both labeled and unlabeled data. Since different teachers may have diverse prediction correctness on the same sample, on labeled data we weight the predictions of different teachers according to their correctness. In addition, we weight the distillation loss based on the overall prediction correctness of the teacher ensemble to distill high-quality knowledge. On unlabeled data, there is no groundtruth to evaluate prediction correctness. Fortunately, the disagreement among teachers is an indication of sample hardness, and thereby we weight the distillation loss based on teachers' disagreement to emphasize knowledge distillation on important samples. Extensive experiments on four datasets show the effectiveness of our proposed ensemble distillation method.

| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2204.00548](https://arxiv.org/abs/2204.00548) [cs.LG]** |
|           | (or **[arXiv:2204.00548v1](https://arxiv.org/abs/2204.00548v1) [cs.LG]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.00548Focus to learn more |





<h2 id="2022-04-04-3">3. Better Intermediates Improve CTC Inference
</h2>

Title: [Better Intermediates Improve CTC Inference](https://arxiv.org/abs/2204.00176)

Authors: [Tatsuya Komatsu](https://arxiv.org/search/cs?searchtype=author&query=Komatsu%2C+T), [Yusuke Fujita](https://arxiv.org/search/cs?searchtype=author&query=Fujita%2C+Y), [Jaesong Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+J), [Lukas Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+L), [Shinji Watanabe](https://arxiv.org/search/cs?searchtype=author&query=Watanabe%2C+S), [Yusuke Kida](https://arxiv.org/search/cs?searchtype=author&query=Kida%2C+Y)

> This paper proposes a method for improved CTC inference with searched intermediates and multi-pass conditioning. The paper first formulates self-conditioned CTC as a probabilistic model with an intermediate prediction as a latent representation and provides a tractable conditioning framework. We then propose two new conditioning methods based on the new formulation: (1) Searched intermediate conditioning that refines intermediate predictions with beam-search, (2) Multi-pass conditioning that uses predictions of previous inference for conditioning the next inference. These new approaches enable better conditioning than the original self-conditioned CTC during inference and improve the final performance. Experiments with the LibriSpeech dataset show relative 3%/12% performance improvement at the maximum in test clean/other sets compared to the original self-conditioned CTC.

| Comments: | 5 pages, submitted INTERSPEECH2022                           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2204.00176](https://arxiv.org/abs/2204.00176) [cs.CL]** |
|           | (or **[arXiv:2204.00176v1](https://arxiv.org/abs/2204.00176v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.00176Focus to learn more |





<h2 id="2022-04-04-4">4. WavFT: Acoustic model finetuning with labelled and unlabelled data
</h2>

Title: [WavFT: Acoustic model finetuning with labelled and unlabelled data](https://arxiv.org/abs/2204.00348)

Authors: [Utkarsh Chauhan](https://arxiv.org/search/cs?searchtype=author&query=Chauhan%2C+U), [Vikas Joshi](https://arxiv.org/search/cs?searchtype=author&query=Joshi%2C+V), [Rupesh R. Mehta](https://arxiv.org/search/cs?searchtype=author&query=Mehta%2C+R+R)

> Unsupervised and self-supervised learning methods have leveraged unlabelled data to improve the pretrained models. However, these methods need significantly large amount of unlabelled data and the computational cost of training models with such large amount of data can be prohibitively high. We address this issue by using unlabelled data during finetuning, instead of pretraining. We propose acoustic model finetuning (FT) using labelled and unlabelled data. The model is jointly trained to learn representations to classify senones, as well as learn contextual acoustic representations. Our training objective is a combination of cross entropy loss, suitable for classification task, and contrastive loss, suitable to learn acoustic representations. The proposed approach outperforms conventional finetuning with 11.2% and 9.19% word error rate relative (WERR) reduction on Gujarati and Bengali languages respectively.

| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2204.00348](https://arxiv.org/abs/2204.00348) [cs.CL]** |
|           | (or **[arXiv:2204.00348v1](https://arxiv.org/abs/2204.00348v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.00348Focus to learn more |





<h2 id="2022-04-04-5">5. Uncertainty Determines the Adequacy of the Mode and the Tractability of Decoding in Sequence-to-Sequence Models
</h2>

Title: [Uncertainty Determines the Adequacy of the Mode and the Tractability of Decoding in Sequence-to-Sequence Models](https://arxiv.org/abs/2204.00471)

Authors: [Felix Stahlberg](https://arxiv.org/search/cs?searchtype=author&query=Stahlberg%2C+F), [Ilia Kulikov](https://arxiv.org/search/cs?searchtype=author&query=Kulikov%2C+I), [Shankar Kumar](https://arxiv.org/search/cs?searchtype=author&query=Kumar%2C+S)

> In many natural language processing (NLP) tasks the same input (e.g. source sentence) can have multiple possible outputs (e.g. translations). To analyze how this ambiguity (also known as intrinsic uncertainty) shapes the distribution learned by neural sequence models we measure sentence-level uncertainty by computing the degree of overlap between references in multi-reference test sets from two different NLP tasks: machine translation (MT) and grammatical error correction (GEC). At both the sentence- and the task-level, intrinsic uncertainty has major implications for various aspects of search such as the inductive biases in beam search and the complexity of exact search. In particular, we show that well-known pathologies such as a high number of beam search errors, the inadequacy of the mode, and the drop in system performance with large beam sizes apply to tasks with high level of ambiguity such as MT but not to less uncertain tasks such as GEC. Furthermore, we propose a novel exact n-best search algorithm for neural sequence models, and show that intrinsic uncertainty affects model uncertainty as the model tends to overly spread out the probability mass for uncertain tasks and sentences.

| Comments: | ACL 2022 paper                                               |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2204.00471](https://arxiv.org/abs/2204.00471) [cs.CL]** |
|           | (or **[arXiv:2204.00471v1](https://arxiv.org/abs/2204.00471v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.00471Focus to learn more |








# 2022-04-01

[Return to Index](#Index)



<h2 id="2022-04-01-1">1. MAE-AST: Masked Autoencoding Audio Spectrogram Transformer
</h2>

Title: [MAE-AST: Masked Autoencoding Audio Spectrogram Transformer](https://arxiv.org/abs/2203.16691)

Authors: [Alan Baade](https://arxiv.org/search/eess?searchtype=author&query=Baade%2C+A), [Puyuan Peng](https://arxiv.org/search/eess?searchtype=author&query=Peng%2C+P), [David Harwath](https://arxiv.org/search/eess?searchtype=author&query=Harwath%2C+D)

> In this paper, we propose a simple yet powerful improvement over the recent Self-Supervised Audio Spectrogram Transformer (SSAST) model for speech and audio classification. Specifically, we leverage the insight that the SSAST uses a very high masking ratio (75%) during pretraining, meaning that the vast majority of self-attention compute is performed on mask tokens. We address this by integrating the encoder-decoder architecture from Masked Autoencoders are Scalable Vision Learners (MAE) into the SSAST, where a deep encoder operates on only unmasked input, and a shallow decoder operates on encoder outputs and mask tokens. We find that MAE-like pretraining can provide a 3x speedup and 2x memory usage reduction over the vanilla SSAST using current audio pretraining strategies with ordinary model and input sizes. When fine-tuning on downstream tasks, which only uses the encoder, we find that our approach outperforms the SSAST on a variety of downstream tasks. We further conduct comprehensive evaluations into different strategies of pretraining and explore differences in MAE-style pretraining between the visual and audio domains.

| Comments: | Submitted to INTERSPEECH. 5 pages, 2 figures, 5 tables       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Audio and Speech Processing (eess.AS)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD) |
| Cite as:  | **[arXiv:2203.16691](https://arxiv.org/abs/2203.16691) [eess.AS]** |
|           | (or **[arXiv:2203.16691v1](https://arxiv.org/abs/2203.16691v1) [eess.AS]** for this version) |





<h2 id="2022-03-31-2">2. Exploiting Single-Channel Speech for Multi-Channel End-to-End Speech Recognition: A Comparative Study
</h2>

Title: [Exploiting Single-Channel Speech for Multi-Channel End-to-End Speech Recognition: A Comparative Study](https://arxiv.org/abs/2203.16757)

Authors: [Keyu An](https://arxiv.org/search/eess?searchtype=author&query=An%2C+K), [Zhijian Ou](https://arxiv.org/search/eess?searchtype=author&query=Ou%2C+Z)

> Recently, the end-to-end training approach for multi-channel ASR has shown its effectiveness, which usually consists of a beamforming front-end and a recognition back-end. However, the end-to-end training becomes more difficult due to the integration of multiple modules, particularly considering that multi-channel speech data recorded in real environments are limited in size. This raises the demand to exploit the single-channel data for multi-channel end-to-end ASR. In this paper, we systematically compare the performance of three schemes to exploit external single-channel data for multi-channel end-to-end ASR, namely back-end pre-training, data scheduling, and data simulation, under different settings such as the sizes of the single-channel data and the choices of the front-end. Extensive experiments on CHiME-4 and AISHELL-4 datasets demonstrate that while all three methods improve the multi-channel end-to-end speech recognition performance, data simulation outperforms the other two, at the cost of longer training time. Data scheduling outperforms back-end pre-training marginally but nearly consistently, presumably because that in the pre-training stage, the back-end tends to overfit on the single-channel data, especially when the single-channel data size is small.

| Comments: | submitted to INTERSPEECH 2022. arXiv admin note: substantial text overlap with [arXiv:2107.02670](https://arxiv.org/abs/2107.02670) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Audio and Speech Processing (eess.AS)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2203.16757](https://arxiv.org/abs/2203.16757) [eess.AS]** |
|           | (or **[arXiv:2203.16757v1](https://arxiv.org/abs/2203.16757v1) [eess.AS]** for this version) |





<h2 id="2022-03-31-3">3. How Does Pre-trained Wav2Vec2.0 Perform on Domain Shifted ASR? An Extensive Benchmark on Air Traffic Control Communications
</h2>

Title: [How Does Pre-trained Wav2Vec2.0 Perform on Domain Shifted ASR? An Extensive Benchmark on Air Traffic Control Communications](https://arxiv.org/abs/2203.16822)

Authors: [Juan Zuluaga-Gomez](https://arxiv.org/search/eess?searchtype=author&query=Zuluaga-Gomez%2C+J), [Amrutha Prasad](https://arxiv.org/search/eess?searchtype=author&query=Prasad%2C+A), [Iuliia Nigmatulina](https://arxiv.org/search/eess?searchtype=author&query=Nigmatulina%2C+I), [Saeed Sarfjoo](https://arxiv.org/search/eess?searchtype=author&query=Sarfjoo%2C+S), [Petr Motlicek](https://arxiv.org/search/eess?searchtype=author&query=Motlicek%2C+P), [Matthias Kleinert](https://arxiv.org/search/eess?searchtype=author&query=Kleinert%2C+M), [Hartmut Helmke](https://arxiv.org/search/eess?searchtype=author&query=Helmke%2C+H), [Oliver Ohneiser](https://arxiv.org/search/eess?searchtype=author&query=Ohneiser%2C+O), [Qingran Zhan](https://arxiv.org/search/eess?searchtype=author&query=Zhan%2C+Q)

> Recent work on self-supervised pre-training focus on leveraging large-scale unlabeled speech data to build robust end-to-end (E2E) acoustic models (AM) that can be later fine-tuned on downstream tasks e.g., automatic speech recognition (ASR). Yet, few works investigated the impact on performance when the data substantially differs between the pre-training and downstream fine-tuning phases (i.e., domain shift). We target this scenario by analyzing the robustness of Wav2Vec2.0 and XLS-R models on downstream ASR for a completely unseen domain, i.e., air traffic control (ATC) communications. We benchmark the proposed models on four challenging ATC test sets (signal-to-noise ratio varies between 5 to 20 dB). Relative word error rate (WER) reduction between 20% to 40% are obtained in comparison to hybrid-based state-of-the-art ASR baselines by fine-tuning E2E acoustic models with a small fraction of labeled data. We also study the impact of fine-tuning data size on WERs, going from 5 minutes (few-shot) to 15 hours.

| Comments: | This paper has been submitted to Interspeech 2022            |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Audio and Speech Processing (eess.AS)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2203.16822](https://arxiv.org/abs/2203.16822) [eess.AS]** |
|           | (or **[arXiv:2203.16822v1](https://arxiv.org/abs/2203.16822v1) [eess.AS]** for this version) |





<h2 id="2022-03-31-4">4. Interpretation of Black Box NLP Models: A Survey
</h2>

Title: [Interpretation of Black Box NLP Models: A Survey](https://arxiv.org/abs/2203.17081)

Authors: [Shivani Choudhary](https://arxiv.org/search/cs?searchtype=author&query=Choudhary%2C+S), [Niladri Chatterjee](https://arxiv.org/search/cs?searchtype=author&query=Chatterjee%2C+N), [Subir Kumar Saha](https://arxiv.org/search/cs?searchtype=author&query=Saha%2C+S+K)

> An increasing number of machine learning models have been deployed in domains with high stakes such as finance and healthcare. Despite their superior performances, many models are black boxes in nature which are hard to explain. There are growing efforts for researchers to develop methods to interpret these black-box models. Post hoc explanations based on perturbations, such as LIME, are widely used approaches to interpret a machine learning model after it has been built. This class of methods has been shown to exhibit large instability, posing serious challenges to the effectiveness of the method itself and harming user trust. In this paper, we propose S-LIME, which utilizes a hypothesis testing framework based on central limit theorem for determining the number of perturbation points needed to guarantee stability of the resulting explanation. Experiments on both simulated and real world data sets are provided to demonstrate the effectiveness of our method.

| Subjects: | **Machine Learning (cs.LG)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Information Retrieval (cs.IR) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2203.17081](https://arxiv.org/abs/2203.17081) [cs.LG]** |
|           | (or **[arXiv:2203.17081v1](https://arxiv.org/abs/2203.17081v1) [cs.LG]** for this version) |





<h2 id="2022-03-31-5">5. Scaling Up Models and Data with 𝚝𝟻𝚡 and 
</h2>

Title: [Scaling Up Models and Data with 𝚝𝟻𝚡 and ](https://arxiv.org/abs/2203.17189)

Authors: [Adam Roberts](https://arxiv.org/search/cs?searchtype=author&query=Roberts%2C+A), [Hyung Won Chung](https://arxiv.org/search/cs?searchtype=author&query=Chung%2C+H+W), [Anselm Levskaya](https://arxiv.org/search/cs?searchtype=author&query=Levskaya%2C+A), [Gaurav Mishra](https://arxiv.org/search/cs?searchtype=author&query=Mishra%2C+G), [James Bradbury](https://arxiv.org/search/cs?searchtype=author&query=Bradbury%2C+J), [Daniel Andor](https://arxiv.org/search/cs?searchtype=author&query=Andor%2C+D), [Sharan Narang](https://arxiv.org/search/cs?searchtype=author&query=Narang%2C+S), [Brian Lester](https://arxiv.org/search/cs?searchtype=author&query=Lester%2C+B), [Colin Gaffney](https://arxiv.org/search/cs?searchtype=author&query=Gaffney%2C+C), [Afroz Mohiuddin](https://arxiv.org/search/cs?searchtype=author&query=Mohiuddin%2C+A), [Curtis Hawthorne](https://arxiv.org/search/cs?searchtype=author&query=Hawthorne%2C+C), [Aitor Lewkowycz](https://arxiv.org/search/cs?searchtype=author&query=Lewkowycz%2C+A), [Alex Salcianu](https://arxiv.org/search/cs?searchtype=author&query=Salcianu%2C+A), [Marc van Zee](https://arxiv.org/search/cs?searchtype=author&query=van+Zee%2C+M), [Jacob Austin](https://arxiv.org/search/cs?searchtype=author&query=Austin%2C+J), [Sebastian Goodman](https://arxiv.org/search/cs?searchtype=author&query=Goodman%2C+S), [Livio Baldini Soares](https://arxiv.org/search/cs?searchtype=author&query=Soares%2C+L+B), [Haitang Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu%2C+H), [Sasha Tsvyashchenko](https://arxiv.org/search/cs?searchtype=author&query=Tsvyashchenko%2C+S), [Aakanksha Chowdhery](https://arxiv.org/search/cs?searchtype=author&query=Chowdhery%2C+A), [Jasmijn Bastings](https://arxiv.org/search/cs?searchtype=author&query=Bastings%2C+J), [Jannis Bulian](https://arxiv.org/search/cs?searchtype=author&query=Bulian%2C+J), [Xavier Garcia](https://arxiv.org/search/cs?searchtype=author&query=Garcia%2C+X), [Jianmo Ni](https://arxiv.org/search/cs?searchtype=author&query=Ni%2C+J), [Andrew Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+A), [Kathleen Kenealy](https://arxiv.org/search/cs?searchtype=author&query=Kenealy%2C+K), [Jonathan H. Clark](https://arxiv.org/search/cs?searchtype=author&query=Clark%2C+J+H), [Stephan Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+S), [Dan Garrette](https://arxiv.org/search/cs?searchtype=author&query=Garrette%2C+D), [James Lee-Thorp](https://arxiv.org/search/cs?searchtype=author&query=Lee-Thorp%2C+J), [Colin Raffel](https://arxiv.org/search/cs?searchtype=author&query=Raffel%2C+C), [Noam Shazeer](https://arxiv.org/search/cs?searchtype=author&query=Shazeer%2C+N), [Marvin Ritter](https://arxiv.org/search/cs?searchtype=author&query=Ritter%2C+M), [Maarten Bosma](https://arxiv.org/search/cs?searchtype=author&query=Bosma%2C+M), [Alexandre Passos](https://arxiv.org/search/cs?searchtype=author&query=Passos%2C+A), [Jeremy Maitin-Shepard](https://arxiv.org/search/cs?searchtype=author&query=Maitin-Shepard%2C+J), [Noah Fiedel](https://arxiv.org/search/cs?searchtype=author&query=Fiedel%2C+N), [Mark Omernick](https://arxiv.org/search/cs?searchtype=author&query=Omernick%2C+M), [Brennan Saeta](https://arxiv.org/search/cs?searchtype=author&query=Saeta%2C+B), [Ryan Sepassi](https://arxiv.org/search/cs?searchtype=author&query=Sepassi%2C+R), [Alexander Spiridonov](https://arxiv.org/search/cs?searchtype=author&query=Spiridonov%2C+A), [Joshua Newlan](https://arxiv.org/search/cs?searchtype=author&query=Newlan%2C+J), [Andrea Gesmundo](https://arxiv.org/search/cs?searchtype=author&query=Gesmundo%2C+A)

> Recent neural network-based language models have benefited greatly from scaling up the size of training datasets and the number of parameters in the models themselves. Scaling can be complicated due to various factors including the need to distribute computation on supercomputer clusters (e.g., TPUs), prevent bottlenecks when infeeding data, and ensure reproducible results. In this work, we present two software libraries that ease these issues: 𝚝𝟻𝚡 simplifies the process of building and training large language models at scale while maintaining ease of use, and 𝚜𝚎𝚚𝚒𝚘 provides a task-based API for simple creation of fast and reproducible training data and evaluation pipelines. These open-source libraries have been used to train models with hundreds of billions of parameters on datasets with multiple terabytes of training data. 
> Along with the libraries, we release configurations and instructions for T5-like encoder-decoder models as well as GPT-like decoder-only architectures. 
> 𝚝𝟻𝚡 and 𝚜𝚎𝚚𝚒𝚘 are open source and available at [this https URL](https://github.com/google-research/t5x) and [this https URL](https://github.com/google/seqio), respectively.

| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2203.17189](https://arxiv.org/abs/2203.17189) [cs.LG]** |
|           | (or **[arXiv:2203.17189v1](https://arxiv.org/abs/2203.17189v1) [cs.LG]** for this version) |





<h2 id="2022-03-31-6">6. Mixed-Phoneme BERT: Improving BERT with Mixed Phoneme and Sup-Phoneme Representations for Text to Speech
</h2>

Title: [Mixed-Phoneme BERT: Improving BERT with Mixed Phoneme and Sup-Phoneme Representations for Text to Speech](https://arxiv.org/abs/2203.17190)

Authors: [Guangyan Zhang](https://arxiv.org/search/eess?searchtype=author&query=Zhang%2C+G), [Kaitao Song](https://arxiv.org/search/eess?searchtype=author&query=Song%2C+K), [Xu Tan](https://arxiv.org/search/eess?searchtype=author&query=Tan%2C+X), [Daxin Tan](https://arxiv.org/search/eess?searchtype=author&query=Tan%2C+D), [Yuzi Yan](https://arxiv.org/search/eess?searchtype=author&query=Yan%2C+Y), [Yanqing Liu](https://arxiv.org/search/eess?searchtype=author&query=Liu%2C+Y), [Gang Wang](https://arxiv.org/search/eess?searchtype=author&query=Wang%2C+G), [Wei Zhou](https://arxiv.org/search/eess?searchtype=author&query=Zhou%2C+W), [Tao Qin](https://arxiv.org/search/eess?searchtype=author&query=Qin%2C+T), [Tan Lee](https://arxiv.org/search/eess?searchtype=author&query=Lee%2C+T), [Sheng Zhao](https://arxiv.org/search/eess?searchtype=author&query=Zhao%2C+S)

> Recently, leveraging BERT pre-training to improve the phoneme encoder in text to speech (TTS) has drawn increasing attention. However, the works apply pre-training with character-based units to enhance the TTS phoneme encoder, which is inconsistent with the TTS fine-tuning that takes phonemes as input. Pre-training only with phonemes as input can alleviate the input mismatch but lack the ability to model rich representations and semantic information due to limited phoneme vocabulary. In this paper, we propose MixedPhoneme BERT, a novel variant of the BERT model that uses mixed phoneme and sup-phoneme representations to enhance the learning capability. Specifically, we merge the adjacent phonemes into sup-phonemes and combine the phoneme sequence and the merged sup-phoneme sequence as the model input, which can enhance the model capacity to learn rich contextual representations. Experiment results demonstrate that our proposed Mixed-Phoneme BERT significantly improves the TTS performance with 0.30 CMOS gain compared with the FastSpeech 2 baseline. The Mixed-Phoneme BERT achieves 3x inference speedup and similar voice quality to the previous TTS pre-trained model PnG BERT

| Comments: | submitted to interspeech 2022                                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Audio and Speech Processing (eess.AS)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2203.17190](https://arxiv.org/abs/2203.17190) [eess.AS]** |
|           | (or **[arXiv:2203.17190v1](https://arxiv.org/abs/2203.17190v1) [eess.AS]** for this version) |





<h2 id="2022-03-31-7">7. VL-InterpreT: An Interactive Visualization Tool for Interpreting Vision-Language Transformers
</h2>

Title: [VL-InterpreT: An Interactive Visualization Tool for Interpreting Vision-Language Transformers](https://arxiv.org/abs/2203.17247)

Authors: [Estelle Aflalo](https://arxiv.org/search/cs?searchtype=author&query=Aflalo%2C+E), [Meng Du](https://arxiv.org/search/cs?searchtype=author&query=Du%2C+M), [Shao-Yen Tseng](https://arxiv.org/search/cs?searchtype=author&query=Tseng%2C+S), [Yongfei Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Y), [Chenfei Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+C), [Nan Duan](https://arxiv.org/search/cs?searchtype=author&query=Duan%2C+N), [Vasudev Lal](https://arxiv.org/search/cs?searchtype=author&query=Lal%2C+V)

> Breakthroughs in transformer-based models have revolutionized not only the NLP field, but also vision and multimodal systems. However, although visualization and interpretability tools have become available for NLP models, internal mechanisms of vision and multimodal transformers remain largely opaque. With the success of these transformers, it is increasingly critical to understand their inner workings, as unraveling these black-boxes will lead to more capable and trustworthy models. To contribute to this quest, we propose VL-InterpreT, which provides novel interactive visualizations for interpreting the attentions and hidden representations in multimodal transformers. VL-InterpreT is a task agnostic and integrated tool that (1) tracks a variety of statistics in attention heads throughout all layers for both vision and language components, (2) visualizes cross-modal and intra-modal attentions through easily readable heatmaps, and (3) plots the hidden representations of vision and language tokens as they pass through the transformer layers. In this paper, we demonstrate the functionalities of VL-InterpreT through the analysis of KD-VLP, an end-to-end pretraining vision-language multimodal transformer-based model, in the tasks of Visual Commonsense Reasoning (VCR) and WebQA, two visual question answering benchmarks. Furthermore, we also present a few interesting findings about multimodal transformer behaviors that were learned through our tool.

| Comments: | CVPR 2022 demo track                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2203.17247](https://arxiv.org/abs/2203.17247) [cs.CV]** |
|           | (or **[arXiv:2203.17247v1](https://arxiv.org/abs/2203.17247v1) [cs.CV]** for this version) |





<h2 id="2022-03-31-8">8. Is Word Error Rate a good evaluation metric for Speech Recognition in Indic Languages?
</h2>

Title: [Is Word Error Rate a good evaluation metric for Speech Recognition in Indic Languages?](https://arxiv.org/abs/2203.16601)

Authors: [Priyanshi Shah](https://arxiv.org/search/cs?searchtype=author&query=Shah%2C+P), [Harveen Singh Chadha](https://arxiv.org/search/cs?searchtype=author&query=Chadha%2C+H+S), [Anirudh Gupta](https://arxiv.org/search/cs?searchtype=author&query=Gupta%2C+A), [Ankur Dhuriya](https://arxiv.org/search/cs?searchtype=author&query=Dhuriya%2C+A), [Neeraj Chhimwal](https://arxiv.org/search/cs?searchtype=author&query=Chhimwal%2C+N), [Rishabh Gaur](https://arxiv.org/search/cs?searchtype=author&query=Gaur%2C+R), [Vivek Raghavan](https://arxiv.org/search/cs?searchtype=author&query=Raghavan%2C+V)

> We propose a new method for the calculation of error rates in Automatic Speech Recognition (ASR). This new metric is for languages that contain half characters and where the same character can be written in different forms. We implement our methodology in Hindi which is one of the main languages from Indic context and we think this approach is scalable to other similar languages containing a large character set. We call our metrics Alternate Word Error Rate (AWER) and Alternate Character Error Rate (ACER). 
> We train our ASR models using wav2vec 2.0\cite{baevski2020wav2vec} for Indic languages. Additionally we use language models to improve our model performance. Our results show a significant improvement in analyzing the error rates at word and character level and the interpretability of the ASR system is improved upto 3\% in AWER and 7\% in ACER for Hindi. Our experiments suggest that in languages which have complex pronunciation, there are multiple ways of writing words without changing their meaning. In such cases AWER and ACER will be more useful rather than WER and CER as metrics. Furthermore, we open source a new benchmarking dataset of 21 hours for Hindi with the new metric scripts.

| Comments: | This paper was submitted to Interspeech 2022                 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2203.16601](https://arxiv.org/abs/2203.16601) [cs.CL]** |
|           | (or **[arXiv:2203.16601v1](https://arxiv.org/abs/2203.16601v1) [cs.CL]** for this version) |





<h2 id="2022-03-31-9">9. PADA: Pruning Assisted Domain Adaptation for Self-Supervised Speech Representations
</h2>

Title: [PADA: Pruning Assisted Domain Adaptation for Self-Supervised Speech Representations](https://arxiv.org/abs/2203.16965)

Authors: [Lodagala V S V Durga Prasad](https://arxiv.org/search/cs?searchtype=author&query=Prasad%2C+L+V+S+V+D), [Sreyan Ghosh](https://arxiv.org/search/cs?searchtype=author&query=Ghosh%2C+S), [S. Umesh](https://arxiv.org/search/cs?searchtype=author&query=Umesh%2C+S)

> While self-supervised speech representation learning (SSL) models serve a variety of downstream tasks, these models have been observed to overfit to the domain from which the unlabelled data originates. To alleviate this issue, we propose PADA (Pruning Assisted Domain Adaptation) and zero out redundant weights from models pre-trained on large amounts of out-of-domain (OOD) data. Intuitively, this helps to make space for the target-domain ASR finetuning. The redundant weights can be identified through various pruning strategies which have been discussed in detail as a part of this work. Specifically, we investigate the effect of the recently discovered Task-Agnostic and Task-Aware pruning on PADA and propose a new pruning paradigm based on the latter, which we call Cross-Domain Task-Aware Pruning (CD-TAW). CD-TAW obtains the initial pruning mask from a well fine-tuned OOD model, which makes it starkly different from the rest of the pruning strategies discussed in the paper. Our proposed CD-TAW methodology achieves up to 20.6% relative WER improvement over our baseline when fine-tuned on a 2-hour subset of Switchboard data without language model (LM) decoding. Furthermore, we conduct a detailed analysis to highlight the key design choices of our proposed method.

| Comments: | Submitted to Interspeech 2022                                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2203.16965](https://arxiv.org/abs/2203.16965) [cs.CL]** |
|           | (or **[arXiv:2203.16965v1](https://arxiv.org/abs/2203.16965v1) [cs.CL]** for this version) |





<h2 id="2022-03-31-10">10. Analyzing the factors affecting usefulness of Self-Supervised Pre-trained Representations for Speech Recognition
</h2>

Title: [Analyzing the factors affecting usefulness of Self-Supervised Pre-trained Representations for Speech Recognition](https://arxiv.org/abs/2203.16973)

Authors: [Lodagala V S V Durga Prasad](https://arxiv.org/search/cs?searchtype=author&query=Prasad%2C+L+V+S+V+D), [Ashish Seth](https://arxiv.org/search/cs?searchtype=author&query=Seth%2C+A), [Sreyan Ghosh](https://arxiv.org/search/cs?searchtype=author&query=Ghosh%2C+S), [S. Umesh](https://arxiv.org/search/cs?searchtype=author&query=Umesh%2C+S)

> Self-supervised learning (SSL) to learn high-level speech representations has been a popular approach to building Automatic Speech Recognition (ASR) systems in low-resource settings. However, the common assumption made in literature is that a considerable amount of unlabeled data is available for the same domain or language that can be leveraged for SSL pre-training, which we acknowledge is not feasible in a real-world setting. In this paper, as part of the Interspeech Gram Vaani ASR challenge, we try to study the effect of domain, language, dataset size, and other aspects of our upstream pre-training SSL data on the final performance low-resource downstream ASR task. We also build on the continued pre-training paradigm to study the effect of prior knowledge possessed by models trained using SSL. Extensive experiments and studies reveal that the performance of ASR systems is susceptible to the data used for SSL pre-training. Their performance improves with an increase in similarity and volume of pre-training data. We believe our work will be helpful to the speech community in building better ASR systems in low-resource settings and steer research towards improving generalization in SSL-based pre-training for speech systems.

| Comments: | Submitted to Interspeech 2022                                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2203.16973](https://arxiv.org/abs/2203.16973) [cs.CL]** |
|           | (or **[arXiv:2203.16973v1](https://arxiv.org/abs/2203.16973v1) [cs.CL]** for this version) |





<h2 id="2022-03-31-11">11. PANGUBOT: Efficient Generative Dialogue Pre-training from Pre-trained Language Model
</h2>

Title: [PANGUBOT: Efficient Generative Dialogue Pre-training from Pre-trained Language Model](https://arxiv.org/abs/2203.17090)

Authors: [Fei Mi](https://arxiv.org/search/cs?searchtype=author&query=Mi%2C+F), [Yitong Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Y), [Yulong Zeng](https://arxiv.org/search/cs?searchtype=author&query=Zeng%2C+Y), [Jingyan Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+J), [Yasheng Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Y), [Chuanfei Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+C), [Lifeng Shang](https://arxiv.org/search/cs?searchtype=author&query=Shang%2C+L), [Xin Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+X), [Shiqi Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+S), [Qun Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Q)

> In this paper, we introduce PANGUBOT, a Chinese pre-trained open-domain dialogue generation model based on a large pre-trained language model (PLM) PANGU-alpha (Zeng et al.,2021). Different from other pre-trained dialogue models trained over a massive amount of dialogue data from scratch, we aim to build a powerful dialogue model with relatively fewer data and computation costs by inheriting valuable language capabilities and knowledge from PLMs. To this end, we train PANGUBOT from the large PLM PANGU-alpha, which has been proven well-performed on a variety of Chinese natural language tasks. We investigate different aspects of responses generated by PANGUBOT, including response quality, knowledge, and safety. We show that PANGUBOT outperforms state-of-the-art Chinese dialogue systems (CDIALGPT (Wang et al., 2020), EVA (Zhou et al., 2021)) w.r.t. the above three aspects. We also demonstrate that PANGUBOT can be easily deployed to generate emotional responses without further training. Throughout our empirical analysis, we also point out that the PANGUBOT response quality, knowledge correctness, and safety are still far from perfect, and further explorations are indispensable to building reliable and smart dialogue systems.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2203.17090](https://arxiv.org/abs/2203.17090) [cs.CL]** |
|           | (or **[arXiv:2203.17090v1](https://arxiv.org/abs/2203.17090v1) [cs.CL]** for this version) |










# 2022-03-31

[Return to Index](#Index)



<h2 id="2022-03-31-1">1. WAVPROMPT: Towards Few-Shot Spoken Language Understanding with Frozen Language Models
</h2>

Title: [WAVPROMPT: Towards Few-Shot Spoken Language Understanding with Frozen Language Models](https://arxiv.org/abs/2203.15863)

Authors: [Heting Gao](https://arxiv.org/search/eess?searchtype=author&query=Gao%2C+H), [Junrui Ni](https://arxiv.org/search/eess?searchtype=author&query=Ni%2C+J), [Kaizhi Qian](https://arxiv.org/search/eess?searchtype=author&query=Qian%2C+K), [Yang Zhang](https://arxiv.org/search/eess?searchtype=author&query=Zhang%2C+Y), [Shiyu Chang](https://arxiv.org/search/eess?searchtype=author&query=Chang%2C+S), [Mark Hasegawa-Johnson](https://arxiv.org/search/eess?searchtype=author&query=Hasegawa-Johnson%2C+M)

> Large-scale auto-regressive language models pretrained on massive text have demonstrated their impressive ability to perform new natural language tasks with only a few text examples, without the need for fine-tuning. Recent studies further show that such a few-shot learning ability can be extended to the text-image setting by training an encoder to encode the images into embeddings functioning like the text embeddings of the language model. Interested in exploring the possibility of transferring the few-shot learning ability to the audio-text setting, we propose a novel speech understanding framework, WavPrompt, where we finetune a wav2vec model to generate a sequence of audio embeddings understood by the language model. We show that WavPrompt is a few-shot learner that can perform speech understanding tasks better than a naive text baseline. We conduct detailed ablation studies on different components and hyperparameters to empirically identify the best model configuration. In addition, we conduct a non-speech understanding experiment to show WavPrompt can extract more information than just the transcriptions.

| Comments: | submitted to INTERSPEECH 2022                                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Audio and Speech Processing (eess.AS)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2203.15863](https://arxiv.org/abs/2203.15863) [eess.AS]** |
|           | (or **[arXiv:2203.15863v1](https://arxiv.org/abs/2203.15863v1) [eess.AS]** for this version) |
|           | https://doi.org/10.48550/arXiv.2203.15863Focus to learn more |



