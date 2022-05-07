# MA C.'s Daily Paper Of Interest - May a., 2022

# Index

- [2022-05-06](#2022-05-06)
  - [1. Language Models Can See: Plugging Visual Controls in Text Generation](#2022-05-06-1)
  - [2. Original or Translated? A Causal Analysis of the Impact of Translationese on Machine Translation Performance](#2022-05-06-2)
  - [3. Cross-modal Contrastive Learning for Speech Translation](#2022-05-06-3)
  - [4. A Simple Contrastive Learning Objective for Alleviating Neural Text Degeneration](#2022-05-06-4)
  - [5. Efficient yet Competitive Speech Translation: FBK@IWSLT2022](#2022-05-06-5)
- [2022-05-05](#2022-05-05)
  - [1. P3 Ranker: Mitigating the Gaps between Pre-training and Ranking Fine-tuning with Prompt-based Learning and Pre-finetuning](#2022-05-05-1)
  - [2. Improving In-Context Few-Shot Learning via Self-Supervised Training](#2022-05-05-2)
  - [3. Provably Confidential Language Modelling](#2022-05-05-3)
  - [4. Non-Autoregressive Machine Translation: It's Not as Fast as it Seems](#2022-05-05-4)
  - [5. ON-TRAC Consortium Systems for the IWSLT 2022 Dialect and Low-resource Speech Translation Tasks](#2022-05-05-5)
  - [6. A Few Thousand Translations Go a Long Way! Leveraging Pre-trained Models for African News Translation](#2022-05-05-6)
  - [7. Same Neurons, Different Languages: Probing Morphosyntax in Multilingual Pre-trained Models](#2022-05-05-7)
  - [8. Reproducibility Beyond the Research Community: Experience from NLP Beginners](#2022-05-05-8)
- [2022-05-04](#2022-05-04)
  - [1. Hausa Visual Genome: A Dataset for Multi-Modal English to Hausa Machine Translation](#2022-05-04-1)
  - [2. Contrastive Learning for Prompt-Based Few-Shot Language Learners](#2022-05-04-2)
  - [3. Meta Learning for Natural Language Processing: A Survey](#2022-05-04-3)
  - [4. Learning to Transfer Prompts for Text Generation](#2022-05-04-4)
  - [5. Adaptable Adapters](#2022-05-04-5)
  - [6. Training Mixed-Domain Translation Models via Federated Learning](#2022-05-04-6)
  - [7. OmniKnight: Multilingual Neural Machine Translation with Language-Specific Self-Distillation](#2022-05-04-7)
- [2022-05-03](#2022-05-03)
  - [1. Multimodal Representation Learning With Text and Images](#2022-05-03-1)
  - [2. EasyNLP: A Comprehensive and Easy-to-use Toolkit for Natural Language Processing](#2022-05-03-2)
  - [3. AdapterBias: Parameter-efficient Token-dependent Representation Shift for Adapters in NLP Tasks](#2022-05-03-3)
  - [4. Nearest Neighbor Knowledge Distillation for Neural Machine Translation](#2022-05-03-4)
  - [5. Bilingual End-to-End ASR with Byte-Level Subwords](#2022-05-03-5)
  - [6. Debiased Contrastive Learning of Unsupervised Sentence Representations](#2022-05-03-6)
  - [7. The Implicit Length Bias of Label Smoothing on Beam Search Decoding](#2022-05-03-7)
  - [8. Quality-Aware Decoding for Neural Machine Translation](#2022-05-03-8)
  - [9. OPT: Open Pre-trained Transformer Language Models](#2022-05-03-9)
  - [10. Wav2Seq: Pre-training Speech-to-Text Encoder-Decoder Models Using Pseudo Languages](#2022-05-03-10)
- [2022-05-02](#2022-05-02)
  - [1. Vision-Language Pre-Training for Boosting Scene Text Detectors](#2022-05-03-1)
  - [2. Polyglot Prompt: Multilingual Multitask PrompTraining](#2022-05-03-2)
  - [3. How Robust is Neural Machine Translation to Language Imbalance in Multilingual Tokenizer Training?](#2022-05-03-3)
- [2022-04-29](#2022-04-29)	
  - [1. UniTE: Unified Translation Evaluation](#2022-04-29-1)


- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MAC-Daily-POI-index.md)



# 2022-05-06

[Return to Index](#Index)



<h2 id="2022-05-06-1">1. Language Models Can See: Plugging Visual Controls in Text Generation
</h2>

Title: [Language Models Can See: Plugging Visual Controls in Text Generation](https://arxiv.org/abs/2205.02655)

Authors: [Yixuan Su](https://arxiv.org/search/cs?searchtype=author&query=Su%2C+Y), [Tian Lan](https://arxiv.org/search/cs?searchtype=author&query=Lan%2C+T), [Yahui Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Y), [Fangyu Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+F), [Dani Yogatama](https://arxiv.org/search/cs?searchtype=author&query=Yogatama%2C+D), [Yan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Y), [Lingpeng Kong](https://arxiv.org/search/cs?searchtype=author&query=Kong%2C+L), [Nigel Collier](https://arxiv.org/search/cs?searchtype=author&query=Collier%2C+N)

> Generative language models (LMs) such as GPT-2/3 can be prompted to generate text with remarkable quality. While they are designed for text-prompted generation, it remains an open question how the generation process could be guided by modalities beyond text such as images. In this work, we propose a training-free framework, called MAGIC (iMAge-Guided text generatIon with CLIP), for plugging in visual controls in the generation process and enabling LMs to perform multimodal tasks (e.g., image captioning) in a zero-shot manner. MAGIC is a simple yet efficient plug-and-play framework, which directly combines an off-the-shelf LM (i.e., GPT-2) and an image-text matching model (i.e., CLIP) for image-grounded text generation. During decoding, MAGIC influences the generation of the LM by introducing a CLIP-induced score, called magic score, which regularizes the generated result to be semantically related to a given image while being coherent to the previously generated context. Notably, the proposed decoding scheme does not involve any gradient update operation, therefore being computationally efficient. On the challenging task of zero-shot image captioning, MAGIC outperforms the state-of-the-art method by notable margins with a nearly 27 times decoding speedup. MAGIC is a flexible framework and is theoretically compatible with any text generation tasks that incorporate image grounding. In the experiments, we showcase that it is also capable of performing visually grounded story generation given both an image and a text prompt.

| Comments: | 20 pages, 5 figures, 5 tables                                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2205.02655](https://arxiv.org/abs/2205.02655) [cs.CV]** |
|           | (or **[arXiv:2205.02655v1](https://arxiv.org/abs/2205.02655v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.02655Focus to learn more |





<h2 id="2022-05-06-2">2. Original or Translated? A Causal Analysis of the Impact of Translationese on Machine Translation Performance
</h2>

Title: [Original or Translated? A Causal Analysis of the Impact of Translationese on Machine Translation Performance](https://arxiv.org/abs/2205.02293)

Authors: [Jingwei Ni](https://arxiv.org/search/cs?searchtype=author&query=Ni%2C+J), [Zhijing Jin](https://arxiv.org/search/cs?searchtype=author&query=Jin%2C+Z), [Markus Freitag](https://arxiv.org/search/cs?searchtype=author&query=Freitag%2C+M), [Mrinmaya Sachan](https://arxiv.org/search/cs?searchtype=author&query=Sachan%2C+M), [Bernhard Schölkopf](https://arxiv.org/search/cs?searchtype=author&query=Schölkopf%2C+B)

> Human-translated text displays distinct features from naturally written text in the same language. This phenomena, known as translationese, has been argued to confound the machine translation (MT) evaluation. Yet, we find that existing work on translationese neglects some important factors and the conclusions are mostly correlational but not causal. In this work, we collect CausalMT, a dataset where the MT training data are also labeled with the human translation directions. We inspect two critical factors, the train-test direction match (whether the human translation directions in the training and test sets are aligned), and data-model direction match (whether the model learns in the same direction as the human translation direction in the dataset). We show that these two factors have a large causal effect on the MT performance, in addition to the test-model direction mismatch highlighted by existing work on the impact of translationese. In light of our findings, we provide a set of suggestions for MT training and evaluation. Our code and data are at [this https URL](https://github.com/EdisonNi-hku/CausalMT)

| Comments: | NAACL 2022                                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2205.02293](https://arxiv.org/abs/2205.02293) [cs.CL]** |
|           | (or **[arXiv:2205.02293v1](https://arxiv.org/abs/2205.02293v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.02293Focus to learn more |





<h2 id="2022-05-06-3">3. Cross-modal Contrastive Learning for Speech Translation
</h2>

Title: [Cross-modal Contrastive Learning for Speech Translation](https://arxiv.org/abs/2205.02444)

Authors: [Rong Ye](https://arxiv.org/search/cs?searchtype=author&query=Ye%2C+R), [Mingxuan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+M), [Lei Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+L)

> How can we learn unified representations for spoken utterances and their written text? Learning similar representations for semantically similar speech and text is important for speech translation. To this end, we propose ConST, a cross-modal contrastive learning method for end-to-end speech-to-text translation. We evaluate ConST and a variety of previous baselines on a popular benchmark MuST-C. Experiments show that the proposed ConST consistently outperforms the previous methods on, and achieves an average BLEU of 29.4. The analysis further verifies that ConST indeed closes the representation gap of different modalities -- its learned representation improves the accuracy of cross-modal speech-text retrieval from 4% to 88%. Code and models are available at [this https URL](https://github.com/ReneeYe/ConST).

| Comments: | NAACL 2022 main conference (Long Paper)                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2205.02444](https://arxiv.org/abs/2205.02444) [cs.CL]** |
|           | (or **[arXiv:2205.02444v1](https://arxiv.org/abs/2205.02444v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.02444Focus to learn more |





<h2 id="2022-05-06-4">4. A Simple Contrastive Learning Objective for Alleviating Neural Text Degeneration
</h2>

Title: [A Simple Contrastive Learning Objective for Alleviating Neural Text Degeneration](https://arxiv.org/abs/2205.02517)

Authors: [Shaojie Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+S), [Ruqing Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+R), [Svitlana Vakulenko](https://arxiv.org/search/cs?searchtype=author&query=Vakulenko%2C+S), [Maarten de Rijke](https://arxiv.org/search/cs?searchtype=author&query=de+Rijke%2C+M)

> The cross-entropy objective has proved to be an all-purpose training objective for autoregressive language models (LMs). However, without considering the penalization of problematic tokens, LMs trained using cross-entropy exhibit text degeneration. To address this, unlikelihood training has been proposed to force unlikely tokens to be assigned a low probability by a LM. But unlikelihood does not consider the relationship between the label tokens and the unlikely token candidates, thus showing marginal improvements in degeneration. We propose a new contrastive token learning objective that inherits the advantages of cross-entropy and unlikelihood training and avoids their limitations. The key idea is to force a LM to generate high probabilities for label tokens at each step while low probabilities of negative candidates. Comprehensive experiments on language modeling and open-domain dialogue generation tasks show that the proposed contrastive token objective yields less repetitive texts, with a higher generation quality than unlikelihood training, achieving the new state-of-the-art performance.

| Comments: | 20 pages, 10 figures, 7 tables                               |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2205.02517](https://arxiv.org/abs/2205.02517) [cs.CL]** |
|           | (or **[arXiv:2205.02517v1](https://arxiv.org/abs/2205.02517v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.02517Focus to learn more |





<h2 id="2022-05-06-5">5. Efficient yet Competitive Speech Translation: FBK@IWSLT2022
</h2>

Title: [Efficient yet Competitive Speech Translation: FBK@IWSLT2022](https://arxiv.org/abs/2205.02629)

Authors: [Marco Gaido](https://arxiv.org/search/cs?searchtype=author&query=Gaido%2C+M), [Sara Papi](https://arxiv.org/search/cs?searchtype=author&query=Papi%2C+S), [Dennis Fucci](https://arxiv.org/search/cs?searchtype=author&query=Fucci%2C+D), [Giuseppe Fiameni](https://arxiv.org/search/cs?searchtype=author&query=Fiameni%2C+G), [Matteo Negri](https://arxiv.org/search/cs?searchtype=author&query=Negri%2C+M), [Marco Turchi](https://arxiv.org/search/cs?searchtype=author&query=Turchi%2C+M)

> The primary goal of this FBK's systems submission to the IWSLT 2022 offline and simultaneous speech translation tasks is to reduce model training costs without sacrificing translation quality. As such, we first question the need of ASR pre-training, showing that it is not essential to achieve competitive results. Second, we focus on data filtering, showing that a simple method that looks at the ratio between source and target characters yields a quality improvement of 1 BLEU. Third, we compare different methods to reduce the detrimental effect of the audio segmentation mismatch between training data manually segmented at sentence level and inference data that is automatically segmented. Towards the same goal of training cost reduction, we participate in the simultaneous task with the same model trained for offline ST. The effectiveness of our lightweight training strategy is shown by the high score obtained on the MuST-C en-de corpus (26.7 BLEU) and is confirmed in high-resource data conditions by a 1.6 BLEU improvement on the IWSLT2020 test set over last year's winning system.

| Comments: | IWSLT 2022 System Description                                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2205.02629](https://arxiv.org/abs/2205.02629) [cs.CL]** |
|           | (or **[arXiv:2205.02629v1](https://arxiv.org/abs/2205.02629v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.02629Focus to learn more |





# 2022-05-05

[Return to Index](#Index)



<h2 id="2022-05-05-1">1. P3 Ranker: Mitigating the Gaps between Pre-training and Ranking Fine-tuning with Prompt-based Learning and Pre-finetuning
</h2>

Title: [P3 Ranker: Mitigating the Gaps between Pre-training and Ranking Fine-tuning with Prompt-based Learning and Pre-finetuning](https://arxiv.org/abs/2205.01886)
Authors: [Xiaomeng Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu%2C+X) (1), [Shi Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+S) (2), [Chenyan Xiong](https://arxiv.org/search/cs?searchtype=author&query=Xiong%2C+C) (3), [Zhenghao Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Z) (1), [Zhiyuan Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Z) (2), [Ge Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+G) (1) ((1) Northeastern University, (2) Tsinghua University, (3) Microsoft Research)

> Compared to other language tasks, applying pre-trained language models (PLMs) for search ranking often requires more nuances and training signals. In this paper, we identify and study the two mismatches between pre-training and ranking fine-tuning: the training schema gap regarding the differences in training objectives and model architectures, and the task knowledge gap considering the discrepancy between the knowledge needed in ranking and that learned during pre-training. To mitigate these gaps, we propose Pre-trained, Prompt-learned and Pre-finetuned Neural Ranker (P3 Ranker). P3 Ranker leverages prompt-based learning to convert the ranking task into a pre-training like schema and uses pre-finetuning to initialize the model on intermediate supervised tasks. Experiments on MS MARCO and Robust04 show the superior performances of P3 Ranker in few-shot ranking. Analyses reveal that P3 Ranker is able to better accustom to the ranking task through prompt-based learning and retrieve necessary ranking-oriented knowledge gleaned in pre-finetuning, resulting in data-efficient PLM adaptation. Our code is available at \url{[this https URL](https://github.com/NEUIR/P3Ranker)}.

| Comments:    | Accepted by SIGIR 2022                                       |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Information Retrieval (cs.IR)**; Computation and Language (cs.CL) |
| ACM classes: | H.3.3                                                        |
| Cite as:     | **[arXiv:2205.01886](https://arxiv.org/abs/2205.01886) [cs.IR]** |
|              | (or **[arXiv:2205.01886v1](https://arxiv.org/abs/2205.01886v1) [cs.IR]** for this version) |
|              | https://doi.org/10.48550/arXiv.2205.01886Focus to learn more |
| Related DOI: | https://doi.org/10.1145/3477495.3531786Focus to learn more   |





<h2 id="2022-05-05-2">2. Improving In-Context Few-Shot Learning via Self-Supervised Training
</h2>

Title: [Improving In-Context Few-Shot Learning via Self-Supervised Training](https://arxiv.org/abs/2205.01703)
Authors: [Mingda Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+M), [Jingfei Du](https://arxiv.org/search/cs?searchtype=author&query=Du%2C+J), [Ramakanth Pasunuru](https://arxiv.org/search/cs?searchtype=author&query=Pasunuru%2C+R), [Todor Mihaylov](https://arxiv.org/search/cs?searchtype=author&query=Mihaylov%2C+T), [Srini Iyer](https://arxiv.org/search/cs?searchtype=author&query=Iyer%2C+S), [Veselin Stoyanov](https://arxiv.org/search/cs?searchtype=author&query=Stoyanov%2C+V), [Zornitsa Kozareva](https://arxiv.org/search/cs?searchtype=author&query=Kozareva%2C+Z)

> Self-supervised pretraining has made few-shot learning possible for many NLP tasks. But the pretraining objectives are not typically adapted specifically for in-context few-shot learning. In this paper, we propose to use self-supervision in an intermediate training stage between pretraining and downstream few-shot usage with the goal to teach the model to perform in-context few shot learning. We propose and evaluate four self-supervised objectives on two benchmarks. We find that the intermediate self-supervision stage produces models that outperform strong baselines. Ablation study shows that several factors affect the downstream performance, such as the amount of training data and the diversity of the self-supervised objectives. Human-annotated cross-task supervision and self-supervision are complementary. Qualitative analysis suggests that the self-supervised-trained models are better at following task requirements.

| Comments: | NAACL 2022                                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2205.01703](https://arxiv.org/abs/2205.01703) [cs.CL]** |
|           | (or **[arXiv:2205.01703v1](https://arxiv.org/abs/2205.01703v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.01703Focus to learn more |





<h2 id="2022-05-05-3">3. Provably Confidential Language Modelling
</h2>

Title: [Provably Confidential Language Modelling](https://arxiv.org/abs/2205.01863)
Authors: [Xuandong Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+X), [Lei Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+L), [Yu-Xiang Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Y)

> Large language models are shown to memorize privacy information such as social security numbers in training data. Given the sheer scale of the training corpus, it is challenging to screen and filter these privacy data, either manually or automatically. In this paper, we propose Confidentially Redacted Training (CRT), a method to train language generation models while protecting the confidential segments. We borrow ideas from differential privacy (which solves a related but distinct problem) and show that our method is able to provably prevent unintended memorization by randomizing parts of the training process. Moreover, we show that redaction with an approximately correct screening policy amplifies the confidentiality guarantee. We implement the method for both LSTM and GPT language models. Our experimental results show that the models trained by CRT obtain almost the same perplexity while preserving strong confidentiality.

| Comments: | NAACL 2022                                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Cryptography and Security (cs.CR); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2205.01863](https://arxiv.org/abs/2205.01863) [cs.CL]** |
|           | (or **[arXiv:2205.01863v1](https://arxiv.org/abs/2205.01863v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.01863Focus to learn more |





<h2 id="2022-05-05-4">4. Non-Autoregressive Machine Translation: It's Not as Fast as it Seems
</h2>

Title: [Non-Autoregressive Machine Translation: It's Not as Fast as it Seems](https://arxiv.org/abs/2205.01966)
Authors: [Jindřich Helcl](https://arxiv.org/search/cs?searchtype=author&query=Helcl%2C+J), [Barry Haddow](https://arxiv.org/search/cs?searchtype=author&query=Haddow%2C+B), [Alexandra Birch](https://arxiv.org/search/cs?searchtype=author&query=Birch%2C+A)

> Efficient machine translation models are commercially important as they can increase inference speeds, and reduce costs and carbon emissions. Recently, there has been much interest in non-autoregressive (NAR) models, which promise faster translation. In parallel to the research on NAR models, there have been successful attempts to create optimized autoregressive models as part of the WMT shared task on efficient translation. In this paper, we point out flaws in the evaluation methodology present in the literature on NAR models and we provide a fair comparison between a state-of-the-art NAR model and the autoregressive submissions to the shared task. We make the case for consistent evaluation of NAR models, and also for the importance of comparing NAR models with other widely used methods for improving efficiency. We run experiments with a connectionist-temporal-classification-based (CTC) NAR model implemented in C++ and compare it with AR models using wall clock times. Our results show that, although NAR models are faster on GPUs, with small batch sizes, they are almost always slower under more realistic usage conditions. We call for more realistic and extensive evaluation of NAR models in future work.

| Comments: | NAACL 2022, Camera-ready                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2205.01966](https://arxiv.org/abs/2205.01966) [cs.CL]** |
|           | (or **[arXiv:2205.01966v1](https://arxiv.org/abs/2205.01966v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.01966Focus to learn more |





<h2 id="2022-05-05-5">5. ON-TRAC Consortium Systems for the IWSLT 2022 Dialect and Low-resource Speech Translation Tasks
</h2>

Title: [ON-TRAC Consortium Systems for the IWSLT 2022 Dialect and Low-resource Speech Translation Tasks](https://arxiv.org/abs/2205.01987)
Authors: [Marcely Zanon Boito](https://arxiv.org/search/cs?searchtype=author&query=Boito%2C+M+Z), [John Ortega](https://arxiv.org/search/cs?searchtype=author&query=Ortega%2C+J), [Hugo Riguidel](https://arxiv.org/search/cs?searchtype=author&query=Riguidel%2C+H), [Antoine Laurent](https://arxiv.org/search/cs?searchtype=author&query=Laurent%2C+A), [Loïc Barrault](https://arxiv.org/search/cs?searchtype=author&query=Barrault%2C+L), [Fethi Bougares](https://arxiv.org/search/cs?searchtype=author&query=Bougares%2C+F), [Firas Chaabani](https://arxiv.org/search/cs?searchtype=author&query=Chaabani%2C+F), [Ha Nguyen](https://arxiv.org/search/cs?searchtype=author&query=Nguyen%2C+H), [Florentin Barbier](https://arxiv.org/search/cs?searchtype=author&query=Barbier%2C+F), [Souhir Gahbiche](https://arxiv.org/search/cs?searchtype=author&query=Gahbiche%2C+S), [Yannick Estève](https://arxiv.org/search/cs?searchtype=author&query=Estève%2C+Y)

> This paper describes the ON-TRAC Consortium translation systems developed for two challenge tracks featured in the Evaluation Campaign of IWSLT 2022: low-resource and dialect speech translation. For the Tunisian Arabic-English dataset (low-resource and dialect tracks), we build an end-to-end model as our joint primary submission, and compare it against cascaded models that leverage a large fine-tuned wav2vec 2.0 model for ASR. Our results show that in our settings pipeline approaches are still very competitive, and that with the use of transfer learning, they can outperform end-to-end models for speech translation (ST). For the Tamasheq-French dataset (low-resource track) our primary submission leverages intermediate representations from a wav2vec 2.0 model trained on 234 hours of Tamasheq audio, while our contrastive model uses a French phonetic transcription of the Tamasheq audio as input in a Conformer speech translation architecture jointly trained on automatic speech recognition, ST and machine translation losses. Our results highlight that self-supervised models trained on smaller sets of target data are more effective to low-resource end-to-end ST fine-tuning, compared to large off-the-shelf models. Results also illustrate that even approximate phonetic transcriptions can improve ST scores.

| Comments: | IWSLT 2022 system paper                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2205.01987](https://arxiv.org/abs/2205.01987) [cs.CL]** |
|           | (or **[arXiv:2205.01987v1](https://arxiv.org/abs/2205.01987v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.01987Focus to learn more |





<h2 id="2022-05-05-6">6. A Few Thousand Translations Go a Long Way! Leveraging Pre-trained Models for African News Translation
</h2>

Title: [A Few Thousand Translations Go a Long Way! Leveraging Pre-trained Models for African News Translation](https://arxiv.org/abs/2205.02022)
Authors: [David Ifeoluwa Adelani](https://arxiv.org/search/cs?searchtype=author&query=Adelani%2C+D+I), [Jesujoba Oluwadara Alabi](https://arxiv.org/search/cs?searchtype=author&query=Alabi%2C+J+O), [Angela Fan](https://arxiv.org/search/cs?searchtype=author&query=Fan%2C+A), [Julia Kreutzer](https://arxiv.org/search/cs?searchtype=author&query=Kreutzer%2C+J), [Xiaoyu Shen](https://arxiv.org/search/cs?searchtype=author&query=Shen%2C+X), [Machel Reid](https://arxiv.org/search/cs?searchtype=author&query=Reid%2C+M), [Dana Ruiter](https://arxiv.org/search/cs?searchtype=author&query=Ruiter%2C+D), [Dietrich Klakow](https://arxiv.org/search/cs?searchtype=author&query=Klakow%2C+D), [Peter Nabende](https://arxiv.org/search/cs?searchtype=author&query=Nabende%2C+P), [Ernie Chang](https://arxiv.org/search/cs?searchtype=author&query=Chang%2C+E), [Tajuddeen Gwadabe](https://arxiv.org/search/cs?searchtype=author&query=Gwadabe%2C+T), [Freshia Sackey](https://arxiv.org/search/cs?searchtype=author&query=Sackey%2C+F), [Bonaventure F. P. Dossou](https://arxiv.org/search/cs?searchtype=author&query=Dossou%2C+B+F+P), [Chris Chinenye Emezue](https://arxiv.org/search/cs?searchtype=author&query=Emezue%2C+C+C), [Colin Leong](https://arxiv.org/search/cs?searchtype=author&query=Leong%2C+C), [Michael Beukman](https://arxiv.org/search/cs?searchtype=author&query=Beukman%2C+M), [Shamsuddeen Hassan Muhammad](https://arxiv.org/search/cs?searchtype=author&query=Muhammad%2C+S+H), [Guyo Dub Jarso](https://arxiv.org/search/cs?searchtype=author&query=Jarso%2C+G+D), [Oreen Yousuf](https://arxiv.org/search/cs?searchtype=author&query=Yousuf%2C+O), [Andre Niyongabo Rubungo](https://arxiv.org/search/cs?searchtype=author&query=Rubungo%2C+A+N), [Gilles Hacheme](https://arxiv.org/search/cs?searchtype=author&query=Hacheme%2C+G), [Eric Peter Wairagala](https://arxiv.org/search/cs?searchtype=author&query=Wairagala%2C+E+P), [Muhammad Umair Nasir](https://arxiv.org/search/cs?searchtype=author&query=Nasir%2C+M+U), [Benjamin Ayoade Ajibade](https://arxiv.org/search/cs?searchtype=author&query=Ajibade%2C+B+A), [Tunde Oluwaseyi Ajayi](https://arxiv.org/search/cs?searchtype=author&query=Ajayi%2C+T+O), [Yvonne Wambui Gitau](https://arxiv.org/search/cs?searchtype=author&query=Gitau%2C+Y+W), [Jade Abbott](https://arxiv.org/search/cs?searchtype=author&query=Abbott%2C+J), [Mohamed Ahmed](https://arxiv.org/search/cs?searchtype=author&query=Ahmed%2C+M), [Millicent Ochieng](https://arxiv.org/search/cs?searchtype=author&query=Ochieng%2C+M), [Anuoluwapo Aremu](https://arxiv.org/search/cs?searchtype=author&query=Aremu%2C+A), [Perez Ogayo](https://arxiv.org/search/cs?searchtype=author&query=Ogayo%2C+P), [Jonathan Mukiibi](https://arxiv.org/search/cs?searchtype=author&query=Mukiibi%2C+J), [Fatoumata Ouoba Kabore](https://arxiv.org/search/cs?searchtype=author&query=Kabore%2C+F+O), [Godson Koffi Kalipe](https://arxiv.org/search/cs?searchtype=author&query=Kalipe%2C+G+K), [Derguene Mbaye](https://arxiv.org/search/cs?searchtype=author&query=Mbaye%2C+D), [Allahsera Auguste Tapo](https://arxiv.org/search/cs?searchtype=author&query=Tapo%2C+A+A), [Victoire Memdjokam Koagne](https://arxiv.org/search/cs?searchtype=author&query=Koagne%2C+V+M), [Edwin Munkoh-Buabeng](https://arxiv.org/search/cs?searchtype=author&query=Munkoh-Buabeng%2C+E), [Valencia Wagner](https://arxiv.org/search/cs?searchtype=author&query=Wagner%2C+V), [Idris Abdulmumin](https://arxiv.org/search/cs?searchtype=author&query=Abdulmumin%2C+I), [Ayodele Awokoya](https://arxiv.org/search/cs?searchtype=author&query=Awokoya%2C+A), [Happy Buzaaba](https://arxiv.org/search/cs?searchtype=author&query=Buzaaba%2C+H), [Blessing Sibanda](https://arxiv.org/search/cs?searchtype=author&query=Sibanda%2C+B), [Andiswa Bukula](https://arxiv.org/search/cs?searchtype=author&query=Bukula%2C+A), [Sam Manthalu](https://arxiv.org/search/cs?searchtype=author&query=Manthalu%2C+S)

> Recent advances in the pre-training of language models leverage large-scale datasets to create multilingual models. However, low-resource languages are mostly left out in these datasets. This is primarily because many widely spoken languages are not well represented on the web and therefore excluded from the large-scale crawls used to create datasets. Furthermore, downstream users of these models are restricted to the selection of languages originally chosen for pre-training. This work investigates how to optimally leverage existing pre-trained models to create low-resource translation systems for 16 African languages. We focus on two questions: 1) How can pre-trained models be used for languages not included in the initial pre-training? and 2) How can the resulting translation models effectively transfer to new domains? To answer these questions, we create a new African news corpus covering 16 languages, of which eight languages are not part of any existing evaluation dataset. We demonstrate that the most effective strategy for transferring both to additional languages and to additional domains is to fine-tune large pre-trained models on small quantities of high-quality translation data.

| Comments: | Accepted to NAACL 2022                                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2205.02022](https://arxiv.org/abs/2205.02022) [cs.CL]** |
|           | (or **[arXiv:2205.02022v1](https://arxiv.org/abs/2205.02022v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.02022Focus to learn more |





<h2 id="2022-05-05-7">7. Same Neurons, Different Languages: Probing Morphosyntax in Multilingual Pre-trained Models
</h2>

Title: [Same Neurons, Different Languages: Probing Morphosyntax in Multilingual Pre-trained Models](https://arxiv.org/abs/2205.02023)
Authors: [Karolina Stańczak](https://arxiv.org/search/cs?searchtype=author&query=Stańczak%2C+K), [Edoardo Ponti](https://arxiv.org/search/cs?searchtype=author&query=Ponti%2C+E), [Lucas Torroba Hennigen](https://arxiv.org/search/cs?searchtype=author&query=Hennigen%2C+L+T), [Ryan Cotterell](https://arxiv.org/search/cs?searchtype=author&query=Cotterell%2C+R), [Isabelle Augenstein](https://arxiv.org/search/cs?searchtype=author&query=Augenstein%2C+I)

> The success of multilingual pre-trained models is underpinned by their ability to learn representations shared by multiple languages even in absence of any explicit supervision. However, it remains unclear how these models learn to generalise across languages. In this work, we conjecture that multilingual pre-trained models can derive language-universal abstractions about grammar. In particular, we investigate whether morphosyntactic information is encoded in the same subset of neurons in different languages. We conduct the first large-scale empirical study over 43 languages and 14 morphosyntactic categories with a state-of-the-art neuron-level probe. Our findings show that the cross-lingual overlap between neurons is significant, but its extent may vary across categories and depends on language proximity and pre-training data size.

| Comments: | Accepted at NAACL 2022 (Main Conference)                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2205.02023](https://arxiv.org/abs/2205.02023) [cs.CL]** |
|           | (or **[arXiv:2205.02023v1](https://arxiv.org/abs/2205.02023v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.02023Focus to learn more |





<h2 id="2022-05-05-8">8. Reproducibility Beyond the Research Community: Experience from NLP Beginners
</h2>

Title: [Reproducibility Beyond the Research Community: Experience from NLP Beginners](https://arxiv.org/abs/2205.02182)
Authors: [Shane Storks](https://arxiv.org/search/cs?searchtype=author&query=Storks%2C+S), [Keunwoo Peter Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+K+P), [Joyce Chai](https://arxiv.org/search/cs?searchtype=author&query=Chai%2C+J)

> As NLP research attracts public attention and excitement, it becomes increasingly important for it to be accessible to a broad audience. As the research community works to democratize NLP, it remains unclear whether beginners to the field can easily apply the latest developments. To understand their needs, we conducted a study with 93 students in an introductory NLP course, where students reproduced results of recent NLP papers. Surprisingly, our results suggest that their technical skill (i.e., programming experience) has limited impact on their effort spent completing the exercise. Instead, we find accessibility efforts by research authors to be key to a successful experience, including thorough documentation and easy access to required models and datasets.

| Comments: | Accepted to NAACL 2022                                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2205.02182](https://arxiv.org/abs/2205.02182) [cs.CL]** |
|           | (or **[arXiv:2205.02182v1](https://arxiv.org/abs/2205.02182v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.02182Focus to learn more |





# 2022-05-04

[Return to Index](#Index)



<h2 id="2022-05-04-1">1. Hausa Visual Genome: A Dataset for Multi-Modal English to Hausa Machine Translation
</h2>

Title: [Hausa Visual Genome: A Dataset for Multi-Modal English to Hausa Machine Translation](https://arxiv.org/abs/2205.01133)

Authors: [Idris Abdulmumin](https://arxiv.org/search/cs?searchtype=author&query=Abdulmumin%2C+I), [Satya Ranjan Dash](https://arxiv.org/search/cs?searchtype=author&query=Dash%2C+S+R), [Musa Abdullahi Dawud](https://arxiv.org/search/cs?searchtype=author&query=Dawud%2C+M+A), [Shantipriya Parida](https://arxiv.org/search/cs?searchtype=author&query=Parida%2C+S), [Shamsuddeen Hassan Muhammad](https://arxiv.org/search/cs?searchtype=author&query=Muhammad%2C+S+H), [Ibrahim Sa'id Ahmad](https://arxiv.org/search/cs?searchtype=author&query=Ahmad%2C+I+S), [Subhadarshi Panda](https://arxiv.org/search/cs?searchtype=author&query=Panda%2C+S), [Ondřej Bojar](https://arxiv.org/search/cs?searchtype=author&query=Bojar%2C+O), [Bashir Shehu Galadanci](https://arxiv.org/search/cs?searchtype=author&query=Galadanci%2C+B+S), [Bello Shehu Bello](https://arxiv.org/search/cs?searchtype=author&query=Bello%2C+B+S)

> Multi-modal Machine Translation (MMT) enables the use of visual information to enhance the quality of translations. The visual information can serve as a valuable piece of context information to decrease the ambiguity of input sentences. Despite the increasing popularity of such a technique, good and sizeable datasets are scarce, limiting the full extent of their potential. Hausa, a Chadic language, is a member of the Afro-Asiatic language family. It is estimated that about 100 to 150 million people speak the language, with more than 80 million indigenous speakers. This is more than any of the other Chadic languages. Despite a large number of speakers, the Hausa language is considered low-resource in natural language processing (NLP). This is due to the absence of sufficient resources to implement most NLP tasks. While some datasets exist, they are either scarce, machine-generated, or in the religious domain. Therefore, there is a need to create training and evaluation data for implementing machine learning tasks and bridging the research gap in the language. This work presents the Hausa Visual Genome (HaVG), a dataset that contains the description of an image or a section within the image in Hausa and its equivalent in English. To prepare the dataset, we started by translating the English description of the images in the Hindi Visual Genome (HVG) into Hausa automatically. Afterward, the synthetic Hausa data was carefully post-edited considering the respective images. The dataset comprises 32,923 images and their descriptions that are divided into training, development, test, and challenge test set. The Hausa Visual Genome is the first dataset of its kind and can be used for Hausa-English machine translation, multi-modal research, and image description, among various other natural language processing and generation tasks.

| Comments: | Accepted at Language Resources and Evaluation Conference 2022 (LREC2022) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2205.01133](https://arxiv.org/abs/2205.01133) [cs.CL]** |
|           | (or **[arXiv:2205.01133v1](https://arxiv.org/abs/2205.01133v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.01133Focus to learn more |





<h2 id="2022-05-04-2">2. Contrastive Learning for Prompt-Based Few-Shot Language Learners
</h2>

Title: [Contrastive Learning for Prompt-Based Few-Shot Language Learners](https://arxiv.org/abs/2205.01308)

Authors: [Yiren Jian](https://arxiv.org/search/cs?searchtype=author&query=Jian%2C+Y), [Chongyang Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+C), [Soroush Vosoughi](https://arxiv.org/search/cs?searchtype=author&query=Vosoughi%2C+S)

> The impressive performance of GPT-3 using natural language prompts and in-context learning has inspired work on better fine-tuning of moderately-sized models under this paradigm. Following this line of work, we present a contrastive learning framework that clusters inputs from the same class for better generality of models trained with only limited examples. Specifically, we propose a supervised contrastive framework that clusters inputs from the same class under different augmented "views" and repel the ones from different classes. We create different "views" of an example by appending it with different language prompts and contextual demonstrations. Combining a contrastive loss with the standard masked language modeling (MLM) loss in prompt-based few-shot learners, the experimental results show that our method can improve over the state-of-the-art methods in a diverse set of 15 language tasks. Our framework makes minimal assumptions on the task or the base model, and can be applied to many recent methods with little modification. The code will be made available at: [this https URL](https://github.com/yiren-jian/LM-SupCon).

| Comments: | accepted to NAACL 2022                                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2205.01308](https://arxiv.org/abs/2205.01308) [cs.CL]** |
|           | (or **[arXiv:2205.01308v1](https://arxiv.org/abs/2205.01308v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.01308Focus to learn more |





<h2 id="2022-05-04-3">3. Meta Learning for Natural Language Processing: A Survey
</h2>

Title: [Meta Learning for Natural Language Processing: A Survey](https://arxiv.org/abs/2205.01500)

Authors: [Hung-yi Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+H), [Shang-Wen Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+S), [Ngoc Thang Vu](https://arxiv.org/search/cs?searchtype=author&query=Vu%2C+N+T)

> Deep learning has been the mainstream technique in natural language processing (NLP) area. However, the techniques require many labeled data and are less generalizable across domains. Meta-learning is an arising field in machine learning studying approaches to learn better learning algorithms. Approaches aim at improving algorithms in various aspects, including data efficiency and generalizability. Efficacy of approaches has been shown in many NLP tasks, but there is no systematic survey of these approaches in NLP, which hinders more researchers from joining the field. Our goal with this survey paper is to offer researchers pointers to relevant meta-learning works in NLP and attract more attention from the NLP community to drive future innovation. This paper first introduces the general concepts of meta-learning and the common approaches. Then we summarize task construction settings and application of meta-learning for various NLP problems and review the development of meta-learning in NLP community.

| Comments: | Accepted by NAACL 2022                                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2205.01500](https://arxiv.org/abs/2205.01500) [cs.CL]** |
|           | (or **[arXiv:2205.01500v1](https://arxiv.org/abs/2205.01500v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.01500Focus to learn more |





<h2 id="2022-05-04-4">4. Learning to Transfer Prompts for Text Generation
</h2>

Title: [Learning to Transfer Prompts for Text Generation](https://arxiv.org/abs/2205.01543)

Authors: [Junyi Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+J), [Tianyi Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang%2C+T), [Jian-Yun Nie](https://arxiv.org/search/cs?searchtype=author&query=Nie%2C+J), [Ji-Rong Wen](https://arxiv.org/search/cs?searchtype=author&query=Wen%2C+J), [Wayne Xin Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+W+X)

> Pretrained language models (PLMs) have made remarkable progress in text generation tasks via fine-tuning. While, it is challenging to fine-tune PLMs in a data-scarce situation. Therefore, it is non-trivial to develop a general and lightweight model that can adapt to various text generation tasks based on PLMs. To fulfill this purpose, the recent prompt-based learning offers a potential solution. In this paper, we improve this technique and propose a novel prompt-based method (PTG) for text generation in a transferable setting. First, PTG learns a set of source prompts for various source generation tasks and then transfers these prompts as target prompts to perform target generation tasks. To consider both task- and instance-level information, we design an adaptive attention mechanism to derive the target prompts. For each data instance, PTG learns a specific target prompt by attending to highly relevant source prompts. In extensive experiments, PTG yields competitive or better results than fine-tuning methods. We release our source prompts as an open resource, where users can add or reuse them to improve new text generation tasks for future research. Code and data can be available at [this https URL](https://github.com/RUCAIBox/Transfer-Prompts-for-Text-Generation).

| Comments: | Accepted by NAACL 2022 main conference (Long Paper)          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2205.01543](https://arxiv.org/abs/2205.01543) [cs.CL]** |
|           | (or **[arXiv:2205.01543v1](https://arxiv.org/abs/2205.01543v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.01543Focus to learn more |





<h2 id="2022-05-04-5">5. Adaptable Adapters
</h2>

Title: [Adaptable Adapters](https://arxiv.org/abs/2205.01549)

Authors: [Nafise Sadat Moosavi](https://arxiv.org/search/cs?searchtype=author&query=Moosavi%2C+N+S), [Quentin Delfosse](https://arxiv.org/search/cs?searchtype=author&query=Delfosse%2C+Q), [Kristian Kersting](https://arxiv.org/search/cs?searchtype=author&query=Kersting%2C+K), [Iryna Gurevych](https://arxiv.org/search/cs?searchtype=author&query=Gurevych%2C+I)

> State-of-the-art pretrained NLP models contain a hundred million to trillion parameters. Adapters provide a parameter-efficient alternative for the full finetuning in which we can only finetune lightweight neural network layers on top of pretrained weights. Adapter layers are initialized randomly. However, existing work uses the same adapter architecture -- i.e., the same adapter layer on top of each layer of the pretrained model -- for every dataset, regardless of the properties of the dataset or the amount of available training data. In this work, we introduce adaptable adapters that contain (1) learning different activation functions for different layers and different input data, and (2) a learnable switch to select and only use the beneficial adapter layers. We show that adaptable adapters achieve on-par performances with the standard adapter architecture while using a considerably smaller number of adapter layers. In addition, we show that the selected adapter architecture by adaptable adapters transfers well across different data settings and similar tasks. We propose to use adaptable adapters for designing efficient and effective adapter architectures. The resulting adapters (a) contain about 50% of the learning parameters of the standard adapter and are therefore more efficient at training and inference, and require less storage space, and (b) achieve considerably higher performances in low-data settings.

| Comments: | Accepted at NAACL-2022 main conference                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2205.01549](https://arxiv.org/abs/2205.01549) [cs.CL]** |
|           | (or **[arXiv:2205.01549v1](https://arxiv.org/abs/2205.01549v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.01549Focus to learn more |





<h2 id="2022-05-04-6">6. Training Mixed-Domain Translation Models via Federated Learning
</h2>

Title: [Training Mixed-Domain Translation Models via Federated Learning](https://arxiv.org/abs/2205.01557)

Authors: [Peyman Passban](https://arxiv.org/search/cs?searchtype=author&query=Passban%2C+P), [Tanya Roosta](https://arxiv.org/search/cs?searchtype=author&query=Roosta%2C+T), [Rahul Gupta](https://arxiv.org/search/cs?searchtype=author&query=Gupta%2C+R), [Ankit Chadha](https://arxiv.org/search/cs?searchtype=author&query=Chadha%2C+A), [Clement Chung](https://arxiv.org/search/cs?searchtype=author&query=Chung%2C+C)

> Training mixed-domain translation models is a complex task that demands tailored architectures and costly data preparation techniques. In this work, we leverage federated learning (FL) in order to tackle the problem. Our investigation demonstrates that with slight modifications in the training process, neural machine translation (NMT) engines can be easily adapted when an FL-based aggregation is applied to fuse different domains. Experimental results also show that engines built via FL are able to perform on par with state-of-the-art baselines that rely on centralized training techniques. We evaluate our hypothesis in the presence of five datasets with different sizes, from different domains, to translate from German into English and discuss how FL and NMT can mutually benefit from each other. In addition to providing benchmarking results on the union of FL and NMT, we also propose a novel technique to dynamically control the communication bandwidth by selecting impactful parameters during FL updates. This is a significant achievement considering the large size of NMT engines that need to be exchanged between FL parties.

| Comments: | accepted at NAACL 2022 (main conference)                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2205.01557](https://arxiv.org/abs/2205.01557) [cs.CL]** |
|           | (or **[arXiv:2205.01557v1](https://arxiv.org/abs/2205.01557v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.01557Focus to learn more |





<h2 id="2022-05-04-7">7. OmniKnight: Multilingual Neural Machine Translation with Language-Specific Self-Distillation
</h2>

Title: [OmniKnight: Multilingual Neural Machine Translation with Language-Specific Self-Distillation](https://arxiv.org/abs/2205.01620)

Authors: [Yichong Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+Y), [Xiaocheng Feng](https://arxiv.org/search/cs?searchtype=author&query=Feng%2C+X), [Xinwei Geng](https://arxiv.org/search/cs?searchtype=author&query=Geng%2C+X), [Bing Qin](https://arxiv.org/search/cs?searchtype=author&query=Qin%2C+B)

> Although all-in-one-model multilingual neural machine translation (MNMT) has achieved remarkable progress in recent years, its selected best overall checkpoint fails to achieve the best performance simultaneously in all language pairs. It is because that the best checkpoints for each individual language pair (i.e., language-specific best checkpoints) scatter in different epochs. In this paper, we present a novel training strategy dubbed Language-Specific Self-Distillation (LSSD) for bridging the gap between language-specific best checkpoints and the overall best checkpoint. In detail, we regard each language-specific best checkpoint as a teacher to distill the overall best checkpoint. Moreover, we systematically explore three variants of our LSSD, which perform distillation statically, selectively, and adaptively. Experimental results on two widely-used benchmarks show that LSSD obtains consistent improvements towards all language pairs and achieves the state-of-the-art

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2205.01620](https://arxiv.org/abs/2205.01620) [cs.CL]** |
|           | (or **[arXiv:2205.01620v1](https://arxiv.org/abs/2205.01620v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.01620Focus to learn more |





# 2022-05-03

[Return to Index](#Index)



<h2 id="2022-05-03-1">1. Multimodal Representation Learning With Text and Images
</h2>

Title: [Multimodal Representation Learning With Text and Images](https://arxiv.org/abs/2205.00142)

Authors:[Aishwarya Jayagopal](https://arxiv.org/search/cs?searchtype=author&query=Jayagopal%2C+A), [Ankireddy Monica Aiswarya](https://arxiv.org/search/cs?searchtype=author&query=Aiswarya%2C+A+M), [Ankita Garg](https://arxiv.org/search/cs?searchtype=author&query=Garg%2C+A), [Srinivasan Kolumam Nandakumar](https://arxiv.org/search/cs?searchtype=author&query=Nandakumar%2C+S+K)

> In recent years, multimodal AI has seen an upward trend as researchers are integrating data of different types such as text, images, speech into modelling to get the best results. This project leverages multimodal AI and matrix factorization techniques for representation learning, on text and image data simultaneously, thereby employing the widely used techniques of Natural Language Processing (NLP) and Computer Vision. The learnt representations are evaluated using downstream classification and regression tasks. The methodology adopted can be extended beyond the scope of this project as it uses Auto-Encoders for unsupervised representation learning.

| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL); Computer Vision and Pattern Recognition (cs.CV) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2205.00142](https://arxiv.org/abs/2205.00142) [cs.LG]** |
|           | (or **[arXiv:2205.00142v1](https://arxiv.org/abs/2205.00142v1) [cs.LG]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.00142Focus to learn more |





<h2 id="2022-05-03-2">2. EasyNLP: A Comprehensive and Easy-to-use Toolkit for Natural Language Processing
</h2>

Title: [EasyNLP: A Comprehensive and Easy-to-use Toolkit for Natural Language Processing](https://arxiv.org/abs/2205.00258)

Authors:[Chengyu Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+C), [Minghui Qiu](https://arxiv.org/search/cs?searchtype=author&query=Qiu%2C+M), [Taolin Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+T), [Tingting Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+T), [Lei Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+L), [Jianing Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+J), [Ming Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+M), [Jun Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+J), [Wei Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+W)

> The success of Pre-Trained Models (PTMs) has reshaped the development of Natural Language Processing (NLP). Yet, it is not easy to obtain high-performing models and deploy them online for industrial practitioners. To bridge this gap, EasyNLP is designed to make it easy to build NLP applications, which supports a comprehensive suite of NLP algorithms. It further features knowledge-enhanced pre-training, knowledge distillation and few-shot learning functionalities for large-scale PTMs, and provides a unified framework of model training, inference and deployment for real-world applications. Currently, EasyNLP has powered over ten business units within Alibaba Group and is seamlessly integrated to the Platform of AI (PAI) products on Alibaba Cloud. The source code of our EasyNLP toolkit is released at GitHub ([this https URL](https://github.com/alibaba/EasyNLP)).

| Comments: | 8 pages                                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2205.00258](https://arxiv.org/abs/2205.00258) [cs.CL]** |
|           | (or **[arXiv:2205.00258v1](https://arxiv.org/abs/2205.00258v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.00258Focus to learn more |





<h2 id="2022-05-03-3">3. AdapterBias: Parameter-efficient Token-dependent Representation Shift for Adapters in NLP Tasks
</h2>

Title: [AdapterBias: Parameter-efficient Token-dependent Representation Shift for Adapters in NLP Tasks](https://arxiv.org/abs/2205.00305)

Authors:[Chin-Lun Fu](https://arxiv.org/search/cs?searchtype=author&query=Fu%2C+C), [Zih-Ching Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Z), [Yun-Ru Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+Y), [Hung-yi Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee%2C+H)

> Transformer-based pre-trained models with millions of parameters require large storage. Recent approaches tackle this shortcoming by training adapters, but these approaches still require a relatively large number of parameters. In this study, AdapterBias, a surprisingly simple yet effective adapter architecture, is proposed. AdapterBias adds a token-dependent shift to the hidden output of transformer layers to adapt to downstream tasks with only a vector and a linear layer. Extensive experiments are conducted to demonstrate the effectiveness of AdapterBias. The experiments show that our proposed method can dramatically reduce the trainable parameters compared to the previous works with a minimal decrease in task performances compared with fine-tuned pre-trained models. We further find that AdapterBias automatically learns to assign more significant representation shifts to the tokens related to the task in consideration.

| Comments: | The first two authors contributed equally. This paper will be published in Findings of NAACL 2022 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2205.00305](https://arxiv.org/abs/2205.00305) [cs.CL]** |
|           | (or **[arXiv:2205.00305v1](https://arxiv.org/abs/2205.00305v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.00305Focus to learn more |





<h2 id="2022-05-03-4">4. Nearest Neighbor Knowledge Distillation for Neural Machine Translation
</h2>

Title: [Nearest Neighbor Knowledge Distillation for Neural Machine Translation](https://arxiv.org/abs/2205.00479)

Authors:[Zhixian Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Z), [Renliang Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+R), [Xiaojun Wan](https://arxiv.org/search/cs?searchtype=author&query=Wan%2C+X)

> k-nearest-neighbor machine translation (NN-MT), proposed by Khandelwal et al. (2021), has achieved many state-of-the-art results in machine translation tasks. Although effective, NN-MT requires conducting NN searches through the large datastore for each decoding step during inference, prohibitively increasing the decoding cost and thus leading to the difficulty for the deployment in real-world applications. In this paper, we propose to move the time-consuming NN search forward to the preprocessing phase, and then introduce Nearest Neighbor Knowledge Distillation (NN-KD) that trains the base NMT model to directly learn the knowledge of NN. Distilling knowledge retrieved by NN can encourage the NMT model to take more reasonable target tokens into consideration, thus addressing the overcorrection problem. Extensive experimental results show that, the proposed method achieves consistent improvement over the state-of-the-art baselines including NN-MT, while maintaining the same training and decoding speed as the standard NMT model.

| Comments: | Accepted to NAACL 2022 Main Conference                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2205.00479](https://arxiv.org/abs/2205.00479) [cs.CL]** |
|           | (or **[arXiv:2205.00479v1](https://arxiv.org/abs/2205.00479v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.00479Focus to learn more |





<h2 id="2022-05-03-5">5. Bilingual End-to-End ASR with Byte-Level Subwords
</h2>

Title: [Bilingual End-to-End ASR with Byte-Level Subwords](https://arxiv.org/abs/2205.00485)

Authors:[Liuhui Deng](https://arxiv.org/search/cs?searchtype=author&query=Deng%2C+L), [Roger Hsiao](https://arxiv.org/search/cs?searchtype=author&query=Hsiao%2C+R), [Arnab Ghoshal](https://arxiv.org/search/cs?searchtype=author&query=Ghoshal%2C+A)

> In this paper, we investigate how the output representation of an end-to-end neural network affects multilingual automatic speech recognition (ASR). We study different representations including character-level, byte-level, byte pair encoding (BPE), and byte-level byte pair encoding (BBPE) representations, and analyze their strengths and weaknesses. We focus on developing a single end-to-end model to support utterance-based bilingual ASR, where speakers do not alternate between two languages in a single utterance but may change languages across utterances. We conduct our experiments on English and Mandarin dictation tasks, and we find that BBPE with penalty schemes can improve utterance-based bilingual ASR performance by 2% to 5% relative even with smaller number of outputs and fewer parameters. We conclude with analysis that indicates directions for further improving multilingual ASR.

| Comments: | 5 pages, to be published in IEEE ICASSP 2022                 |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2205.00485](https://arxiv.org/abs/2205.00485) [cs.CL]** |
|           | (or **[arXiv:2205.00485v1](https://arxiv.org/abs/2205.00485v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.00485Focus to learn more |





<h2 id="2022-05-03-6">6. Debiased Contrastive Learning of Unsupervised Sentence Representations
</h2>

Title: [Debiased Contrastive Learning of Unsupervised Sentence Representations](https://arxiv.org/abs/2205.00656)

Authors:[Kun Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+K), [Beichen Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+B), [Wayne Xin Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+W+X), [Ji-Rong Wen](https://arxiv.org/search/cs?searchtype=author&query=Wen%2C+J)

> Recently, contrastive learning has been shown to be effective in improving pre-trained language models (PLM) to derive high-quality sentence representations. It aims to pull close positive examples to enhance the alignment while push apart irrelevant negatives for the uniformity of the whole representation space. However, previous works mostly adopt in-batch negatives or sample from training data at random. Such a way may cause the sampling bias that improper negatives (e.g. false negatives and anisotropy representations) are used to learn sentence representations, which will hurt the uniformity of the representation space. To address it, we present a new framework \textbf{DCLR} (\underline{D}ebiased \underline{C}ontrastive \underline{L}earning of unsupervised sentence \underline{R}epresentations) to alleviate the influence of these improper negatives. In DCLR, we design an instance weighting method to punish false negatives and generate noise-based negatives to guarantee the uniformity of the representation space. Experiments on seven semantic textual similarity tasks show that our approach is more effective than competitive baselines. Our code and data are publicly available at the link: \textcolor{blue}{\url{[this https URL](https://github.com/RUCAIBox/DCLR)}}.

| Comments: | 11 pages, accepted by ACL 2022 main conference               |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2205.00656](https://arxiv.org/abs/2205.00656) [cs.CL]** |
|           | (or **[arXiv:2205.00656v1](https://arxiv.org/abs/2205.00656v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.00656Focus to learn more |





<h2 id="2022-05-03-7">7. The Implicit Length Bias of Label Smoothing on Beam Search Decoding
</h2>

Title: [The Implicit Length Bias of Label Smoothing on Beam Search Decoding](https://arxiv.org/abs/2205.00659)

Authors:[Bowen Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+B), [Pidong Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+P), [Yuan Cao](https://arxiv.org/search/cs?searchtype=author&query=Cao%2C+Y)

> Label smoothing is ubiquitously applied in Neural Machine Translation (NMT) training. While label smoothing offers a desired regularization effect during model training, in this paper we demonstrate that it nevertheless introduces length biases in the beam search decoding procedure. Our analysis shows that label smoothing implicitly applies a length penalty term to output sequence, causing a bias towards shorter translations. We also show that for a model fully optimized with label smoothing, translation length is implicitly upper bounded by a fixed constant independent of input. We verify our theory by applying a simple rectification function at inference time to restore the unbiased distributions from the label-smoothed model predictions. This rectification method led to consistent quality improvements on WMT English-German, English-French, English-Czech and English-Chinese tasks, up to +0.3 BLEU at beam size 4 and +2.8 BLEU at beam size 200.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2205.00659](https://arxiv.org/abs/2205.00659) [cs.CL]** |
|           | (or **[arXiv:2205.00659v1](https://arxiv.org/abs/2205.00659v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.00659Focus to learn more |





<h2 id="2022-05-03-8">8. Quality-Aware Decoding for Neural Machine Translation
</h2>

Title: [Quality-Aware Decoding for Neural Machine Translation](https://arxiv.org/abs/2205.00978)

Authors:[Patrick Fernandes](https://arxiv.org/search/cs?searchtype=author&query=Fernandes%2C+P), [António Farinhas](https://arxiv.org/search/cs?searchtype=author&query=Farinhas%2C+A), [Ricardo Rei](https://arxiv.org/search/cs?searchtype=author&query=Rei%2C+R), [José G. C. de Souza](https://arxiv.org/search/cs?searchtype=author&query=de+Souza%2C+J+G+C), [Perez Ogayo](https://arxiv.org/search/cs?searchtype=author&query=Ogayo%2C+P), [Graham Neubig](https://arxiv.org/search/cs?searchtype=author&query=Neubig%2C+G), [André F. T. Martins](https://arxiv.org/search/cs?searchtype=author&query=Martins%2C+A+F+T)

> Despite the progress in machine translation quality estimation and evaluation in the last years, decoding in neural machine translation (NMT) is mostly oblivious to this and centers around finding the most probable translation according to the model (MAP decoding), approximated with beam search. In this paper, we bring together these two lines of research and propose quality-aware decoding for NMT, by leveraging recent breakthroughs in reference-free and reference-based MT evaluation through various inference methods like N-best reranking and minimum Bayes risk decoding. We perform an extensive comparison of various possible candidate generation and ranking methods across four datasets and two model classes and find that quality-aware decoding consistently outperforms MAP-based decoding according both to state-of-the-art automatic metrics (COMET and BLEURT) and to human assessments. Our code is available at [this https URL](https://github.com/deep-spin/qaware-decode).

| Comments: | NAACL2022                                                    |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2205.00978](https://arxiv.org/abs/2205.00978) [cs.CL]** |
|           | (or **[arXiv:2205.00978v1](https://arxiv.org/abs/2205.00978v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.00978Focus to learn more |





<h2 id="2022-05-03-9">9. OPT: Open Pre-trained Transformer Language Models
</h2>

Title: [OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/abs/2205.01068)

Authors:[Susan Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+S), [Stephen Roller](https://arxiv.org/search/cs?searchtype=author&query=Roller%2C+S), [Naman Goyal](https://arxiv.org/search/cs?searchtype=author&query=Goyal%2C+N), [Mikel Artetxe](https://arxiv.org/search/cs?searchtype=author&query=Artetxe%2C+M), [Moya Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+M), [Shuohui Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+S), [Christopher Dewan](https://arxiv.org/search/cs?searchtype=author&query=Dewan%2C+C), [Mona Diab](https://arxiv.org/search/cs?searchtype=author&query=Diab%2C+M), [Xian Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+X), [Xi Victoria Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+X+V), [Todor Mihaylov](https://arxiv.org/search/cs?searchtype=author&query=Mihaylov%2C+T), [Myle Ott](https://arxiv.org/search/cs?searchtype=author&query=Ott%2C+M), [Sam Shleifer](https://arxiv.org/search/cs?searchtype=author&query=Shleifer%2C+S), [Kurt Shuster](https://arxiv.org/search/cs?searchtype=author&query=Shuster%2C+K), [Daniel Simig](https://arxiv.org/search/cs?searchtype=author&query=Simig%2C+D), [Punit Singh Koura](https://arxiv.org/search/cs?searchtype=author&query=Koura%2C+P+S), [Anjali Sridhar](https://arxiv.org/search/cs?searchtype=author&query=Sridhar%2C+A), [Tianlu Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+T), [Luke Zettlemoyer](https://arxiv.org/search/cs?searchtype=author&query=Zettlemoyer%2C+L)

> Large language models, which are often trained for hundreds of thousands of compute days, have shown remarkable capabilities for zero- and few-shot learning. Given their computational cost, these models are difficult to replicate without significant capital. For the few that are available through APIs, no access is granted to the full model weights, making them difficult to study. We present Open Pre-trained Transformers (OPT), a suite of decoder-only pre-trained transformers ranging from 125M to 175B parameters, which we aim to fully and responsibly share with interested researchers. We show that OPT-175B is comparable to GPT-3, while requiring only 1/7th the carbon footprint to develop. We are also releasing our logbook detailing the infrastructure challenges we faced, along with code for experimenting with all of the released models.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2205.01068](https://arxiv.org/abs/2205.01068) [cs.CL]** |
|           | (or **[arXiv:2205.01068v1](https://arxiv.org/abs/2205.01068v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.01068Focus to learn more |





<h2 id="2022-05-03-10">10. Wav2Seq: Pre-training Speech-to-Text Encoder-Decoder Models Using Pseudo Languages
</h2>

Title: [Wav2Seq: Pre-training Speech-to-Text Encoder-Decoder Models Using Pseudo Languages](https://arxiv.org/abs/2205.01086)

Authors:[Felix Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+F), [Kwangyoun Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+K), [Shinji Watanabe](https://arxiv.org/search/cs?searchtype=author&query=Watanabe%2C+S), [Kyu Han](https://arxiv.org/search/cs?searchtype=author&query=Han%2C+K), [Ryan McDonald](https://arxiv.org/search/cs?searchtype=author&query=McDonald%2C+R), [Kilian Q. Weinberger](https://arxiv.org/search/cs?searchtype=author&query=Weinberger%2C+K+Q), [Yoav Artzi](https://arxiv.org/search/cs?searchtype=author&query=Artzi%2C+Y)

> We introduce Wav2Seq, the first self-supervised approach to pre-train both parts of encoder-decoder models for speech data. We induce a pseudo language as a compact discrete representation, and formulate a self-supervised pseudo speech recognition task -- transcribing audio inputs into pseudo subword sequences. This process stands on its own, or can be applied as low-cost second-stage pre-training. We experiment with automatic speech recognition (ASR), spoken named entity recognition, and speech-to-text translation. We set new state-of-the-art results for end-to-end spoken named entity recognition, and show consistent improvements on 20 language pairs for speech-to-text translation, even when competing methods use additional text data for training. Finally, on ASR, our approach enables encoder-decoder methods to benefit from pre-training for all parts of the network, and shows comparable performance to highly optimized recent methods.

| Comments: | Code available at [this https URL](https://github.com/asappresearch/wav2seq) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2205.01086](https://arxiv.org/abs/2205.01086) [cs.CL]** |
|           | (or **[arXiv:2205.01086v1](https://arxiv.org/abs/2205.01086v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.01086Focus to learn more |





# 2022-05-02

[Return to Index](#Index)



<h2 id="2022-05-02-1">1. Vision-Language Pre-Training for Boosting Scene Text Detectors
</h2>

Title: [Vision-Language Pre-Training for Boosting Scene Text Detectors](https://arxiv.org/abs/2204.13867)

Authors: [Sibo Song](https://arxiv.org/search/cs?searchtype=author&query=Song%2C+S), [Jianqiang Wan](https://arxiv.org/search/cs?searchtype=author&query=Wan%2C+J), [Zhibo Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Z), [Jun Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang%2C+J), [Wenqing Cheng](https://arxiv.org/search/cs?searchtype=author&query=Cheng%2C+W), [Xiang Bai](https://arxiv.org/search/cs?searchtype=author&query=Bai%2C+X), [Cong Yao](https://arxiv.org/search/cs?searchtype=author&query=Yao%2C+C)

> Recently, vision-language joint representation learning has proven to be highly effective in various scenarios. In this paper, we specifically adapt vision-language joint learning for scene text detection, a task that intrinsically involves cross-modal interaction between the two modalities: vision and language, since text is the written form of language. Concretely, we propose to learn contextualized, joint representations through vision-language pre-training, for the sake of enhancing the performance of scene text detectors. Towards this end, we devise a pre-training architecture with an image encoder, a text encoder and a cross-modal encoder, as well as three pretext tasks: image-text contrastive learning (ITC), masked language modeling (MLM) and word-in-image prediction (WIP). The pre-trained model is able to produce more informative representations with richer semantics, which could readily benefit existing scene text detectors (such as EAST and PSENet) in the down-stream text detection task. Extensive experiments on standard benchmarks demonstrate that the proposed paradigm can significantly improve the performance of various representative text detectors, outperforming previous pre-training approaches. The code and pre-trained models will be publicly released.

| Comments: | Accepted by CVPR 2022                                        |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2204.13867](https://arxiv.org/abs/2204.13867) [cs.CV]** |
|           | (or **[arXiv:2204.13867v1](https://arxiv.org/abs/2204.13867v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.13867Focus to learn more |





<h2 id="2022-05-02-2">2. Polyglot Prompt: Multilingual Multitask PrompTraining
</h2>

Title: [Polyglot Prompt: Multilingual Multitask PrompTraining](https://arxiv.org/abs/2204.14264)

Authors: [Jinlan Fu](https://arxiv.org/search/cs?searchtype=author&query=Fu%2C+J), [See-Kiong Ng](https://arxiv.org/search/cs?searchtype=author&query=Ng%2C+S), [Pengfei Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+P)

> This paper aims for a potential architectural breakthrough for multilingual learning and asks: could different tasks from different languages be modeled in a monolithic framework (without any task/language-specific module)? The benefit of achieving this is not only that systems trained on low resources scenario can be assisted by more other languages and tasks, but opening new doors for future multilingual research. We approach this goal by developing a learning framework Polyglot Prompt, where prompting methods are introduced to learn a unified semantic space for different languages and tasks after proper multilingual prompt engineering. Experimentally, we perform a comprehensive evaluation on 6 tasks (topic classification, sentiment classification, named entity recognition, question answering, natural language inference, summarization), 24 datasets, and 49 languages, which shows the efficacy of multilingual multitask prompting training and suggests several interesting observations. e.g., English prompts are polyglots since directly applying them to task samples in other languages could result in a better improvement. We also present an interpretable multilingual evaluation methodology and show how the proposed framework, multilingual multitask prompt training, works. We release all datasets prompted in the best setting and will release our code soon.

| Comments: | 19 pages, 64 figures                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2204.14264](https://arxiv.org/abs/2204.14264) [cs.CL]** |
|           | (or **[arXiv:2204.14264v1](https://arxiv.org/abs/2204.14264v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.14264Focus to learn more |





<h2 id="2022-05-02-3">3. How Robust is Neural Machine Translation to Language Imbalance in Multilingual Tokenizer Training?
</h2>

Title: [How Robust is Neural Machine Translation to Language Imbalance in Multilingual Tokenizer Training?](https://arxiv.org/abs/2204.14268)

Authors: [Shiyue Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+S), [Vishrav Chaudhary](https://arxiv.org/search/cs?searchtype=author&query=Chaudhary%2C+V), [Naman Goyal](https://arxiv.org/search/cs?searchtype=author&query=Goyal%2C+N), [James Cross](https://arxiv.org/search/cs?searchtype=author&query=Cross%2C+J), [Guillaume Wenzek](https://arxiv.org/search/cs?searchtype=author&query=Wenzek%2C+G), [Mohit Bansal](https://arxiv.org/search/cs?searchtype=author&query=Bansal%2C+M), [Francisco Guzman](https://arxiv.org/search/cs?searchtype=author&query=Guzman%2C+F)

> A multilingual tokenizer is a fundamental component of multilingual neural machine translation. It is trained from a multilingual corpus. Since a skewed data distribution is considered to be harmful, a sampling strategy is usually used to balance languages in the corpus. However, few works have systematically answered how language imbalance in tokenizer training affects downstream performance. In this work, we analyze how translation performance changes as the data ratios among languages vary in the tokenizer training corpus. We find that while relatively better performance is often observed when languages are more equally sampled, the downstream performance is more robust to language imbalance than we usually expected. Two features, UNK rate and closeness to the character level, can warn of poor downstream performance before performing the task. We also distinguish language sampling for tokenizer training from sampling for model training and show that the model is more sensitive to the latter.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2204.14268](https://arxiv.org/abs/2204.14268) [cs.CL]** |
|           | (or **[arXiv:2204.14268v1](https://arxiv.org/abs/2204.14268v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.14268Focus to learn more |





# 2022-04-29

[Return to Index](#Index)



<h2 id="2022-04-29-1">1. UniTE: Unified Translation Evaluation
</h2>


Title: [UniTE: Unified Translation Evaluation](https://arxiv.org/abs/2204.13346)

Authors: [Yu Wan](https://arxiv.org/search/cs?searchtype=author&query=Wan%2C+Y), [Dayiheng Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+D), [Baosong Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+B), [Haibo Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+H), [Boxing Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+B), [Derek F. Wong](https://arxiv.org/search/cs?searchtype=author&query=Wong%2C+D+F), [Lidia S. Chao](https://arxiv.org/search/cs?searchtype=author&query=Chao%2C+L+S)

> Translation quality evaluation plays a crucial role in machine translation. According to the input format, it is mainly separated into three tasks, i.e., reference-only, source-only and source-reference-combined. Recent methods, despite their promising results, are specifically designed and optimized on one of them. This limits the convenience of these methods, and overlooks the commonalities among tasks. In this paper, we propose UniTE, which is the first unified framework engaged with abilities to handle all three evaluation tasks. Concretely, we propose monotonic regional attention to control the interaction among input segments, and unified pretraining to better adapt multi-task learning. We testify our framework on WMT 2019 Metrics and WMT 2020 Quality Estimation benchmarks. Extensive analyses show that our \textit{single model} can universally surpass various state-of-the-art or winner methods across tasks. Both source code and associated models are available at [this https URL](https://github.com/NLP2CT/UniTE).

| Comments: | ACL2022                                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2204.13346](https://arxiv.org/abs/2204.13346) [cs.CL]** |
|           | (or **[arXiv:2204.13346v1](https://arxiv.org/abs/2204.13346v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2204.13346Focus to learn more |

