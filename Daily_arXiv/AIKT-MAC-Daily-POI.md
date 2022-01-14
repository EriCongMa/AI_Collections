# MA C.'s Daily Paper Of Interest - January, 2022

# Index


- [2022-01-14](#2022-01-14)

  - [1. Towards Automated Error Analysis: Learning to Characterize Errors](#2022-01-14)

- [2022-01-13](#2022-01-13)

  - [1. PromptBERT: Improving BERT Sentence Embeddings with Prompts](#2022-01-13-1)
  - [2. How Does Data Corruption Affect Natural Language Understanding Models? A Study on GLUE datasets](#2022-01-13-2)

- [2022-01-12](#2022-01-12)

  - [1. Uni-EDEN: Universal Encoder-Decoder Network by Multi-Granular Vision-Language Pre-training](#2022-01-12-1)
  - [2. CVSS Corpus and Massively Multilingual Speech-to-Speech Translation](#2022-01-12-2)
  - [3. Quantifying Robustness to Adversarial Word Substitutions](#2022-01-12-3)

- [2022-01-11](#2022-01-11)

  - [1. Towards the Next 1000 Languages in Multilingual Machine Translation: Exploring the Synergy Between Supervised and Self-Supervised Learning](#2022-01-11-1)
  - [2. Black-Box Tuning for Language-Model-as-a-Service](#2022-01-11-2)
  - [3. SCROLLS: Standardized CompaRison Over Long Language Sequences](#2022-01-11-3)

- [2022-01-10](#2022-01-10)

  - [1. Automatic Speech Recognition Datasets in Cantonese Language: A Survey and a New Dataset](#2022-01-10-1)
  - [2. Semantic-based Data Augmentation for Math Word Problems](#2022-01-10-2)
  - [3. Repairing Adversarial Texts through Perturbation](#2022-01-10-3)
  - [4. Code-Switching Text Augmentation for Multilingual Speech Processing](#2022-01-10-4)

- [2022-01-07](#2022-01-07)
  - [1. Compact Bidirectional Transformer for Image Captioning](#2022-01-07-1)
  - [2. Self-Training Vision Language BERTs with a Unified Conditional Model](#2022-01-07-2)
  - [3. Phrase-level Adversarial Example Generation for Neural Machine Translation](#2022-01-07-3)
- [2022-01-06](#2022-01-06)

  - [1. All You Need In Sign Language Production](#2022-01-06-1)
  - [2. SMDT: Selective Memory-Augmented Neural Document Translation](#2022-01-06-2)
- [2022-01-05](#2022-01-05)

  - [1. Interactive Attention AI to translate low light photos to captions for night scene understanding in women safety](#2022-01-05-1)
  - [2. StyleM: Stylized Metrics for Image Captioning Built with Contrastive N-grams](#2022-01-05-2)
- [2022-01-04](#2022-01-04)

  - [1. How do lexical semantics affect translation? An empirical study](#2022-01-04-1)
  - [2. Which Student is Best? A Comprehensive Knowledge Distillation Exam for Task-Specific BERT Models](#2022-01-04-2)
  - [3. Robust Natural Language Processing: Recent Advances, Challenges, and Future Directions](#2022-01-04-3)
- [2022-01-03](#2022-01-03)

  - [1. ERNIE-ViLG: Unified Generative Pre-training for Bidirectional Vision-Language Generation](#2022-01-03-1)
  - [2. Deconfounded Visual Grounding](#2022-01-03-2)
  - [3. Materialized Knowledge Bases from Commonsense Transformers](#2022-01-03-3)
  - [4. ViNMT: Neural Machine Translation Tookit](#2022-01-03-4)
- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MAC-Daily-POI-index.md)



# 2022-01-14

[Return to Index](#Index)



<h2 id="2022-01-14-1">1. Towards Automated Error Analysis: Learning to Characterize Errors
</h2>

Title: [Towards Automated Error Analysis: Learning to Characterize Errors](https://arxiv.org/abs/2201.05017)

Authors: [Tong Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+T), [Shivang Singh](https://arxiv.org/search/cs?searchtype=author&query=Singh%2C+S), [Raymond J. Mooney](https://arxiv.org/search/cs?searchtype=author&query=Mooney%2C+R+J)

> Characterizing the patterns of errors that a system makes helps researchers focus future development on increasing its accuracy and robustness. We propose a novel form of "meta learning" that automatically learns interpretable rules that characterize the types of errors that a system makes, and demonstrate these rules' ability to help understand and improve two NLP systems. Our approach works by collecting error cases on validation data, extracting meta-features describing these samples, and finally learning rules that characterize errors using these features. We apply our approach to VilBERT, for Visual Question Answering, and RoBERTa, for Common Sense Question Answering. Our system learns interpretable rules that provide insights into systemic errors these systems make on the given tasks. Using these insights, we are also able to "close the loop" and modestly improve performance of these systems.

| Comments: | 12 pages, 11 figures                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2201.05017](https://arxiv.org/abs/2201.05017) [cs.CL]** |
|           | (or **[arXiv:2201.05017v1](https://arxiv.org/abs/2201.05017v1) [cs.CL]** for this version) |









# 2022-01-13

[Return to Index](#Index)



<h2 id="2022-01-13-1">1. PromptBERT: Improving BERT Sentence Embeddings with Prompts
</h2>

Title: [PromptBERT: Improving BERT Sentence Embeddings with Prompts](https://arxiv.org/abs/2201.04337)

Authors: [Ting Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+T), [Shaohan Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+S), [Zihan Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Z), [Deqing Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+D), [Fuzhen Zhuang](https://arxiv.org/search/cs?searchtype=author&query=Zhuang%2C+F), [Furu Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+F), [Haizhen Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+H), [Liangjie Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+L), [Qi Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Q)

> The poor performance of the original BERT for sentence semantic similarity has been widely discussed in previous works. We find that unsatisfactory performance is mainly due to the static token embeddings biases and the ineffective BERT layers, rather than the high cosine similarity of the sentence embeddings. To this end, we propose a prompt based sentence embeddings method which can reduce token embeddings biases and make the original BERT layers more effective. By reformulating the sentence embeddings task as the fillin-the-blanks problem, our method significantly improves the performance of original BERT. We discuss two prompt representing methods and three prompt searching methods for prompt based sentence embeddings. Moreover, we propose a novel unsupervised training objective by the technology of template denoising, which substantially shortens the performance gap between the supervised and unsupervised setting. For experiments, we evaluate our method on both non fine-tuned and fine-tuned settings. Even a non fine-tuned method can outperform the fine-tuned methods like unsupervised ConSERT on STS tasks. Our fine-tuned method outperforms the state-of-the-art method SimCSE in both unsupervised and supervised settings. Compared to SimCSE, we achieve 2.29 and 2.58 points improvements on BERT and RoBERTa respectively under the unsupervised setting.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2201.04337](https://arxiv.org/abs/2201.04337) [cs.CL]** |
|           | (or **[arXiv:2201.04337v1](https://arxiv.org/abs/2201.04337v1) [cs.CL]** for this version) |





<h2 id="2022-01-13-2">2. How Does Data Corruption Affect Natural Language Understanding Models? A Study on GLUE datasets
</h2>

Title: [How Does Data Corruption Affect Natural Language Understanding Models? A Study on GLUE datasets](https://arxiv.org/abs/2201.04467)

Authors: [Aarne Talman](https://arxiv.org/search/cs?searchtype=author&query=Talman%2C+A), [Marianna Apidianaki](https://arxiv.org/search/cs?searchtype=author&query=Apidianaki%2C+M), [Stergios Chatzikyriakidis](https://arxiv.org/search/cs?searchtype=author&query=Chatzikyriakidis%2C+S), [Jörg Tiedemann](https://arxiv.org/search/cs?searchtype=author&query=Tiedemann%2C+J)

> A central question in natural language understanding (NLU) research is whether high performance demonstrates the models' strong reasoning capabilities. We present an extensive series of controlled experiments where pre-trained language models are exposed to data that have undergone specific corruption transformations. The transformations involve removing instances of specific word classes and often lead to non-sensical sentences. Our results show that performance remains high for most GLUE tasks when the models are fine-tuned or tested on corrupted data, suggesting that the models leverage other cues for prediction even in non-sensical contexts. Our proposed data transformations can be used as a diagnostic tool for assessing the extent to which a specific dataset constitutes a proper testbed for evaluating models' language understanding capabilities.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2201.04467](https://arxiv.org/abs/2201.04467) [cs.CL]** |
|           | (or **[arXiv:2201.04467v1](https://arxiv.org/abs/2201.04467v1) [cs.CL]** for this version) |





# 2022-01-12

[Return to Index](#Index)



<h2 id="2022-01-12-1">1. Uni-EDEN: Universal Encoder-Decoder Network by Multi-Granular Vision-Language Pre-training
</h2>

Title:  [Uni-EDEN: Universal Encoder-Decoder Network by Multi-Granular Vision-Language Pre-training](https://arxiv.org/abs/2201.04026)

Authors: [Yehao Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Y), [Jiahao Fan](https://arxiv.org/search/cs?searchtype=author&query=Fan%2C+J), [Yingwei Pan](https://arxiv.org/search/cs?searchtype=author&query=Pan%2C+Y), [Ting Yao](https://arxiv.org/search/cs?searchtype=author&query=Yao%2C+T), [Weiyao Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+W), [Tao Mei](https://arxiv.org/search/cs?searchtype=author&query=Mei%2C+T)

> Vision-language pre-training has been an emerging and fast-developing research topic, which transfers multi-modal knowledge from rich-resource pre-training task to limited-resource downstream tasks. Unlike existing works that predominantly learn a single generic encoder, we present a pre-trainable Universal Encoder-DEcoder Network (Uni-EDEN) to facilitate both vision-language perception (e.g., visual question answering) and generation (e.g., image captioning). Uni-EDEN is a two-stream Transformer based structure, consisting of three modules: object and sentence encoders that separately learns the representations of each modality, and sentence decoder that enables both multi-modal reasoning and sentence generation via inter-modal interaction. Considering that the linguistic representations of each image can span different granularities in this hierarchy including, from simple to comprehensive, individual label, a phrase, and a natural sentence, we pre-train Uni-EDEN through multi-granular vision-language proxy tasks: Masked Object Classification (MOC), Masked Region Phrase Generation (MRPG), Image-Sentence Matching (ISM), and Masked Sentence Generation (MSG). In this way, Uni-EDEN is endowed with the power of both multi-modal representation extraction and language modeling. Extensive experiments demonstrate the compelling generalizability of Uni-EDEN by fine-tuning it to four vision-language perception and generation downstream tasks.

| Comments: | ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Multimedia (cs.MM) |
| Cite as:  | **[arXiv:2201.04026](https://arxiv.org/abs/2201.04026) [cs.CV]** |
|           | (or **[arXiv:2201.04026v1](https://arxiv.org/abs/2201.04026v1) [cs.CV]** for this version) |





<h2 id="2022-01-12-2">2. CVSS Corpus and Massively Multilingual Speech-to-Speech Translation
</h2>

Title:  [CVSS Corpus and Massively Multilingual Speech-to-Speech Translation](https://arxiv.org/abs/2201.03713)

Authors: [Ye Jia](https://arxiv.org/search/cs?searchtype=author&query=Jia%2C+Y), [Michelle Tadmor Ramanovich](https://arxiv.org/search/cs?searchtype=author&query=Ramanovich%2C+M+T), [Quan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Q), [Heiga Zen](https://arxiv.org/search/cs?searchtype=author&query=Zen%2C+H)

> We introduce CVSS, a massively multilingual-to-English speech-to-speech translation (S2ST) corpus, covering sentence-level parallel S2ST pairs from 21 languages into English. CVSS is derived from the Common Voice speech corpus and the CoVoST 2 speech-to-text translation (ST) corpus, by synthesizing the translation text from CoVoST 2 into speech using state-of-the-art TTS systems. Two versions of translation speeches are provided: 1) CVSS-C: All the translation speeches are in a single high-quality canonical voice; 2) CVSS-T: The translation speeches are in voices transferred from the corresponding source speeches. In addition, CVSS provides normalized translation text which matches the pronunciation in the translation speech. On each version of CVSS, we built baseline multilingual direct S2ST models and cascade S2ST models, verifying the effectiveness of the corpus. To build strong cascade S2ST baselines, we trained an ST model on CoVoST 2, which outperforms the previous state-of-the-art trained on the corpus without extra data by 5.8 BLEU. Nevertheless, the performance of the direct S2ST models approaches the strong cascade baselines when trained from scratch, and with only 0.1 or 0.7 BLEU difference on ASR transcribed translation when initialized from matching ST models.

| Comments: | Submitted to LREC 2022                                       |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2201.03713](https://arxiv.org/abs/2201.03713) [cs.CL]** |
|           | (or **[arXiv:2201.03713v1](https://arxiv.org/abs/2201.03713v1) [cs.CL]** for this version) |





<h2 id="2022-01-12-3">3. Quantifying Robustness to Adversarial Word Substitutions
</h2>

Title:  [Quantifying Robustness to Adversarial Word Substitutions](https://arxiv.org/abs/2201.03829)

Authors: [Yuting Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Y), [Pei Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+P), [FeiFei Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+F), [Juan Cao](https://arxiv.org/search/cs?searchtype=author&query=Cao%2C+J), [Meishan Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+M), [Jian Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+J), [Jintao Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+J)

> Deep-learning-based NLP models are found to be vulnerable to word substitution perturbations. Before they are widely adopted, the fundamental issues of robustness need to be addressed. Along this line, we propose a formal framework to evaluate word-level robustness. First, to study safe regions for a model, we introduce robustness radius which is the boundary where the model can resist any perturbation. As calculating the maximum robustness radius is computationally hard, we estimate its upper and lower bound. We repurpose attack methods as ways of seeking upper bound and design a pseudo-dynamic programming algorithm for a tighter upper bound. Then verification method is utilized for a lower bound. Further, for evaluating the robustness of regions outside a safe radius, we reexamine robustness from another view: quantification. A robustness metric with a rigorous statistical guarantee is introduced to measure the quantification of adversarial examples, which indicates the model's susceptibility to perturbations outside the safe radius. The metric helps us figure out why state-of-the-art models like BERT can be easily fooled by a few word substitutions, but generalize well in the presence of real-world noises.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2201.03829](https://arxiv.org/abs/2201.03829) [cs.CL]** |
|           | (or **[arXiv:2201.03829v1](https://arxiv.org/abs/2201.03829v1) [cs.CL]** for this version) |





# 2022-01-11

[Return to Index](#Index)



<h2 id="2022-01-11-1">1. Towards the Next 1000 Languages in Multilingual Machine Translation: Exploring the Synergy Between Supervised and Self-Supervised Learning
</h2>

Title: [Towards the Next 1000 Languages in Multilingual Machine Translation: Exploring the Synergy Between Supervised and Self-Supervised Learning](https://arxiv.org/abs/2201.03110)

Authors: [Aditya Siddhant](https://arxiv.org/search/cs?searchtype=author&query=Siddhant%2C+A), [Ankur Bapna](https://arxiv.org/search/cs?searchtype=author&query=Bapna%2C+A), [Orhan Firat](https://arxiv.org/search/cs?searchtype=author&query=Firat%2C+O), [Yuan Cao](https://arxiv.org/search/cs?searchtype=author&query=Cao%2C+Y), [Mia Xu Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+M+X), [Isaac Caswell](https://arxiv.org/search/cs?searchtype=author&query=Caswell%2C+I), [Xavier Garcia](https://arxiv.org/search/cs?searchtype=author&query=Garcia%2C+X)

> Achieving universal translation between all human language pairs is the holy-grail of machine translation (MT) research. While recent progress in massively multilingual MT is one step closer to reaching this goal, it is becoming evident that extending a multilingual MT system simply by training on more parallel data is unscalable, since the availability of labeled data for low-resource and non-English-centric language pairs is forbiddingly limited. To this end, we present a pragmatic approach towards building a multilingual MT model that covers hundreds of languages, using a mixture of supervised and self-supervised objectives, depending on the data availability for different language pairs. We demonstrate that the synergy between these two training paradigms enables the model to produce high-quality translations in the zero-resource setting, even surpassing supervised translation quality for low- and mid-resource languages. We conduct a wide array of experiments to understand the effect of the degree of multilingual supervision, domain mismatches and amounts of parallel and monolingual data on the quality of our self-supervised multilingual models. To demonstrate the scalability of the approach, we train models with over 200 languages and demonstrate high performance on zero-resource translation on several previously under-studied languages. We hope our findings will serve as a stepping stone towards enabling translation for the next thousand languages.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2201.03110](https://arxiv.org/abs/2201.03110) [cs.CL]** |
|           | (or **[arXiv:2201.03110v1](https://arxiv.org/abs/2201.03110v1) [cs.CL]** for this version) |





<h2 id="2022-01-11-2">2. Black-Box Tuning for Language-Model-as-a-Service
</h2>

Title: [Black-Box Tuning for Language-Model-as-a-Service](https://arxiv.org/abs/2201.03514)

Authors: [Tianxiang Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+T), [Yunfan Shao](https://arxiv.org/search/cs?searchtype=author&query=Shao%2C+Y), [Hong Qian](https://arxiv.org/search/cs?searchtype=author&query=Qian%2C+H), [Xuanjing Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+X), [Xipeng Qiu](https://arxiv.org/search/cs?searchtype=author&query=Qiu%2C+X)

> Extremely large pre-trained language models (PTMs) such as GPT-3 are usually released as a service, allowing users to design task-specific prompts to query the PTMs through some black-box APIs. In such a scenario, which we call Language-Model-as-a-Service (LMaaS), gradients of the PTMs are usually not available. Can we optimize the task prompts by only accessing the model inference APIs? Based on recent observations that large PTMs have a very low intrinsic dimensionality, this work proposes the Black-Box Tuning to optimize PTMs through derivative-free algorithms. In particular, we invoke the CMA-ES to optimize the continuous prompt prepended to the input text by iteratively calling PTM inference APIs. Our experimental results demonstrate that, black-box tuning with RoBERTa on a few labeled samples not only significantly outperforms manual prompt and GPT-3's in-context learning, but also surpasses the gradient-based counterparts, namely prompt tuning and full model tuning.

| Comments: | Work in progress                                             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2201.03514](https://arxiv.org/abs/2201.03514) [cs.CL]** |
|           | (or **[arXiv:2201.03514v1](https://arxiv.org/abs/2201.03514v1) [cs.CL]** for this version) |





<h2 id="2022-01-11-3">3. SCROLLS: Standardized CompaRison Over Long Language Sequences
</h2>

Title: [SCROLLS: Standardized CompaRison Over Long Language Sequences](https://arxiv.org/abs/2201.03533)

Authors: [Uri Shaham](https://arxiv.org/search/cs?searchtype=author&query=Shaham%2C+U), [Elad Segal](https://arxiv.org/search/cs?searchtype=author&query=Segal%2C+E), [Maor Ivgi](https://arxiv.org/search/cs?searchtype=author&query=Ivgi%2C+M), [Avia Efrat](https://arxiv.org/search/cs?searchtype=author&query=Efrat%2C+A), [Ori Yoran](https://arxiv.org/search/cs?searchtype=author&query=Yoran%2C+O), [Adi Haviv](https://arxiv.org/search/cs?searchtype=author&query=Haviv%2C+A), [Ankit Gupta](https://arxiv.org/search/cs?searchtype=author&query=Gupta%2C+A), [Wenhan Xiong](https://arxiv.org/search/cs?searchtype=author&query=Xiong%2C+W), [Mor Geva](https://arxiv.org/search/cs?searchtype=author&query=Geva%2C+M), [Jonathan Berant](https://arxiv.org/search/cs?searchtype=author&query=Berant%2C+J), [Omer Levy](https://arxiv.org/search/cs?searchtype=author&query=Levy%2C+O)

> NLP benchmarks have largely focused on short texts, such as sentences and paragraphs, even though long texts comprise a considerable amount of natural language in the wild. We introduce SCROLLS, a suite of tasks that require reasoning over long texts. We examine existing long-text datasets, and handpick ones where the text is naturally long, while prioritizing tasks that involve synthesizing information across the input. SCROLLS contains summarization, question answering, and natural language inference tasks, covering multiple domains, including literature, science, business, and entertainment. Initial baselines, including Longformer Encoder-Decoder, indicate that there is ample room for improvement on SCROLLS. We make all datasets available in a unified text-to-text format and host a live leaderboard to facilitate research on model architecture and pretraining methods.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Machine Learning (stat.ML) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2201.03533](https://arxiv.org/abs/2201.03533) [cs.CL]** |
|           | (or **[arXiv:2201.03533v1](https://arxiv.org/abs/2201.03533v1) [cs.CL]** for this version) |







# 2022-01-10

[Return to Index](#Index)



<h2 id="2022-01-10-1">1. Automatic Speech Recognition Datasets in Cantonese Language: A Survey and a New Dataset
</h2>

Title: [Automatic Speech Recognition Datasets in Cantonese Language: A Survey and a New Dataset](https://arxiv.org/abs/2201.02419)

Authors: [Tiezheng Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+T), [Rita Frieske](https://arxiv.org/search/cs?searchtype=author&query=Frieske%2C+R), [Peng Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+P), [Samuel Cahyawijaya](https://arxiv.org/search/cs?searchtype=author&query=Cahyawijaya%2C+S), [Cheuk Tung Shadow Yiu](https://arxiv.org/search/cs?searchtype=author&query=Yiu%2C+C+T+S), [Holy Lovenia](https://arxiv.org/search/cs?searchtype=author&query=Lovenia%2C+H), [Wenliang Dai](https://arxiv.org/search/cs?searchtype=author&query=Dai%2C+W), [Elham J. Barezi](https://arxiv.org/search/cs?searchtype=author&query=Barezi%2C+E+J), [Qifeng Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Q), [Xiaojuan Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+X), [Bertram E. Shi](https://arxiv.org/search/cs?searchtype=author&query=Shi%2C+B+E), [Pascale Fung](https://arxiv.org/search/cs?searchtype=author&query=Fung%2C+P)

> Automatic speech recognition (ASR) on low resource languages improves access of linguistic minorities to technological advantages provided by Artificial Intelligence (AI). In this paper, we address a problem of data scarcity of Hong Kong Cantonese language by creating a new Cantonese dataset. Our dataset, Multi-Domain Cantonese Corpus (MDCC), consists of 73.6 hours of clean read speech paired with transcripts, collected from Cantonese audiobooks from Hong Kong. It combines philosophy, politics, education, culture, lifestyle and family domains, covering a wide range of topics. We also review all existing Cantonese datasets and perform experiments on the two biggest datasets (MDCC and Common Voice zh-HK). We analyze the existing datasets according to their speech type, data source, total size and availability. The results of experiments conducted with Fairseq S2T Transformer, a state-of-the-art ASR model, show the effectiveness of our dataset. In addition, we create a powerful and robust Cantonese ASR model by applying multi-dataset learning on MDCC and Common Voice zh-HK.

| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2201.02419](https://arxiv.org/abs/2201.02419) [cs.CL]** |
|           | (or **[arXiv:2201.02419v1](https://arxiv.org/abs/2201.02419v1) [cs.CL]** for this version) |





<h2 id="2022-01-10-2">2. Semantic-based Data Augmentation for Math Word Problems
</h2>

Title: [Semantic-based Data Augmentation for Math Word Problems](https://arxiv.org/abs/2201.02489)

Authors: [Ailisi Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+A), [Jiaqing Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+J), [Yanghua Xiao](https://arxiv.org/search/cs?searchtype=author&query=Xiao%2C+Y)

> It's hard for neural MWP solvers to deal with tiny local variances. In MWP task, some local changes conserve the original semantic while the others may totally change the underlying logic. Currently, existing datasets for MWP task contain limited samples which are key for neural models to learn to disambiguate different kinds of local variances in questions and solve the questions correctly. In this paper, we propose a set of novel data augmentation approaches to supplement existing datasets with such data that are augmented with different kinds of local variances, and help to improve the generalization ability of current neural models. New samples are generated by knowledge guided entity replacement, and logic guided problem reorganization. The augmentation approaches are ensured to keep the consistency between the new data and their labels. Experimental results have shown the necessity and the effectiveness of our methods.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2201.02489](https://arxiv.org/abs/2201.02489) [cs.CL]** |
|           | (or **[arXiv:2201.02489v1](https://arxiv.org/abs/2201.02489v1) [cs.CL]** for this version) |





<h2 id="2022-01-10-3">3. Repairing Adversarial Texts through Perturbation
</h2>

Title: [Repairing Adversarial Texts through Perturbation](https://arxiv.org/abs/2201.02504)

Authors: [Guoliang Dong](https://arxiv.org/search/cs?searchtype=author&query=Dong%2C+G), [Jingyi Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+J), [Jun Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+J), [Sudipta Chattopadhyay](https://arxiv.org/search/cs?searchtype=author&query=Chattopadhyay%2C+S), [Xinyu Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Ting Dai](https://arxiv.org/search/cs?searchtype=author&query=Dai%2C+T), [Jie Shi](https://arxiv.org/search/cs?searchtype=author&query=Shi%2C+J), [Jin Song Dong](https://arxiv.org/search/cs?searchtype=author&query=Dong%2C+J+S)

> It is known that neural networks are subject to attacks through adversarial perturbations, i.e., inputs which are maliciously crafted through perturbations to induce wrong predictions. Furthermore, such attacks are impossible to eliminate, i.e., the adversarial perturbation is still possible after applying mitigation methods such as adversarial training. Multiple approaches have been developed to detect and reject such adversarial inputs, mostly in the image domain. Rejecting suspicious inputs however may not be always feasible or ideal. First, normal inputs may be rejected due to false alarms generated by the detection algorithm. Second, denial-of-service attacks may be conducted by feeding such systems with adversarial inputs. To address the gap, in this work, we propose an approach to automatically repair adversarial texts at runtime. Given a text which is suspected to be adversarial, we novelly apply multiple adversarial perturbation methods in a positive way to identify a repair, i.e., a slightly mutated but semantically equivalent text that the neural network correctly classifies. Our approach has been experimented with multiple models trained for natural language processing tasks and the results show that our approach is effective, i.e., it successfully repairs about 80\% of the adversarial texts. Furthermore, depending on the applied perturbation method, an adversarial text could be repaired in as short as one second on average.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Cryptography and Security (cs.CR) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2201.02504](https://arxiv.org/abs/2201.02504) [cs.CL]** |
|           | (or **[arXiv:2201.02504v1](https://arxiv.org/abs/2201.02504v1) [cs.CL]** for this version) |





<h2 id="2022-01-10-4">4. Code-Switching Text Augmentation for Multilingual Speech Processing
</h2>

Title: [Code-Switching Text Augmentation for Multilingual Speech Processing](https://arxiv.org/abs/2201.02550)

Authors: [Amir Hussein](https://arxiv.org/search/cs?searchtype=author&query=Hussein%2C+A), [Shammur Absar Chowdhury](https://arxiv.org/search/cs?searchtype=author&query=Chowdhury%2C+S+A), [Ahmed Abdelali](https://arxiv.org/search/cs?searchtype=author&query=Abdelali%2C+A), [Najim Dehak](https://arxiv.org/search/cs?searchtype=author&query=Dehak%2C+N), [Ahmed Ali](https://arxiv.org/search/cs?searchtype=author&query=Ali%2C+A)

> The pervasiveness of intra-utterance Code-switching (CS) in spoken content has enforced ASR systems to handle mixed input. Yet, designing a CS-ASR has many challenges, mainly due to the data scarcity, grammatical structure complexity, and mismatch along with unbalanced language usage distribution. Recent ASR studies showed the predominance of E2E-ASR using multilingual data to handle CS phenomena with little CS data. However, the dependency on the CS data still remains. In this work, we propose a methodology to augment the monolingual data for artificially generating spoken CS text to improve different speech modules. We based our approach on Equivalence Constraint theory while exploiting aligned translation pairs, to generate grammatically valid CS content. Our empirical results show a relative gain of 29-34 % in perplexity and around 2% in WER for two ecological and noisy CS test sets. Finally, the human evaluation suggests that 83.8% of the generated data is acceptable to humans.

| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2201.02550](https://arxiv.org/abs/2201.02550) [cs.CL]** |
|           | (or **[arXiv:2201.02550v1](https://arxiv.org/abs/2201.02550v1) [cs.CL]** for this version) |






# 2022-01-07

[Return to Index](#Index)



<h2 id="2022-01-07-1">1. Compact Bidirectional Transformer for Image Captioning
</h2>

Title: [Compact Bidirectional Transformer for Image Captioning](https://arxiv.org/abs/2201.01984)

Authors: [Yuanen Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+Y), [Zhenzhen Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu%2C+Z), [Daqing Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+D), [Huixia Ben](https://arxiv.org/search/cs?searchtype=author&query=Ben%2C+H), [Meng Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+M)

> Most current image captioning models typically generate captions from left to right. This unidirectional property makes them can only leverage past context but not future context. Though recent refinement-based models can exploit both past and future context by generating a new caption in the second stage based on pre-retrieved or pre-generated captions in the first stage, the decoder of these models generally consists of two networks~(i.e. a retriever or captioner in the first stage and a refiner in the second stage), which can only be executed sequentially. In this paper, we introduce a Compact Bidirectional Transformer model for image captioning that can leverage bidirectional context implicitly and explicitly while the decoder can be executed parallelly. Specifically, it is implemented by tightly coupling left-to-right(L2R) and right-to-left(R2L) flows into a single compact model~(i.e. implicitly) and optionally allowing interaction of the two flows(i.e. explicitly), while the final caption is chosen from either L2R or R2L flow in a sentence-level ensemble manner. We conduct extensive ablation studies on the MSCOCO benchmark and find that the compact architecture, which serves as a regularization for implicitly exploiting bidirectional context, and the sentence-level ensemble play more important roles than the explicit interaction mechanism. By combining with word-level ensemble seamlessly, the effect of the sentence-level ensemble is further enlarged. We further extend the conventional one-flow self-critical training to the two-flows version under this architecture and achieve new state-of-the-art results in comparison with non-vision-language-pretraining models. Source code is available at {\color{magenta}\url{[this https URL](https://github.com/YuanEZhou/CBTrans)}}.

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2201.01984](https://arxiv.org/abs/2201.01984) [cs.CV]** |
|           | (or **[arXiv:2201.01984v1](https://arxiv.org/abs/2201.01984v1) [cs.CV]** for this version) |





<h2 id="2022-01-07-2">2. Self-Training Vision Language BERTs with a Unified Conditional Model
</h2>

Title: [Self-Training Vision Language BERTs with a Unified Conditional Model](https://arxiv.org/abs/2201.02010)

Authors: [Xiaofeng Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+X), [Fengmao Lv](https://arxiv.org/search/cs?searchtype=author&query=Lv%2C+F), [Fayao Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+F), [Guosheng Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+G)

> Natural language BERTs are trained with language corpus in a self-supervised manner. Unlike natural language BERTs, vision language BERTs need paired data to train, which restricts the scale of VL-BERT pretraining. We propose a self-training approach that allows training VL-BERTs from unlabeled image data. The proposed method starts with our unified conditional model -- a vision language BERT model that can perform zero-shot conditional generation. Given different conditions, the unified conditional model can generate captions, dense captions, and even questions. We use the labeled image data to train a teacher model and use the trained model to generate pseudo captions on unlabeled image data. We then combine the labeled data and pseudo labeled data to train a student model. The process is iterated by putting the student model as a new teacher. By using the proposed self-training approach and only 300k unlabeled extra data, we are able to get competitive or even better performances compared to the models of similar model size trained with 3 million extra image data.

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2201.02010](https://arxiv.org/abs/2201.02010) [cs.CV]** |
|           | (or **[arXiv:2201.02010v1](https://arxiv.org/abs/2201.02010v1) [cs.CV]** for this version) |





<h2 id="2022-01-07-3">3. Phrase-level Adversarial Example Generation for Neural Machine Translation
</h2>

Title: [Phrase-level Adversarial Example Generation for Neural Machine Translation](https://arxiv.org/abs/2201.02009)

Authors: [Juncheng Wan](https://arxiv.org/search/cs?searchtype=author&query=Wan%2C+J), [Jian Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+J), [Shuming Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+S), [Dongdong Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+D), [Weinan Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+W), [Yong Yu](https://arxiv.org/search/cs?searchtype=author&query=Yu%2C+Y), [Furu Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+F)

> While end-to-end neural machine translation (NMT) has achieved impressive progress, noisy input usually leads models to become fragile and unstable. Generating adversarial examples as the augmented data is proved to be useful to alleviate this problem. Existing methods for adversarial example generation (AEG) are word-level or character-level. In this paper, we propose a phrase-level adversarial example generation (PAEG) method to enhance the robustness of the model. Our method leverages a gradient-based strategy to substitute phrases of vulnerable positions in the source input. We verify our method on three benchmarks, including LDC Chinese-English, IWSLT14 German-English, and WMT14 English-German tasks. Experimental results demonstrate that our approach significantly improves performance compared to previous methods.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2201.02009](https://arxiv.org/abs/2201.02009) [cs.CL]** |
|           | (or **[arXiv:2201.02009v1](https://arxiv.org/abs/2201.02009v1) [cs.CL]** for this version) |





# 2022-01-06

[Return to Index](#Index)



<h2 id="2022-01-06-1">1. All You Need In Sign Language Production
</h2>

Title: [All You Need In Sign Language Production](https://arxiv.org/abs/2201.01609)

Authors: [Razieh Rastgoo](https://arxiv.org/search/cs?searchtype=author&query=Rastgoo%2C+R), [Kourosh Kiani](https://arxiv.org/search/cs?searchtype=author&query=Kiani%2C+K), [Sergio Escalera](https://arxiv.org/search/cs?searchtype=author&query=Escalera%2C+S), [Vassilis Athitsos](https://arxiv.org/search/cs?searchtype=author&query=Athitsos%2C+V), [Mohammad Sabokrou](https://arxiv.org/search/cs?searchtype=author&query=Sabokrou%2C+M)

> Sign Language is the dominant form of communication language used in the deaf and hearing-impaired community. To make an easy and mutual communication between the hearing-impaired and the hearing communities, building a robust system capable of translating the spoken language into sign language and vice versa is fundamental. To this end, sign language recognition and production are two necessary parts for making such a two-way system. Sign language recognition and production need to cope with some critical challenges. In this survey, we review recent advances in Sign Language Production (SLP) and related areas using deep learning. To have more realistic perspectives to sign language, we present an introduction to the Deaf culture, Deaf centers, psychological perspective of sign language, the main differences between spoken language and sign language. Furthermore, we present the fundamental components of a bi-directional sign language translation system, discussing the main challenges in this area. Also, the backbone architectures and methods in SLP are briefly introduced and the proposed taxonomy on SLP is presented. Finally, a general framework for SLP and performance evaluation, and also a discussion on the recent developments, advantages, and limitations in SLP, commenting on possible lines for future research are presented.

| Comments: | arXiv admin note: substantial text overlap with [arXiv:2103.15910](https://arxiv.org/abs/2103.15910) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2201.01609](https://arxiv.org/abs/2201.01609) [cs.CV]** |
|           | (or **[arXiv:2201.01609v1](https://arxiv.org/abs/2201.01609v1) [cs.CV]** for this version) |





<h2 id="2022-01-06-2">2. SMDT: Selective Memory-Augmented Neural Document Translation
</h2>

Title: [SMDT: Selective Memory-Augmented Neural Document Translation](https://arxiv.org/abs/2201.01631)

Authors: [Xu Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+X), [Jian Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+J), [Haoyang Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+H), [Shuming Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+S), [Dongdong Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+D), [Jinlong Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+J), [Furu Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+F)

> Existing document-level neural machine translation (NMT) models have sufficiently explored different context settings to provide guidance for target generation. However, little attention is paid to inaugurate more diverse context for abundant context information. In this paper, we propose a Selective Memory-augmented Neural Document Translation model to deal with documents containing large hypothesis space of the context. Specifically, we retrieve similar bilingual sentence pairs from the training corpus to augment global context and then extend the two-stream attention model with selective mechanism to capture local context and diverse global contexts. This unified approach allows our model to be trained elegantly on three publicly document-level machine translation datasets and significantly outperforms previous document-level NMT models.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2201.01631](https://arxiv.org/abs/2201.01631) [cs.CL]** |
|           | (or **[arXiv:2201.01631v1](https://arxiv.org/abs/2201.01631v1) [cs.CL]** for this version) |





# 2022-01-05

[Return to Index](#Index)



<h2 id="2022-01-05-1">1. Interactive Attention AI to translate low light photos to captions for night scene understanding in women safety
</h2>

Title: [Interactive Attention AI to translate low light photos to captions for night scene understanding in women safety](https://arxiv.org/abs/2201.00969)

Authors: [Rajagopal A](https://arxiv.org/search/cs?searchtype=author&query=A%2C+R), [Nirmala V](https://arxiv.org/search/cs?searchtype=author&query=V%2C+N), [Arun Muthuraj Vedamanickam](https://arxiv.org/search/cs?searchtype=author&query=Vedamanickam%2C+A+M)

> There is amazing progress in Deep Learning based models for Image captioning and Low Light image enhancement. For the first time in literature, this paper develops a Deep Learning model that translates night scenes to sentences, opening new possibilities for AI applications in the safety of visually impaired women. Inspired by Image Captioning and Visual Question Answering, a novel Interactive Image Captioning is developed. A user can make the AI focus on any chosen person of interest by influencing the attention scoring. Attention context vectors are computed from CNN feature vectors and user-provided start word. The Encoder-Attention-Decoder neural network learns to produce captions from low brightness images. This paper demonstrates how women safety can be enabled by researching a novel AI capability in the Interactive Vision-Language model for perception of the environment in the night.

| Comments:    | In Springer Proceedings. International Conference On Big Data, Machine Learning and Applications 2021. [this http URL](http://bigdml.nits.ac.in/) |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| ACM classes: | I.2.0                                                        |
| Cite as:     | **[arXiv:2201.00969](https://arxiv.org/abs/2201.00969) [cs.CV]** |
|              | (or **[arXiv:2201.00969v1](https://arxiv.org/abs/2201.00969v1) [cs.CV]** for this version) |





<h2 id="2022-01-05-2">2. StyleM: Stylized Metrics for Image Captioning Built with Contrastive N-grams
</h2>

Title: [StyleM: Stylized Metrics for Image Captioning Built with Contrastive N-grams](https://arxiv.org/abs/2201.00975)

Authors: [Chengxi Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+C), [Brent Harrison](https://arxiv.org/search/cs?searchtype=author&query=Harrison%2C+B)

> In this paper, we build two automatic evaluation metrics for evaluating the association between a machine-generated caption and a ground truth stylized caption: OnlyStyle and StyleCIDEr.

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2201.00975](https://arxiv.org/abs/2201.00975) [cs.CV]** |
|           | (or **[arXiv:2201.00975v1](https://arxiv.org/abs/2201.00975v1) [cs.CV]** for this version) |







# 2022-01-04

[Return to Index](#Index)



<h2 id="2022-01-04-1">1. How do lexical semantics affect translation? An empirical study
</h2>

Title: [How do lexical semantics affect translation? An empirical study](https://arxiv.org/abs/2201.00075)

Authors:[Vivek Subramanian](https://arxiv.org/search/cs?searchtype=author&query=Subramanian%2C+V), [Dhanasekar Sundararaman](https://arxiv.org/search/cs?searchtype=author&query=Sundararaman%2C+D)

> Neural machine translation (NMT) systems aim to map text from one language into another. While there are a wide variety of applications of NMT, one of the most important is translation of natural language. A distinguishing factor of natural language is that words are typically ordered according to the rules of the grammar of a given language. Although many advances have been made in developing NMT systems for translating natural language, little research has been done on understanding how the word ordering of and lexical similarity between the source and target language affect translation performance. Here, we investigate these relationships on a variety of low-resource language pairs from the OpenSubtitles2016 database, where the source language is English, and find that the more similar the target language is to English, the greater the translation performance. In addition, we study the impact of providing NMT models with part of speech of words (POS) in the English sequence and find that, for Transformer-based models, the more dissimilar the target language is from English, the greater the benefit provided by POS.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2201.00075](https://arxiv.org/abs/2201.00075) [cs.CL]** |
|           | (or **[arXiv:2201.00075v1](https://arxiv.org/abs/2201.00075v1) [cs.CL]** for this version) |





<h2 id="2022-01-04-2">2. Which Student is Best? A Comprehensive Knowledge Distillation Exam for Task-Specific BERT Models
</h2>

Title: [Which Student is Best? A Comprehensive Knowledge Distillation Exam for Task-Specific BERT Models](https://arxiv.org/abs/2201.00558)

Authors:[Made Nindyatama Nityasya](https://arxiv.org/search/cs?searchtype=author&query=Nityasya%2C+M+N), [Haryo Akbarianto Wibowo](https://arxiv.org/search/cs?searchtype=author&query=Wibowo%2C+H+A), [Rendi Chevi](https://arxiv.org/search/cs?searchtype=author&query=Chevi%2C+R), [Radityo Eko Prasojo](https://arxiv.org/search/cs?searchtype=author&query=Prasojo%2C+R+E), [Alham Fikri Aji](https://arxiv.org/search/cs?searchtype=author&query=Aji%2C+A+F)

> We perform knowledge distillation (KD) benchmark from task-specific BERT-base teacher models to various student models: BiLSTM, CNN, BERT-Tiny, BERT-Mini, and BERT-Small. Our experiment involves 12 datasets grouped in two tasks: text classification and sequence labeling in the Indonesian language. We also compare various aspects of distillations including the usage of word embeddings and unlabeled data augmentation. Our experiments show that, despite the rising popularity of Transformer-based models, using BiLSTM and CNN student models provide the best trade-off between performance and computational resource (CPU, RAM, and storage) compared to pruned BERT models. We further propose some quick wins on performing KD to produce small NLP models via efficient KD training mechanisms involving simple choices of loss functions, word embeddings, and unlabeled data preparation.

| Comments:    | 14 pages, 3 figures, submitted to Elsevier                   |
| ------------ | ------------------------------------------------------------ |
| Subjects:    | **Computation and Language (cs.CL)**                         |
| MSC classes: | 68T50                                                        |
| ACM classes: | I.2.7; I.2.6                                                 |
| Cite as:     | **[arXiv:2201.00558](https://arxiv.org/abs/2201.00558) [cs.CL]** |
|              | (or **[arXiv:2201.00558v1](https://arxiv.org/abs/2201.00558v1) [cs.CL]** for this version) |





<h2 id="2022-01-04-3">3. Robust Natural Language Processing: Recent Advances, Challenges, and Future Directions
</h2>

Title: [Robust Natural Language Processing: Recent Advances, Challenges, and Future Directions](https://arxiv.org/abs/2201.00768)

Authors:[Marwan Omar](https://arxiv.org/search/cs?searchtype=author&query=Omar%2C+M), [Soohyeon Choi](https://arxiv.org/search/cs?searchtype=author&query=Choi%2C+S), [DaeHun Nyang](https://arxiv.org/search/cs?searchtype=author&query=Nyang%2C+D), [David Mohaisen](https://arxiv.org/search/cs?searchtype=author&query=Mohaisen%2C+D)

> Recent natural language processing (NLP) techniques have accomplished high performance on benchmark datasets, primarily due to the significant improvement in the performance of deep learning. The advances in the research community have led to great enhancements in state-of-the-art production systems for NLP tasks, such as virtual assistants, speech recognition, and sentiment analysis. However, such NLP systems still often fail when tested with adversarial attacks. The initial lack of robustness exposed troubling gaps in current models' language understanding capabilities, creating problems when NLP systems are deployed in real life. In this paper, we present a structured overview of NLP robustness research by summarizing the literature in a systemic way across various dimensions. We then take a deep-dive into the various dimensions of robustness, across techniques, metrics, embeddings, and benchmarks. Finally, we argue that robustness should be multi-dimensional, provide insights into current research, identify gaps in the literature to suggest directions worth pursuing to address these gaps.

| Comments: | Survey; 2 figures, 4 tables                                  |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Cryptography and Security (cs.CR); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2201.00768](https://arxiv.org/abs/2201.00768) [cs.CL]** |
|           | (or **[arXiv:2201.00768v1](https://arxiv.org/abs/2201.00768v1) [cs.CL]** for this version) |



# 2022-01-03

[Return to Index](#Index)



<h2 id="2022-01-03-1">1. ERNIE-ViLG: Unified Generative Pre-training for Bidirectional Vision-Language Generation
</h2>

Title: [ERNIE-ViLG: Unified Generative Pre-training for Bidirectional Vision-Language Generation](https://arxiv.org/abs/2112.15283)

Authors: [Han Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+H), [Weichong Yin](https://arxiv.org/search/cs?searchtype=author&query=Yin%2C+W), [Yewei Fang](https://arxiv.org/search/cs?searchtype=author&query=Fang%2C+Y), [Lanxin Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+L), [Boqiang Duan](https://arxiv.org/search/cs?searchtype=author&query=Duan%2C+B), [Zhihua Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+Z), [Yu Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+Y), [Hao Tian](https://arxiv.org/search/cs?searchtype=author&query=Tian%2C+H), [Hua Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+H), [Haifeng Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+H)

> Conventional methods for the image-text generation tasks mainly tackle the naturally bidirectional generation tasks separately, focusing on designing task-specific frameworks to improve the quality and fidelity of the generated samples. Recently, Vision-Language Pre-training models have greatly improved the performance of the image-to-text generation tasks, but large-scale pre-training models for text-to-image synthesis task are still under-developed. In this paper, we propose ERNIE-ViLG, a unified generative pre-training framework for bidirectional image-text generation with transformer model. Based on the image quantization models, we formulate both image generation and text generation as autoregressive generative tasks conditioned on the text/image input. The bidirectional image-text generative modeling eases the semantic alignments across vision and language. For the text-to-image generation process, we further propose an end-to-end training method to jointly learn the visual sequence generator and the image reconstructor. To explore the landscape of large-scale pre-training for bidirectional text-image generation, we train a 10-billion parameter ERNIE-ViLG model on a large-scale dataset of 145 million (Chinese) image-text pairs which achieves state-of-the-art performance for both text-to-image and image-to-text tasks, obtaining an FID of 7.9 on MS-COCO for text-to-image synthesis and best results on COCO-CN and AIC-ICC for image captioning.

| Comments: | 15 pages, 7 figures                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2112.15283](https://arxiv.org/abs/2112.15283) [cs.CV]** |
|           | (or **[arXiv:2112.15283v1](https://arxiv.org/abs/2112.15283v1) [cs.CV]** for this version) |





<h2 id="2022-01-03-2">2. Deconfounded Visual Grounding
</h2>

Title: [Deconfounded Visual Grounding](https://arxiv.org/abs/2112.15324)

Authors: [Jianqiang Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+J), [Yu Qin](https://arxiv.org/search/cs?searchtype=author&query=Qin%2C+Y), [Jiaxin Qi](https://arxiv.org/search/cs?searchtype=author&query=Qi%2C+J), [Qianru Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+Q), [Hanwang Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+H)

> We focus on the confounding bias between language and location in the visual grounding pipeline, where we find that the bias is the major visual reasoning bottleneck. For example, the grounding process is usually a trivial language-location association without visual reasoning, e.g., grounding any language query containing sheep to the nearly central regions, due to that most queries about sheep have ground-truth locations at the image center. First, we frame the visual grounding pipeline into a causal graph, which shows the causalities among image, query, target location and underlying confounder. Through the causal graph, we know how to break the grounding bottleneck: deconfounded visual grounding. Second, to tackle the challenge that the confounder is unobserved in general, we propose a confounder-agnostic approach called: Referring Expression Deconfounder (RED), to remove the confounding bias. Third, we implement RED as a simple language attention, which can be applied in any grounding method. On popular benchmarks, RED improves various state-of-the-art grounding methods by a significant margin. Code will soon be available at: [this https URL](https://github.com/JianqiangH/Deconfounded_VG).

| Comments: | AAAI 2022 Accepted                                           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2112.15324](https://arxiv.org/abs/2112.15324) [cs.CV]** |
|           | (or **[arXiv:2112.15324v1](https://arxiv.org/abs/2112.15324v1) [cs.CV]** for this version) |





<h2 id="2022-01-03-3">3. Materialized Knowledge Bases from Commonsense Transformers
</h2>

Title: [Materialized Knowledge Bases from Commonsense Transformers](https://arxiv.org/abs/2112.14815)

Authors: [Tuan-Phong Nguyen](https://arxiv.org/search/cs?searchtype=author&query=Nguyen%2C+T), [Simon Razniewski](https://arxiv.org/search/cs?searchtype=author&query=Razniewski%2C+S)

> Starting from the COMET methodology by Bosselut et al. (2019), generating commonsense knowledge directly from pre-trained language models has recently received significant attention. Surprisingly, up to now no materialized resource of commonsense knowledge generated this way is publicly available. This paper fills this gap, and uses the materialized resources to perform a detailed analysis of the potential of this approach in terms of precision and recall. Furthermore, we identify common problem cases, and outline use cases enabled by materialized resources. We posit that the availability of these resources is important for the advancement of the field, as it enables an off-the-shelf-use of the resulting knowledge, as well as further analyses on its strengths and weaknesses.

| Comments: | 7 pages                                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2112.14815](https://arxiv.org/abs/2112.14815) [cs.CL]** |
|           | (or **[arXiv:2112.14815v1](https://arxiv.org/abs/2112.14815v1) [cs.CL]** for this version) |





<h2 id="2022-01-03-4">4. ViNMT: Neural Machine Translation Tookit
</h2>

Title: [ViNMT: Neural Machine Translation Tookit](https://arxiv.org/abs/2112.15272)

Authors: [Nguyen Hoang Quan](https://arxiv.org/search/cs?searchtype=author&query=Quan%2C+N+H), [Nguyen Thanh Dat](https://arxiv.org/search/cs?searchtype=author&query=Dat%2C+N+T), [Nguyen Hoang Minh Cong](https://arxiv.org/search/cs?searchtype=author&query=Cong%2C+N+H+M), [Nguyen Van Vinh](https://arxiv.org/search/cs?searchtype=author&query=Van+Vinh%2C+N), [Ngo Thi Vinh](https://arxiv.org/search/cs?searchtype=author&query=Vinh%2C+N+T), [Nguyen Phuong Thai](https://arxiv.org/search/cs?searchtype=author&query=Thai%2C+N+P), [Tran Hong Viet](https://arxiv.org/search/cs?searchtype=author&query=Viet%2C+T+H)

> We present an open-source toolkit for neural machine translation (NMT). The new toolkit is mainly based on vaulted Transformer (Vaswani et al., 2017) along with many other improvements detailed below, in order to create a self-contained, simple to use, consistent and comprehensive framework for Machine Translation tasks of various domains. It is tooled to support both bilingual and multilingual translation tasks, starting from building the model from respective corpora, to inferring new predictions or packaging the model to serving-capable JIT format.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2112.15272](https://arxiv.org/abs/2112.15272) [cs.CL]** |
|           | (or **[arXiv:2112.15272v1](https://arxiv.org/abs/2112.15272v1) [cs.CL]** for this version) |
