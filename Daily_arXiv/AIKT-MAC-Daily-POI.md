# MA C.'s Daily Paper Of Interest - April, 2022

# Index

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



