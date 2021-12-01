# MA C.'s Daily Paper Of Interest - December, 2021

# Index


- [2021-12-1](#2021-12-1)

  - [1. Do We Still Need Automatic Speech Recognition for Spoken Language Understanding?](#2021-12-1-1)
  - [2. Improvement in Machine Translation with Generative Adversarial Networks](#2021-12-1-2)
  - [3. Pureformer: Do We Even Need Attention?](#2021-12-1-3)
  
- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MAC-Daily-POI-index.md)



# 2021-12-1

[Return to Index](#Index)



<h2 id="2021-12-1-1">1. Do We Still Need Automatic Speech Recognition for Spoken Language Understanding?
</h2>

Title: [Do We Still Need Automatic Speech Recognition for Spoken Language Understanding?](https://arxiv.org/abs/2111.14842)

Authors: [Lasse Borgholt](https://arxiv.org/search/eess?searchtype=author&query=Borgholt%2C+L), [Jakob Drachmann Havtorn](https://arxiv.org/search/eess?searchtype=author&query=Havtorn%2C+J+D), [Mostafa Abdou](https://arxiv.org/search/eess?searchtype=author&query=Abdou%2C+M), [Joakim Edin](https://arxiv.org/search/eess?searchtype=author&query=Edin%2C+J), [Lars Maaløe](https://arxiv.org/search/eess?searchtype=author&query=Maaløe%2C+L), [Anders Søgaard](https://arxiv.org/search/eess?searchtype=author&query=Søgaard%2C+A), [Christian Igel](https://arxiv.org/search/eess?searchtype=author&query=Igel%2C+C)

> Spoken language understanding (SLU) tasks are usually solved by first transcribing an utterance with automatic speech recognition (ASR) and then feeding the output to a text-based model. Recent advances in self-supervised representation learning for speech data have focused on improving the ASR component. We investigate whether representation learning for speech has matured enough to replace ASR in SLU. We compare learned speech features from wav2vec 2.0, state-of-the-art ASR transcripts, and the ground truth text as input for a novel speech-based named entity recognition task, a cardiac arrest detection task on real-world emergency calls and two existing SLU benchmarks. We show that learned speech features are superior to ASR transcripts on three classification tasks. For machine translation, ASR transcripts are still the better choice. We highlight the intrinsic robustness of wav2vec 2.0 representations to out-of-vocabulary words as key to better performance.

| Comments: | Under review as a conference paper at ICASSP 2022            |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Audio and Speech Processing (eess.AS)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2111.14842](https://arxiv.org/abs/2111.14842) [eess.AS]** |
|           | (or **[arXiv:2111.14842v1](https://arxiv.org/abs/2111.14842v1) [eess.AS]** for this version) |





<h2 id="2021-12-1-2">2. Improvement in Machine Translation with Generative Adversarial Networks
</h2>

Title: [Improvement in Machine Translation with Generative Adversarial Networks](https://arxiv.org/abs/2111.15166)

Authors: [Jay Ahn](https://arxiv.org/search/cs?searchtype=author&query=Ahn%2C+J), [Hari Madhu](https://arxiv.org/search/cs?searchtype=author&query=Madhu%2C+H), [Viet Nguyen](https://arxiv.org/search/cs?searchtype=author&query=Nguyen%2C+V)

> In this paper, we explore machine translation improvement via Generative Adversarial Network (GAN) architecture. We take inspiration from RelGAN, a model for text generation, and NMT-GAN, an adversarial machine translation model, to implement a model that learns to transform awkward, non-fluent English sentences to fluent ones, while only being trained on monolingual corpora. We utilize a parameter λ to control the amount of deviation from the input sentence, i.e. a trade-off between keeping the original tokens and modifying it to be more fluent. Our results improved upon phrase-based machine translation in some cases. Especially, GAN with a transformer generator shows some promising results. We suggests some directions for future works to build upon this proof-of-concept.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2111.15166](https://arxiv.org/abs/2111.15166) [cs.CL]** |
|           | (or **[arXiv:2111.15166v1](https://arxiv.org/abs/2111.15166v1) [cs.CL]** for this version) |





<h2 id="2021-12-1-3">3. Pureformer: Do We Even Need Attention?
</h2>

Title: [Pureformer: Do We Even Need Attention?](https://arxiv.org/abs/2111.15588)

Authors: [Uladzislau Yorsh](https://arxiv.org/search/cs?searchtype=author&query=Yorsh%2C+U), [Alexander Kovalenko](https://arxiv.org/search/cs?searchtype=author&query=Kovalenko%2C+A)

> In this paper we propose that the dot product pairwise matching attention layer, which is widely used in transformer-based models, is redundant for the model performance. Attention in its original formulation has to be seen rather as a human-level tool to explore and/or visualize relevancy scores in the sequences. Instead, we present a simple and fast alternative without any approximation that, to the best of our knowledge, outperforms existing attention approximations on the text classification task from the Long-Range Arena benchmark.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2111.15588](https://arxiv.org/abs/2111.15588) [cs.CL]** |
|           | (or **[arXiv:2111.15588v1](https://arxiv.org/abs/2111.15588v1) [cs.CL]** for this version) |

