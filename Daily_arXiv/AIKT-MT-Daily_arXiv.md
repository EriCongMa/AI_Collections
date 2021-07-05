# Daily arXiv: Machine Translation - July, 2021

# Index


- [2021-07-05](#2021-07-05)

  - [1. Transformer-F: A Transformer network with effective methods for learning universal sentence representation](#2021-07-05-1)
  - [2. A Primer on Pretrained Multilingual Language Models](#2021-07-05-2)
  - [3. Interactive decoding of words from visual speech recognition models](#2021-07-05-3)
  - [4. Data Centric Domain Adaptation for Historical Text with OCR Errors](#2021-07-05-4)
- [2021-07-02](#2021-07-02)

  - [1. GlyphCRM: Bidirectional Encoder Representation for Chinese Character with its Glyph](#2021-07-02-1)
  - [2. ESPnet-ST IWSLT 2021 Offline Speech Translation System](#2021-07-02-2)
  - [3. Word-Free Spoken Language Understanding for Mandarin-Chinese](#2021-07-02-3)
  - [4. The USTC-NELSLIP Systems for Simultaneous Speech Translation Task at IWSLT 2021](#2021-07-02-4)
  - [5. Zero-pronoun Data Augmentation for Japanese-to-English Translation](#2021-07-02-5)
  - [6. Modeling Target-side Inflection in Placeholder Translation](#2021-07-02-6)
  - [7. CLINE: Contrastive Learning with Semantic Negative Examples for Natural Language Understanding](#2021-07-02-7 )
- [2021-07-01](#2021-07-01)
  - [1. What Can Unsupervised Machine Translation Contribute to High-Resource Language Pairs?](#2021-07-01-1)
  - [2. Mixed Cross Entropy Loss for Neural Machine Translation](#2021-07-01-2)
  - [3. Cross-lingual alignments of ELMo contextual embeddings](#2021-07-01-3)
  - [4. ChineseBERT: Chinese Pretraining Enhanced by Glyph and Pinyin Information](#2021-07-01-4)
  - [5. IMS' Systems for the IWSLT 2021 Low-Resource Speech Translation Task](#2021-07-01-5)
  - [6. XLM-E: Cross-lingual Language Model Pre-training via ELECTRA](#2021-07-01-6)
  - [7. On the Power of Saturated Transformers: A View from Circuit Complexity](#2021-07-01-7)
- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MT-Daily_arXiv-index.md)



# 2021-07-05

[Return to Index](#Index)



<h2 id="2021-07-05-1">1. Transformer-F: A Transformer network with effective methods for learning universal sentence representation
</h2>

Title: [Transformer-F: A Transformer network with effective methods for learning universal sentence representation](https://arxiv.org/abs/2107.00653)

Authors: [Yu Shi](https://arxiv.org/search/cs?searchtype=author&query=Shi%2C+Y)

> The Transformer model is widely used in natural language processing for sentence representation. However, the previous Transformer-based models focus on function words that have limited meaning in most cases and could merely extract high-level semantic abstraction features. In this paper, two approaches are introduced to improve the performance of Transformers. We calculated the attention score by multiplying the part-of-speech weight vector with the correlation coefficient, which helps extract the words with more practical meaning. The weight vector is obtained by the input text sequence based on the importance of the part-of-speech. Furthermore, we fuse the features of each layer to make the sentence representation results more comprehensive and accurate. In experiments, we demonstrate the effectiveness of our model Transformer-F on three standard text classification datasets. Experimental results show that our proposed model significantly boosts the performance of text classification as compared to the baseline model. Specifically, we obtain a 5.28% relative improvement over the vanilla Transformer on the simple tasks.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2107.00653](https://arxiv.org/abs/2107.00653) [cs.CL]** |
|           | (or **[arXiv:2107.00653v1](https://arxiv.org/abs/2107.00653v1) [cs.CL]** for this version) |





<h2 id="2021-07-05-2">2. A Primer on Pretrained Multilingual Language Models
</h2>

Title: [A Primer on Pretrained Multilingual Language Models](https://arxiv.org/abs/2107.00676)

Authors: [Sumanth Doddapaneni](https://arxiv.org/search/cs?searchtype=author&query=Doddapaneni%2C+S), [Gowtham Ramesh](https://arxiv.org/search/cs?searchtype=author&query=Ramesh%2C+G), [Anoop Kunchukuttan](https://arxiv.org/search/cs?searchtype=author&query=Kunchukuttan%2C+A), [Pratyush Kumar](https://arxiv.org/search/cs?searchtype=author&query=Kumar%2C+P), [Mitesh M. Khapra](https://arxiv.org/search/cs?searchtype=author&query=Khapra%2C+M+M)

> Multilingual Language Models (MLLMs) such as mBERT, XLM, XLM-R, \textit{etc.} have emerged as a viable option for bringing the power of pretraining to a large number of languages. Given their success in zero shot transfer learning, there has emerged a large body of work in (i) building bigger MLLMs covering a large number of languages (ii) creating exhaustive benchmarks covering a wider variety of tasks and languages for evaluating MLLMs (iii) analysing the performance of MLLMs on monolingual, zero shot crosslingual and bilingual tasks (iv) understanding the universal language patterns (if any) learnt by MLLMs and (v) augmenting the (often) limited capacity of MLLMs to improve their performance on seen or even unseen languages. In this survey, we review the existing literature covering the above broad areas of research pertaining to MLLMs. Based on our survey, we recommend some promising directions of future research.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2107.00676](https://arxiv.org/abs/2107.00676) [cs.CL]** |
|           | (or **[arXiv:2107.00676v1](https://arxiv.org/abs/2107.00676v1) [cs.CL]** for this version) |







<h2 id="2021-07-05-3">3. Interactive decoding of words from visual speech recognition models
</h2>

Title: [Interactive decoding of words from visual speech recognition models](https://arxiv.org/abs/2107.00692)

Authors: [Brendan Shillingford](https://arxiv.org/search/cs?searchtype=author&query=Shillingford%2C+B), [Yannis Assael](https://arxiv.org/search/cs?searchtype=author&query=Assael%2C+Y), [Misha Denil](https://arxiv.org/search/cs?searchtype=author&query=Denil%2C+M)

> This work describes an interactive decoding method to improve the performance of visual speech recognition systems using user input to compensate for the inherent ambiguity of the task. Unlike most phoneme-to-word decoding pipelines, which produce phonemes and feed these through a finite state transducer, our method instead expands words in lockstep, facilitating the insertion of interaction points at each word position. Interaction points enable us to solicit input during decoding, allowing users to interactively direct the decoding process. We simulate the behavior of user input using an oracle to give an automated evaluation, and show promise for the use of this method for text input.

| Comments: | 8 pages                                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2107.00692](https://arxiv.org/abs/2107.00692) [cs.CL]** |
|           | (or **[arXiv:2107.00692v1](https://arxiv.org/abs/2107.00692v1) [cs.CL]** for this version) |







<h2 id="2021-07-05-4">4. Data Centric Domain Adaptation for Historical Text with OCR Errors
</h2>

Title: [Data Centric Domain Adaptation for Historical Text with OCR Errors](https://arxiv.org/abs/2107.00927)

Authors: [Luisa März](https://arxiv.org/search/cs?searchtype=author&query=März%2C+L), [Stefan Schweter](https://arxiv.org/search/cs?searchtype=author&query=Schweter%2C+S), [Nina Poerner](https://arxiv.org/search/cs?searchtype=author&query=Poerner%2C+N), [Benjamin Roth](https://arxiv.org/search/cs?searchtype=author&query=Roth%2C+B), [Hinrich Schütze](https://arxiv.org/search/cs?searchtype=author&query=Schütze%2C+H)

> We propose new methods for in-domain and cross-domain Named Entity Recognition (NER) on historical data for Dutch and French. For the cross-domain case, we address domain shift by integrating unsupervised in-domain data via contextualized string embeddings; and OCR errors by injecting synthetic OCR errors into the source domain and address data centric domain adaptation. We propose a general approach to imitate OCR errors in arbitrary input data. Our cross-domain as well as our in-domain results outperform several strong baselines and establish state-of-the-art results. We publish preprocessed versions of the French and Dutch Europeana NER corpora.

| Comments: | 14 pages, 2 figures, 6 tables                                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2107.00927](https://arxiv.org/abs/2107.00927) [cs.CL]** |
|           | (or **[arXiv:2107.00927v1](https://arxiv.org/abs/2107.00927v1) [cs.CL]** for this version) |







# 2021-07-02

[Return to Index](#Index)



<h2 id="2021-07-02-1">1. GlyphCRM: Bidirectional Encoder Representation for Chinese Character with its Glyph
</h2>

Title: [GlyphCRM: Bidirectional Encoder Representation for Chinese Character with its Glyph](https://arxiv.org/abs/2107.00395)

Authors: [Yunxin Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Y), [Yu Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+Y), [Baotian Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu%2C+B), [Qingcai Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Q), [Yang Xiang](https://arxiv.org/search/cs?searchtype=author&query=Xiang%2C+Y), [Xiaolong Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Yuxin Ding](https://arxiv.org/search/cs?searchtype=author&query=Ding%2C+Y), [Lin Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+L)

> Previous works indicate that the glyph of Chinese characters contains rich semantic information and has the potential to enhance the representation of Chinese characters. The typical method to utilize the glyph features is by incorporating them into the character embedding space. Inspired by previous methods, we innovatively propose a Chinese pre-trained representation model named as GlyphCRM, which abandons the ID-based character embedding method yet solely based on sequential character images. We render each character into a binary grayscale image and design two-channel position feature maps for it. Formally, we first design a two-layer residual convolutional neural network, namely HanGlyph to generate the initial glyph representation of Chinese characters, and subsequently adopt multiple bidirectional encoder Transformer blocks as the superstructure to capture the context-sensitive information. Meanwhile, we feed the glyph features extracted from each layer of the HanGlyph module into the underlying Transformer blocks by skip-connection method to fully exploit the glyph features of Chinese characters. As the HanGlyph module can obtain a sufficient glyph representation of any Chinese character, the long-standing out-of-vocabulary problem could be effectively solved. Extensive experimental results indicate that GlyphCRM substantially outperforms the previous BERT-based state-of-the-art model on 9 fine-tuning tasks, and it has strong transferability and generalization on specialized fields and low-resource tasks. We hope this work could spark further research beyond the realms of well-established representation of Chinese texts.

| Comments: | 11 pages, 7 figures                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Artificial Intelligence (cs.AI)**; Computation and Language (cs.CL); Computer Vision and Pattern Recognition (cs.CV) |
| Cite as:  | **[arXiv:2107.00395](https://arxiv.org/abs/2107.00395) [cs.AI]** |
|           | (or **[arXiv:2107.00395v1](https://arxiv.org/abs/2107.00395v1) [cs.AI]** for this version) |





<h2 id="2021-07-02-2">2. ESPnet-ST IWSLT 2021 Offline Speech Translation System
</h2>

Title: [ESPnet-ST IWSLT 2021 Offline Speech Translation System](https://arxiv.org/abs/2107.00636)

Authors: [Hirofumi Inaguma](https://arxiv.org/search/eess?searchtype=author&query=Inaguma%2C+H), [Brian Yan](https://arxiv.org/search/eess?searchtype=author&query=Yan%2C+B), [Siddharth Dalmia](https://arxiv.org/search/eess?searchtype=author&query=Dalmia%2C+S), [Pengcheng Gu](https://arxiv.org/search/eess?searchtype=author&query=Gu%2C+P), [Jiatong Shi](https://arxiv.org/search/eess?searchtype=author&query=Shi%2C+J), [Kevin Duh](https://arxiv.org/search/eess?searchtype=author&query=Duh%2C+K), [Shinji Watanabe](https://arxiv.org/search/eess?searchtype=author&query=Watanabe%2C+S)

> This paper describes the ESPnet-ST group's IWSLT 2021 submission in the offline speech translation track. This year we made various efforts on training data, architecture, and audio segmentation. On the data side, we investigated sequence-level knowledge distillation (SeqKD) for end-to-end (E2E) speech translation. Specifically, we used multi-referenced SeqKD from multiple teachers trained on different amounts of bitext. On the architecture side, we adopted the Conformer encoder and the Multi-Decoder architecture, which equips dedicated decoders for speech recognition and translation tasks in a unified encoder-decoder model and enables search in both source and target language spaces during inference. We also significantly improved audio segmentation by using the pyannote.audio toolkit and merging multiple short segments for long context modeling. Experimental evaluations showed that each of them contributed to large improvements in translation performance. Our best E2E system combined all the above techniques with model ensembling and achieved 31.4 BLEU on the 2-ref of tst2021 and 21.2 BLEU and 19.3 BLEU on the two single references of tst2021.

| Comments: | IWSLT 2021                                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Audio and Speech Processing (eess.AS)**; Computation and Language (cs.CL); Sound (cs.SD) |
| Cite as:  | **[arXiv:2107.00636](https://arxiv.org/abs/2107.00636) [eess.AS]** |
|           | (or **[arXiv:2107.00636v1](https://arxiv.org/abs/2107.00636v1) [eess.AS]** for this version) |





<h2 id="2021-07-02-3">3. Word-Free Spoken Language Understanding for Mandarin-Chinese
</h2>

Title: [Word-Free Spoken Language Understanding for Mandarin-Chinese](https://arxiv.org/abs/2107.00186)

Authors: [Zhiyuan Guo](https://arxiv.org/search/cs?searchtype=author&query=Guo%2C+Z), [Yuexin Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Y), [Guo Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+G), [Xingyu Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+X), [Akshat Gupta](https://arxiv.org/search/cs?searchtype=author&query=Gupta%2C+A)

> Spoken dialogue systems such as Siri and Alexa provide great convenience to people's everyday life. However, current spoken language understanding (SLU) pipelines largely depend on automatic speech recognition (ASR) modules, which require a large amount of language-specific training data. In this paper, we propose a Transformer-based SLU system that works directly on phones. This acoustic-based SLU system consists of only two blocks and does not require the presence of ASR module. The first block is a universal phone recognition system, and the second block is a Transformer-based language model for phones. We verify the effectiveness of the system on an intent classification dataset in Mandarin Chinese.

| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2107.00186](https://arxiv.org/abs/2107.00186) [cs.CL]** |
|           | (or **[arXiv:2107.00186v1](https://arxiv.org/abs/2107.00186v1) [cs.CL]** for this version) |





<h2 id="2021-07-02-4">4. The USTC-NELSLIP Systems for Simultaneous Speech Translation Task at IWSLT 2021 </h2>



Title: [The USTC-NELSLIP Systems for Simultaneous Speech Translation Task at IWSLT 2021]()

Authors: [Dan Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+D), [Mengge Du](https://arxiv.org/search/cs?searchtype=author&query=Du%2C+M), [Xiaoxi Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+X), [Yuchen Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu%2C+Y), [Lirong Dai](https://arxiv.org/search/cs?searchtype=author&query=Dai%2C+L)

> This paper describes USTC-NELSLIP's submissions to the IWSLT2021 Simultaneous Speech Translation task. We proposed a novel simultaneous translation model, Cross Attention Augmented Transducer (CAAT), which extends conventional RNN-T to sequence-to-sequence tasks without monotonic constraints, e.g., simultaneous translation. Experiments on speech-to-text (S2T) and text-to-text (T2T) simultaneous translation tasks shows CAAT achieves better quality-latency trade-offs compared to \textit{wait-k}, one of the previous state-of-the-art approaches. Based on CAAT architecture and data augmentation, we build S2T and T2T simultaneous translation systems in this evaluation campaign. Compared to last year's optimal systems, our S2T simultaneous translation system improves by an average of 11.3 BLEU for all latency regimes, and our T2T simultaneous translation system improves by an average of 4.6 BLEU.

| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2107.00279](https://arxiv.org/abs/2107.00279) [cs.CL]** |
|           | (or **[arXiv:2107.00279v1](https://arxiv.org/abs/2107.00279v1) [cs.CL]** for this version) |





<h2 id="2021-07-02-5">5. Zero-pronoun Data Augmentation for Japanese-to-English Translation
</h2>

Title: [Zero-pronoun Data Augmentation for Japanese-to-English Translation](https://arxiv.org/abs/2107.00318)

Authors: [Ryokan Ri](https://arxiv.org/search/cs?searchtype=author&query=Ri%2C+R), [Toshiaki Nakazawa](https://arxiv.org/search/cs?searchtype=author&query=Nakazawa%2C+T), [Yoshimasa Tsuruoka](https://arxiv.org/search/cs?searchtype=author&query=Tsuruoka%2C+Y)

> For Japanese-to-English translation, zero pronouns in Japanese pose a challenge, since the model needs to infer and produce the corresponding pronoun in the target side of the English sentence. However, although fully resolving zero pronouns often needs discourse context, in some cases, the local context within a sentence gives clues to the inference of the zero pronoun. In this study, we propose a data augmentation method that provides additional training signals for the translation model to learn correlations between local context and zero pronouns. We show that the proposed method significantly improves the accuracy of zero pronoun translation with machine translation experiments in the conversational domain.

| Comments: | WAT2021                                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2107.00318](https://arxiv.org/abs/2107.00318) [cs.CL]** |
|           | (or **[arXiv:2107.00318v1](https://arxiv.org/abs/2107.00318v1) [cs.CL]** for this version) |





<h2 id="2021-07-02-6">6. Modeling Target-side Inflection in Placeholder Translation
</h2>

Title: [Modeling Target-side Inflection in Placeholder Translation](https://arxiv.org/abs/2107.00334)

Authors: [Ryokan Ri](https://arxiv.org/search/cs?searchtype=author&query=Ri%2C+R), [Toshiaki Nakazawa](https://arxiv.org/search/cs?searchtype=author&query=Nakazawa%2C+T), [Yoshimasa Tsuruoka](https://arxiv.org/search/cs?searchtype=author&query=Tsuruoka%2C+Y)

> Placeholder translation systems enable the users to specify how a specific phrase is translated in the output sentence. The system is trained to output special placeholder tokens, and the user-specified term is injected into the output through the context-free replacement of the placeholder token. However, this approach could result in ungrammatical sentences because it is often the case that the specified term needs to be inflected according to the context of the output, which is unknown before the translation. To address this problem, we propose a novel method of placeholder translation that can inflect specified terms according to the grammatical construction of the output sentence. We extend the sequence-to-sequence architecture with a character-level decoder that takes the lemma of a user-specified term and the words generated from the word-level decoder to output the correct inflected form of the lemma. We evaluate our approach with a Japanese-to-English translation task in the scientific writing domain, and show that our model can incorporate specified terms in the correct form more successfully than other comparable models.

| Comments: | MT Summit 2021                                               |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2107.00334](https://arxiv.org/abs/2107.00334) [cs.CL]** |
|           | (or **[arXiv:2107.00334v1](https://arxiv.org/abs/2107.00334v1) [cs.CL]** for this version) |





<h2 id="2021-07-02-7">7. CLINE: Contrastive Learning with Semantic Negative Examples for Natural Language Understanding
</h2>

Title: [CLINE: Contrastive Learning with Semantic Negative Examples for Natural Language Understanding](https://arxiv.org/abs/2107.00440)

Authors: [Dong Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+D), [Ning Ding](https://arxiv.org/search/cs?searchtype=author&query=Ding%2C+N), [Piji Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+P), [Hai-Tao Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+H)

> Despite pre-trained language models have proven useful for learning high-quality semantic representations, these models are still vulnerable to simple perturbations. Recent works aimed to improve the robustness of pre-trained models mainly focus on adversarial training from perturbed examples with similar semantics, neglecting the utilization of different or even opposite semantics. Different from the image processing field, the text is discrete and few word substitutions can cause significant semantic changes. To study the impact of semantics caused by small perturbations, we conduct a series of pilot experiments and surprisingly find that adversarial training is useless or even harmful for the model to detect these semantic changes. To address this problem, we propose Contrastive Learning with semantIc Negative Examples (CLINE), which constructs semantic negative examples unsupervised to improve the robustness under semantically adversarial attacking. By comparing with similar and opposite semantic examples, the model can effectively perceive the semantic changes caused by small perturbations. Empirical results show that our approach yields substantial improvements on a range of sentiment analysis, reasoning, and reading comprehension tasks. And CLINE also ensures the compactness within the same semantics and separability across different semantics in sentence-level.

| Comments: | ACL 2021, Main Conference, Long Paper                        |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| Cite as:  | **[arXiv:2107.00440](https://arxiv.org/abs/2107.00440) [cs.CL]** |
|           | (or **[arXiv:2107.00440v1](https://arxiv.org/abs/2107.00440v1) [cs.CL]** for this version) |








# 2021-07-01

[Return to Index](#Index)



<h2 id="2021-07-01-1">1. What Can Unsupervised Machine Translation Contribute to High-Resource Language Pairs?
</h2>

Title: [What Can Unsupervised Machine Translation Contribute to High-Resource Language Pairs?](https://arxiv.org/abs/2106.15818)

Authors: [Kelly Marchisio](https://arxiv.org/search/cs?searchtype=author&query=Marchisio%2C+K), [Markus Freitag](https://arxiv.org/search/cs?searchtype=author&query=Freitag%2C+M), [David Grangier](https://arxiv.org/search/cs?searchtype=author&query=Grangier%2C+D)

> Whereas existing literature on unsupervised machine translation (MT) focuses on exploiting unsupervised techniques for low-resource language pairs where bilingual training data is scare or unavailable, we investigate whether unsupervised MT can also improve translation quality of high-resource language pairs where sufficient bitext does exist. We compare the style of correct translations generated by either supervised or unsupervised MT and find that the unsupervised output is less monotonic and more natural than supervised output. We demonstrate a way to combine the benefits of unsupervised and supervised MT into a single system, resulting in better human evaluation of quality and fluency. Our results open the door to discussions about the potential contributions of unsupervised MT in high-resource settings, and how supervised and unsupervised systems might be mutually-beneficial.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2106.15818](https://arxiv.org/abs/2106.15818) [cs.CL]** |
|           | (or **[arXiv:2106.15818v1](https://arxiv.org/abs/2106.15818v1) [cs.CL]** for this version) |





<h2 id="2021-07-01-2">2. Mixed Cross Entropy Loss for Neural Machine Translation
</h2>

Title: [Mixed Cross Entropy Loss for Neural Machine Translation](https://arxiv.org/abs/2106.15880)

Authors: [Haoran Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+H), [Wei Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu%2C+W)

> In neural machine translation, cross entropy (CE) is the standard loss function in two training methods of auto-regressive models, i.e., teacher forcing and scheduled sampling. In this paper, we propose mixed cross entropy loss (mixed CE) as a substitute for CE in both training approaches. In teacher forcing, the model trained with CE regards the translation problem as a one-to-one mapping process, while in mixed CE this process can be relaxed to one-to-many. In scheduled sampling, we show that mixed CE has the potential to encourage the training and testing behaviours to be similar to each other, more effectively mitigating the exposure bias problem. We demonstrate the superiority of mixed CE over CE on several machine translation datasets, WMT'16 Ro-En, WMT'16 Ru-En, and WMT'14 En-De in both teacher forcing and scheduled sampling setups. Furthermore, in WMT'14 En-De, we also find mixed CE consistently outperforms CE on a multi-reference set as well as a challenging paraphrased reference set. We also found the model trained with mixed CE is able to provide a better probability distribution defined over the translation output space. Our code is available at [this https URL](https://github.com/haorannlp/mix).

| Subjects:          | **Computation and Language (cs.CL)**                         |
| ------------------ | ------------------------------------------------------------ |
| Journal reference: | ICML2021                                                     |
| Cite as:           | **[arXiv:2106.15880](https://arxiv.org/abs/2106.15880) [cs.CL]** |
|                    | (or **[arXiv:2106.15880v1](https://arxiv.org/abs/2106.15880v1) [cs.CL]** for this version) |





<h2 id="2021-07-01-3">3. Cross-lingual alignments of ELMo contextual embeddings
</h2>

Title: [Cross-lingual alignments of ELMo contextual embeddings](https://arxiv.org/abs/2106.15986)

Authors: [Matej Ulčar](https://arxiv.org/search/cs?searchtype=author&query=Ulčar%2C+M), [Marko Robnik-Šikonja](https://arxiv.org/search/cs?searchtype=author&query=Robnik-Šikonja%2C+M)

> Building machine learning prediction models for a specific NLP task requires sufficient training data, which can be difficult to obtain for low-resource languages. Cross-lingual embeddings map word embeddings from a low-resource language to a high-resource language so that a prediction model trained on data from the high-resource language can also be used in the low-resource language. To produce cross-lingual mappings of recent contextual embeddings, anchor points between the embedding spaces have to be words in the same context. We address this issue with a new method for creating datasets for cross-lingual contextual alignments. Based on that, we propose novel cross-lingual mapping methods for ELMo embeddings. Our linear mapping methods use existing vecmap and MUSE alignments on contextual ELMo embeddings. Our new nonlinear ELMoGAN mapping method is based on GANs and does not assume isomorphic embedding spaces. We evaluate the proposed mapping methods on nine languages, using two downstream tasks, NER and dependency parsing. The ELMoGAN method performs well on the NER task, with low cross-lingual loss compared to direct training on some languages. In the dependency parsing, linear alignment variants are more successful.

| Comments: | 26 pages, 5 figures                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2106.15986](https://arxiv.org/abs/2106.15986) [cs.CL]** |
|           | (or **[arXiv:2106.15986v1](https://arxiv.org/abs/2106.15986v1) [cs.CL]** for this version) |





<h2 id="2021-07-01-4">4. ChineseBERT: Chinese Pretraining Enhanced by Glyph and Pinyin Information
</h2>

Title: [ChineseBERT: Chinese Pretraining Enhanced by Glyph and Pinyin Information](https://arxiv.org/abs/2106.16038)

Authors: [Zijun Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+Z), [Xiaoya Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+X), [Xiaofei Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+X), [Yuxian Meng](https://arxiv.org/search/cs?searchtype=author&query=Meng%2C+Y), [Xiang Ao](https://arxiv.org/search/cs?searchtype=author&query=Ao%2C+X), [Qing He](https://arxiv.org/search/cs?searchtype=author&query=He%2C+Q), [Fei Wu](https://arxiv.org/search/cs?searchtype=author&query=Wu%2C+F), [Jiwei Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+J)

> Recent pretraining models in Chinese neglect two important aspects specific to the Chinese language: glyph and pinyin, which carry significant syntax and semantic information for language understanding. In this work, we propose ChineseBERT, which incorporates both the {\it glyph} and {\it pinyin} information of Chinese characters into language model pretraining. The glyph embedding is obtained based on different fonts of a Chinese character, being able to capture character semantics from the visual features, and the pinyin embedding characterizes the pronunciation of Chinese characters, which handles the highly prevalent heteronym phenomenon in Chinese (the same character has different pronunciations with different meanings). Pretrained on large-scale unlabeled Chinese corpus, the proposed ChineseBERT model yields significant performance boost over baseline models with fewer training steps. The porpsoed model achieves new SOTA performances on a wide range of Chinese NLP tasks, including machine reading comprehension, natural language inference, text classification, sentence pair matching, and competitive performances in named entity recognition. Code and pretrained models are publicly available at [this https URL](https://github.com/ShannonAI/ChineseBert).

| Comments: | To appear at ACL2021                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2106.16038](https://arxiv.org/abs/2106.16038) [cs.CL]** |
|           | (or **[arXiv:2106.16038v1](https://arxiv.org/abs/2106.16038v1) [cs.CL]** for this version) |





<h2 id="2021-07-01-5">5. IMS' Systems for the IWSLT 2021 Low-Resource Speech Translation Task
</h2>

Title: [IMS' Systems for the IWSLT 2021 Low-Resource Speech Translation Task](https://arxiv.org/abs/2106.16055)

Authors: [Pavel Denisov](https://arxiv.org/search/cs?searchtype=author&query=Denisov%2C+P), [Manuel Mager](https://arxiv.org/search/cs?searchtype=author&query=Mager%2C+M), [Ngoc Thang Vu](https://arxiv.org/search/cs?searchtype=author&query=Vu%2C+N+T)

> This paper describes the submission to the IWSLT 2021 Low-Resource Speech Translation Shared Task by IMS team. We utilize state-of-the-art models combined with several data augmentation, multi-task and transfer learning approaches for the automatic speech recognition (ASR) and machine translation (MT) steps of our cascaded system. Moreover, we also explore the feasibility of a full end-to-end speech translation (ST) model in the case of very constrained amount of ground truth labeled data. Our best system achieves the best performance among all submitted systems for Congolese Swahili to English and French with BLEU scores 7.7 and 13.7 respectively, and the second best result for Coastal Swahili to English with BLEU score 14.9.

| Comments: | IWSLT 2021                                                   |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2106.16055](https://arxiv.org/abs/2106.16055) [cs.CL]** |
|           | (or **[arXiv:2106.16055v1](https://arxiv.org/abs/2106.16055v1) [cs.CL]** for this version) |





<h2 id="2021-07-01-6">6. XLM-E: Cross-lingual Language Model Pre-training via ELECTRA
</h2>

Title: [XLM-E: Cross-lingual Language Model Pre-training via ELECTRA](https://arxiv.org/abs/2106.16138)

Authors: [Zewen Chi](https://arxiv.org/search/cs?searchtype=author&query=Chi%2C+Z), [Shaohan Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+S), [Li Dong](https://arxiv.org/search/cs?searchtype=author&query=Dong%2C+L), [Shuming Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+S), [Saksham Singhal](https://arxiv.org/search/cs?searchtype=author&query=Singhal%2C+S), [Payal Bajaj](https://arxiv.org/search/cs?searchtype=author&query=Bajaj%2C+P), [Xia Song](https://arxiv.org/search/cs?searchtype=author&query=Song%2C+X), [Furu Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+F)

> In this paper, we introduce ELECTRA-style tasks to cross-lingual language model pre-training. Specifically, we present two pre-training tasks, namely multilingual replaced token detection, and translation replaced token detection. Besides, we pretrain the model, named as XLM-E, on both multilingual and parallel corpora. Our model outperforms the baseline models on various cross-lingual understanding tasks with much less computation cost. Moreover, analysis shows that XLM-E tends to obtain better cross-lingual transferability.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2106.16138](https://arxiv.org/abs/2106.16138) [cs.CL]** |
|           | (or **[arXiv:2106.16138v1](https://arxiv.org/abs/2106.16138v1) [cs.CL]** for this version) |





<h2 id="2021-07-01-7">7. On the Power of Saturated Transformers: A View from Circuit Complexity
</h2>

Title: [On the Power of Saturated Transformers: A View from Circuit Complexity](https://arxiv.org/abs/2106.16213)

Authors: [William Merrill](https://arxiv.org/search/cs?searchtype=author&query=Merrill%2C+W), [Yoav Goldberg](https://arxiv.org/search/cs?searchtype=author&query=Goldberg%2C+Y), [Roy Schwartz](https://arxiv.org/search/cs?searchtype=author&query=Schwartz%2C+R), [Noah A. Smith](https://arxiv.org/search/cs?searchtype=author&query=Smith%2C+N+A)

> Transformers have become a standard architecture for many NLP problems. This has motivated theoretically analyzing their capabilities as models of language, in order to understand what makes them successful, and what their potential weaknesses might be. Recent work has shown that transformers with hard attention are quite limited in capacity, and in fact can be simulated by constant-depth circuits. However, hard attention is a restrictive assumption, which may complicate the relevance of these results for practical transformers. In this work, we analyze the circuit complexity of transformers with saturated attention: a generalization of hard attention that more closely captures the attention patterns learnable in practical transformers. We show that saturated transformers transcend the limitations of hard-attention transformers. With some minor assumptions, we prove that the number of bits needed to represent a saturated transformer memory vector is O(logn), which implies saturated transformers can be simulated by log-depth circuits. Thus, the jump from hard to saturated attention can be understood as increasing the transformer's effective circuit depth by a factor of O(logn).

| Comments: | Preprint                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Computational Complexity (cs.CC); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2106.16213](https://arxiv.org/abs/2106.16213) [cs.CL]** |
|           | (or **[arXiv:2106.16213v1](https://arxiv.org/abs/2106.16213v1) [cs.CL]** for this version) |



