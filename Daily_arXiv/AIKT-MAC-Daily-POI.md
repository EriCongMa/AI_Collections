# MA C.'s Daily Paper Of Interest - June a., 2022

# Index

- [2022-06-15](#2022-06-15)
  - [1. LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning](#2022-06-15-1)
  
  - [2. FreeTransfer-X: Safe and Label-Free Cross-Lingual Transfer from Off-the-Shelf Models](#2022-06-15-2)
  
- [2022-06-14](#2022-06-14)
  - [1. A Unified Continuous Learning Framework for Multi-modal Knowledge Discovery and Pre-training](#2022-06-14-1)

  - [2. The YiTrans End-to-End Speech Translation System for IWSLT 2022 Offline Shared Task](#2022-06-14-2)

  - [3. On the Learning of Non-Autoregressive Transformers](#2022-06-14-3)

  - [4. Language Models are General-Purpose Interfaces](#2022-06-14-4)

- [2022-06-13](#2022-06-13)
  - [1. A Novel Chinese Dialect TTS Frontend with Non-Autoregressive Neural Machine Translation](#2022-06-13-1)

- [2022-06-10](#2022-06-10)
  - [1. Dict-NMT: Bilingual Dictionary based NMT for Extremely Low Resource Languages](#2022-06-10-1)
  - [2. Joint Encoder-Decoder Self-Supervised Pre-training for ASR](#2022-06-10-2)
  - [3. Revisiting End-to-End Speech-to-Text Translation From Scratch](#2022-06-10-3)

- [2022-06-09](#2022-06-09)
  - [1. TURJUMAN: A Public Toolkit for Neural Arabic Machine Translation](#2022-06-09-1)

  - [2. STable: Table Generation Framework for Encoder-Decoder Models](#2022-06-09-2)

- [2022-06-08](#2022-06-08)
  - [1. Tutel: Adaptive Mixture-of-Experts at Scale](#2022-06-08-1)

  - [2. LegoNN: Building Modular Encoder-Decoder Models](#2022-06-08-2)

  - [3. cViL: Cross-Lingual Training of Vision-Language Models using Knowledge Distillation](#2022-06-08-3)

- [2022-06-07](#2022-06-07)
  - [1. Rethinking the Openness of CLIP](#2022-06-07-1)

  - [2. Instance-wise Prompt Tuning for Pretrained Language Models](#2022-06-07-2)

  - [3. Multilingual Neural Machine Translation with Deep Encoder and Multiple Shallow Decoders](#2022-06-07-3)
  - [4. MorisienMT: A Dataset for Mauritian Creole Machine Translation](#2022-06-07-4)

- [2022-06-06](#2022-06-06)
  - [1. A-OKVQA: A Benchmark for Visual Question Answering using World Knowledge](#2022-06-06-1)
- [2022-06-03](#2022-06-03)
  - [1. Squeezeformer: An Efficient Transformer for Automatic Speech Recognition](#2022-06-03-1)

  - [2. VL-BEiT: Generative Vision-Language Pretraining](#2022-06-03-2)
  - [3. BayesFormer: Transformer with Uncertainty Estimation](#2022-06-03-3)
  - [4. Finding the Right Recipe for Low Resource Domain Adaptation in Neural Machine Translation](#2022-06-03-4)
- [2022-06-02](#2022-06-02)
  - [1. VALHALLA: Visual Hallucination for Machine Translation](#2022-06-02-1)

  - [2. Discovering the Hidden Vocabulary of DALLE-2](#2022-06-02-2)

  - [3. On Layer Normalizations and Residual Connections in Transformers](#2022-06-02-3)

  - [4. Optical character recognition quality affects perceived usefulness of historical newspaper clippings](#2022-06-02-4)

  - [5. Exploring Diversity in Back Translation for Low-Resource Machine Translation](#2022-06-02-5)

  - [6. Cross-View Language Modeling: Towards Unified Cross-Lingual Cross-Modal Pre-training](#2022-06-02-6)
- [2022-06-01](#2022-06-01)
  - [1. Parameter-Efficient and Student-Friendly Knowledge Distillation](#2022-06-01-1)

  - [2. ADAPT: Vision-Language Navigation with Modality-Aligned Action Prompts](#2022-06-01-2)

  - [3. CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers](#2022-06-01-3)

  - [4. EMS: Efficient and Effective Massively Multilingual Sentence Representation Learning](#2022-06-01-4)
- [2022-05-31](#2022-05-31)
  - [1. VLUE: A Multi-Task Benchmark for Evaluating Vision-Language Models](#2022-05-31-1)
- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MAC-Daily-POI-index.md)



# 2022-06-15

[Return to Index](#Index)



<h2 id="2022-06-15-1">1. LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning
</h2>

Title: [LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning](https://arxiv.org/abs/2206.06522)

Authors: [Yi-Lin Sung](https://arxiv.org/search/cs?searchtype=author&query=Sung%2C+Y), [Jaemin Cho](https://arxiv.org/search/cs?searchtype=author&query=Cho%2C+J), [Mohit Bansal](https://arxiv.org/search/cs?searchtype=author&query=Bansal%2C+M)

> Fine-tuning large pre-trained models on downstream tasks has been adopted in a variety of domains recently. However, it is costly to update the entire parameter set of large pre-trained models. Although recently proposed parameter-efficient transfer learning (PETL) techniques allow updating a small subset of parameters (e.g. only using 2% of parameters) inside a pre-trained backbone network for a new task, they only reduce the training memory requirement by up to 30%. This is because the gradient computation for the trainable parameters still requires backpropagation through the large pre-trained backbone model. To address this, we propose Ladder Side-Tuning (LST), a new PETL technique that reduces training memory requirements by more substantial amounts. Unlike existing parameter-efficient methods that insert additional parameters inside backbone networks, we train a ladder side network, a small and separate network that takes intermediate activations as input via shortcut connections (ladders) from backbone networks and makes predictions. LST has significantly lower memory requirements than previous methods, because it does not require backpropagation through the backbone network, but instead only through the side network and ladder connections. We evaluate our method with various models (T5, CLIP-T5) on both NLP (GLUE) and vision-language (VQA, GQA, NLVR2, MSCOCO) tasks. LST saves 69% of the memory costs to fine-tune the whole network, while other methods only save 26% of that in similar parameter usages (hence, 2.7x more memory savings). Moreover, LST achieves higher accuracy than Adapter and LoRA in a low-memory regime. To further show the advantage of this better memory efficiency, we also apply LST to larger T5 models (T5-large, T5-3B), attaining better GLUE performance than full fine-tuning and other PETL methods. The exact same trend also holds in our experiments on VL tasks.

| Comments: | 13 pages; our code is available at: [this https URL](https://github.com/ylsung/Ladder-Side-Tuning) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV) |
| Cite as:  | **[arXiv:2206.06522](https://arxiv.org/abs/2206.06522) [cs.CL]** |
|           | (or **[arXiv:2206.06522v1](https://arxiv.org/abs/2206.06522v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.06522Focus to learn more |





<h2 id="2022-06-15-2">2. FreeTransfer-X: Safe and Label-Free Cross-Lingual Transfer from Off-the-Shelf Models
</h2>

Title: [FreeTransfer-X: Safe and Label-Free Cross-Lingual Transfer from Off-the-Shelf Models](https://arxiv.org/abs/2206.06586)

Authors: [Yinpeng Guo](https://arxiv.org/search/cs?searchtype=author&query=Guo%2C+Y), [Liangyou Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+L), [Xin Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+X), [Qun Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Q)

> Cross-lingual transfer (CLT) is of various applications. However, labeled cross-lingual corpus is expensive or even inaccessible, especially in the fields where labels are private, such as diagnostic results of symptoms in medicine and user profiles in business. Nevertheless, there are off-the-shelf models in these sensitive fields. Instead of pursuing the original labels, a workaround for CLT is to transfer knowledge from the off-the-shelf models without labels. To this end, we define a novel CLT problem named FreeTransfer-X that aims to achieve knowledge transfer from the off-the-shelf models in rich-resource languages. To address the problem, we propose a 2-step knowledge distillation (KD, Hinton et al., 2015) framework based on multilingual pre-trained language models (mPLM). The significant improvement over strong neural machine translation (NMT) baselines demonstrates the effectiveness of the proposed method. In addition to reducing annotation cost and protecting private labels, the proposed method is compatible with different networks and easy to be deployed. Finally, a range of analyses indicate the great potential of the proposed method.

| Comments: | to appear in the findings of NAACL 2022                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2206.06586](https://arxiv.org/abs/2206.06586) [cs.CL]** |
|           | (or **[arXiv:2206.06586v1](https://arxiv.org/abs/2206.06586v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.06586Focus to learn more |





# 2022-06-14

[Return to Index](#Index)



<h2 id="2022-06-14-1">1. A Unified Continuous Learning Framework for Multi-modal Knowledge Discovery and Pre-training
</h2>

Title: [A Unified Continuous Learning Framework for Multi-modal Knowledge Discovery and Pre-training](https://arxiv.org/abs/2206.05555)

Authors: [Zhihao Fan](https://arxiv.org/search/cs?searchtype=author&query=Fan%2C+Z), [Zhongyu Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+Z), [Jingjing Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+J), [Siyuan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+S), [Zejun Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Z), [Jiarong Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+J), [Xuanjing Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+X)

> Multi-modal pre-training and knowledge discovery are two important research topics in multi-modal machine learning. Nevertheless, none of existing works make attempts to link knowledge discovery with knowledge guided multi-modal pre-training. In this paper, we propose to unify them into a continuous learning framework for mutual improvement. Taking the open-domain uni-modal datasets of images and texts as input, we maintain a knowledge graph as the foundation to support these two tasks. For knowledge discovery, a pre-trained model is used to identify cross-modal links on the graph. For model pre-training, the knowledge graph is used as the external knowledge to guide the model updating. These two steps are iteratively performed in our framework for continuous learning. The experimental results on MS-COCO and Flickr30K with respect to both knowledge discovery and the pre-trained model validate the effectiveness of our framework.

| Subjects: | **Computation and Language (cs.CL)**; Computer Vision and Pattern Recognition (cs.CV) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.05555](https://arxiv.org/abs/2206.05555) [cs.CL]** |
|           | (or **[arXiv:2206.05555v1](https://arxiv.org/abs/2206.05555v1) [cs.CL]** for this version) |





<h2 id="2022-06-14-2">2. The YiTrans End-to-End Speech Translation System for IWSLT 2022 Offline Shared Task
</h2>

Title: [The YiTrans End-to-End Speech Translation System for IWSLT 2022 Offline Shared Task](https://arxiv.org/abs/2206.05777)

Authors: [Ziqiang Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+Z), [Junyi Ao](https://arxiv.org/search/cs?searchtype=author&query=Ao%2C+J), [Shujie Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+S), [Furu Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+F), [Jinyu Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+J)

> This paper describes the submission of our end-to-end YiTrans speech translation system for the IWSLT 2022 offline task, which translates from English audio to German, Chinese, and Japanese. The YiTrans system is built on large-scale pre-trained encoder-decoder models. More specifically, we first design a multi-stage pre-training strategy to build a multi-modality model with a large amount of labeled and unlabeled data. We then fine-tune the corresponding components of the model for the downstream speech translation tasks. Moreover, we make various efforts to improve performance, such as data filtering, data augmentation, speech segmentation, model ensemble, and so on. Experimental results show that our YiTrans system obtains a significant improvement than the strong baseline on three translation directions, and it achieves +5.2 BLEU improvements over last year's optimal end-to-end system on tst2021 English-German. Our final submissions rank first on English-German and English-Chinese end-to-end systems in terms of the automatic evaluation metric. We make our code and models publicly available.

| Comments: | 11 pages                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2206.05777](https://arxiv.org/abs/2206.05777) [cs.CL]** |
|           | (or **[arXiv:2206.05777v1](https://arxiv.org/abs/2206.05777v1) [cs.CL]** for this version) |





<h2 id="2022-06-14-3">3. On the Learning of Non-Autoregressive Transformers
</h2>

Title: [On the Learning of Non-Autoregressive Transformers](https://arxiv.org/abs/2206.05975)

Authors: [Fei Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+F), [Tianhua Tao](https://arxiv.org/search/cs?searchtype=author&query=Tao%2C+T), [Hao Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+H), [Lei Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+L), [Minlie Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+M)

> Non-autoregressive Transformer (NAT) is a family of text generation models, which aims to reduce the decoding latency by predicting the whole sentences in parallel. However, such latency reduction sacrifices the ability to capture left-to-right dependencies, thereby making NAT learning very challenging. In this paper, we present theoretical and empirical analyses to reveal the challenges of NAT learning and propose a unified perspective to understand existing successes. First, we show that simply training NAT by maximizing the likelihood can lead to an approximation of marginal distributions but drops all dependencies between tokens, where the dropped information can be measured by the dataset's conditional total correlation. Second, we formalize many previous objectives in a unified framework and show that their success can be concluded as maximizing the likelihood on a proxy distribution, leading to a reduced information loss. Empirical studies show that our perspective can explain the phenomena in NAT learning and guide the design of new training methods.

| Comments: | accepted at ICML2022                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2206.05975](https://arxiv.org/abs/2206.05975) [cs.CL]** |
|           | (or **[arXiv:2206.05975v1](https://arxiv.org/abs/2206.05975v1) [cs.CL]** for this version) |





<h2 id="2022-06-14-4">4. Language Models are General-Purpose Interfaces
</h2>

Title: [Language Models are General-Purpose Interfaces](https://arxiv.org/abs/2206.06336)

Authors: [Yaru Hao](https://arxiv.org/search/cs?searchtype=author&query=Hao%2C+Y), [Haoyu Song](https://arxiv.org/search/cs?searchtype=author&query=Song%2C+H), [Li Dong](https://arxiv.org/search/cs?searchtype=author&query=Dong%2C+L), [Shaohan Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+S), [Zewen Chi](https://arxiv.org/search/cs?searchtype=author&query=Chi%2C+Z), [Wenhui Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+W), [Shuming Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+S), [Furu Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+F)

> Foundation models have received much attention due to their effectiveness across a broad range of downstream applications. Though there is a big convergence in terms of architecture, most pretrained models are typically still developed for specific tasks or modalities. In this work, we propose to use language models as a general-purpose interface to various foundation models. A collection of pretrained encoders perceive diverse modalities (such as vision, and language), and they dock with a language model that plays the role of a universal task layer. We propose a semi-causal language modeling objective to jointly pretrain the interface and the modular encoders. We subsume the advantages and capabilities from both causal and non-causal modeling, thereby combining the best of two worlds. Specifically, the proposed method not only inherits the capabilities of in-context learning and open-ended generation from causal language modeling, but also is conducive to finetuning because of the bidirectional encoders. More importantly, our approach seamlessly unlocks the combinations of the above capabilities, e.g., enabling in-context learning or instruction following with finetuned encoders. Experimental results across various language-only and vision-language benchmarks show that our model outperforms or is competitive with specialized models on finetuning, zero-shot generalization, and few-shot learning.

| Comments: | 32 pages. The first three authors contribute equally         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2206.06336](https://arxiv.org/abs/2206.06336) [cs.CL]** |
|           | (or **[arXiv:2206.06336v1](https://arxiv.org/abs/2206.06336v1) [cs.CL]** for this version) |






# 2022-06-13

[Return to Index](#Index)



<h2 id="2022-06-13-1">1. A Novel Chinese Dialect TTS Frontend with Non-Autoregressive Neural Machine Translation
</h2>


Title: [A Novel Chinese Dialect TTS Frontend with Non-Autoregressive Neural Machine Translation](https://arxiv.org/abs/2206.04922)

Authors: [Wudi Bao](https://arxiv.org/search/cs?searchtype=author&query=Bao%2C+W), [Junhui Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+J), [Junjie Pan](https://arxiv.org/search/cs?searchtype=author&query=Pan%2C+J), [Xiang Yin](https://arxiv.org/search/cs?searchtype=author&query=Yin%2C+X)

> Chinese dialect text-to-speech(TTS) system usually can only be utilized by native linguists, because the written form of Chinese dialects has different characters, idioms, grammar and usage from Mandarin, and even the local speaker cannot input a correct sentence. For Mandarin text inputs, Chinese dialect TTS can only generate partly-meaningful speech with relatively poor prosody and naturalness. To lower the bar of use and make it more practical in commercial, we propose a novel Chinese dialect TTS frontend with a translation module. It helps to convert Mandarin text into idiomatic expressions with correct orthography and grammar, so that the intelligibility and naturalness of the synthesized speech can be improved. A non-autoregressive neural machine translation model with a glancing sampling strategy is proposed for the translation task. It is the first known work to incorporate translation with TTS frontend. Our experiments on Cantonese approve that the proposed frontend can help Cantonese TTS system achieve a 0.27 improvement in MOS with Mandarin inputs.

| Comments: | Submitted to INTERSPEECH 2022, 5 pages,5 figures             |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2206.04922](https://arxiv.org/abs/2206.04922) [cs.CL]** |
|           | (or **[arXiv:2206.04922v1](https://arxiv.org/abs/2206.04922v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.04922Focus to learn more |










# 2022-06-10

[Return to Index](#Index)



<h2 id="2022-06-10-1">1. Dict-NMT: Bilingual Dictionary based NMT for Extremely Low Resource Languages
</h2>

Title: [Dict-NMT: Bilingual Dictionary based NMT for Extremely Low Resource Languages](https://arxiv.org/abs/2206.04439)

Authors: [Nalin Kumar](https://arxiv.org/search/cs?searchtype=author&query=Kumar%2C+N), [Deepak Kumar](https://arxiv.org/search/cs?searchtype=author&query=Kumar%2C+D), [Subhankar Mishra](https://arxiv.org/search/cs?searchtype=author&query=Mishra%2C+S)

> Neural Machine Translation (NMT) models have been effective on large bilingual datasets. However, the existing methods and techniques show that the model's performance is highly dependent on the number of examples in training data. For many languages, having such an amount of corpora is a far-fetched dream. Taking inspiration from monolingual speakers exploring new languages using bilingual dictionaries, we investigate the applicability of bilingual dictionaries for languages with extremely low, or no bilingual corpus. In this paper, we explore methods using bilingual dictionaries with an NMT model to improve translations for extremely low resource languages. We extend this work to multilingual systems, exhibiting zero-shot properties. We present a detailed analysis of the effects of the quality of dictionaries, training dataset size, language family, etc., on the translation quality. Results on multiple low-resource test languages show a clear advantage of our bilingual dictionary-based method over the baselines.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.04439](https://arxiv.org/abs/2206.04439) [cs.CL]** |
|           | (or **[arXiv:2206.04439v1](https://arxiv.org/abs/2206.04439v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.04439Focus to learn more |





<h2 id="2022-06-10-2">2. Joint Encoder-Decoder Self-Supervised Pre-training for ASR
</h2>

Title: [Joint Encoder-Decoder Self-Supervised Pre-training for ASR](https://arxiv.org/abs/2206.04465)

Authors: [Arunkumar A](https://arxiv.org/search/cs?searchtype=author&query=A%2C+A), [Umesh S](https://arxiv.org/search/cs?searchtype=author&query=S%2C+U)

> Self-supervised learning (SSL) has shown tremendous success in various speech-related downstream tasks, including Automatic Speech Recognition (ASR). The output embeddings of the SSL model are treated as powerful short-time representations of the speech signal. However, in the ASR task, the main objective is to get the correct sequence of acoustic units, characters, or byte-pair encodings (BPEs). Usually, encoder-decoder architecture works exceptionally well for a sequence-to-sequence task like ASR. Therefore, in this paper, we propose a new paradigm that exploits the power of a decoder during self-supervised learning. We use Hidden Unit BERT (HuBERT) SSL framework to compute the conventional masked prediction loss for the encoder. In addition, we have introduced a decoder in the SSL framework and proposed a target preparation strategy for the decoder. Finally, we use a multitask SSL setup wherein we jointly optimize both the encoder and decoder losses. We hypothesize that the presence of a decoder in the SSL model helps it learn an acoustic unit-based language model, which might improve the performance of an ASR downstream task. We compare our proposed SSL model with HuBERT and show up to 25% relative improvement in performance on ASR by finetuning on various LibriSpeech subsets.

| Comments: | Submitted to Interspeech 2022                                |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2206.04465](https://arxiv.org/abs/2206.04465) [cs.CL]** |
|           | (or **[arXiv:2206.04465v1](https://arxiv.org/abs/2206.04465v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.04465Focus to learn more |





<h2 id="2022-06-10-3">3. Revisiting End-to-End Speech-to-Text Translation From Scratch
</h2>

Title: [Revisiting End-to-End Speech-to-Text Translation From Scratch](https://arxiv.org/abs/2206.04571)

Authors: [Biao Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+B), [Barry Haddow](https://arxiv.org/search/cs?searchtype=author&query=Haddow%2C+B), [Rico Sennrich](https://arxiv.org/search/cs?searchtype=author&query=Sennrich%2C+R)

> End-to-end (E2E) speech-to-text translation (ST) often depends on pretraining its encoder and/or decoder using source transcripts via speech recognition or text translation tasks, without which translation performance drops substantially. However, transcripts are not always available, and how significant such pretraining is for E2E ST has rarely been studied in the literature. In this paper, we revisit this question and explore the extent to which the quality of E2E ST trained on speech-translation pairs alone can be improved. We reexamine several techniques proven beneficial to ST previously, and offer a set of best practices that biases a Transformer-based E2E ST system toward training from scratch. Besides, we propose parameterized distance penalty to facilitate the modeling of locality in the self-attention model for speech. On four benchmarks covering 23 languages, our experiments show that, without using any transcripts or pretraining, the proposed system reaches and even outperforms previous studies adopting pretraining, although the gap remains in (extremely) low-resource settings. Finally, we discuss neural acoustic feature modeling, where a neural model is designed to extract acoustic features from raw speech signals directly, with the goal to simplify inductive biases and add freedom to the model in describing speech. For the first time, we demonstrate its feasibility and show encouraging results on ST tasks.

| Comments: | ICML                                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2206.04571](https://arxiv.org/abs/2206.04571) [cs.CL]** |
|           | (or **[arXiv:2206.04571v1](https://arxiv.org/abs/2206.04571v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.04571Focus to learn more |





# 2022-06-09

[Return to Index](#Index)



<h2 id="2022-06-09-1">1. TURJUMAN: A Public Toolkit for Neural Arabic Machine Translation
</h2>

Title: [TURJUMAN: A Public Toolkit for Neural Arabic Machine Translation](https://arxiv.org/abs/2206.03933)

Authors: [El Moatez Billah Nagoudi](https://arxiv.org/search/cs?searchtype=author&query=Nagoudi%2C+E+M+B), [AbdelRahim Elmadany](https://arxiv.org/search/cs?searchtype=author&query=Elmadany%2C+A), [Muhammad Abdul-Mageed](https://arxiv.org/search/cs?searchtype=author&query=Abdul-Mageed%2C+M)

> We present TURJUMAN, a neural toolkit for translating from 20 languages into Modern Standard Arabic (MSA). TURJUMAN exploits the recently-introduced text-to-text Transformer AraT5 model, endowing it with a powerful ability to decode into Arabic. The toolkit offers the possibility of employing a number of diverse decoding methods, making it suited for acquiring paraphrases for the MSA translations as an added value. To train TURJUMAN, we sample from publicly available parallel data employing a simple semantic similarity method to ensure data quality. This allows us to prepare and release AraOPUS-20, a new machine translation benchmark. We publicly release our translation toolkit (TURJUMAN) as well as our benchmark dataset (AraOPUS-20).

| Comments:          | All authors contributed equally                              |
| ------------------ | ------------------------------------------------------------ |
| Subjects:          | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| Cite as:           | **[arXiv:2206.03933](https://arxiv.org/abs/2206.03933) [cs.CL]** |
|                    | (or **[arXiv:2206.03933v1](https://arxiv.org/abs/2206.03933v1) [cs.CL]** for this version) |
|                    | https://doi.org/10.48550/arXiv.2206.03933Focus to learn more |
| Journal reference: | Proceedings of the 5th Workshop on Open-Source Arabic Corpora and Processing Tools (OSACT5), 2022 |





<h2 id="2022-06-09-2">2. STable: Table Generation Framework for Encoder-Decoder Models
</h2>

Title: [STable: Table Generation Framework for Encoder-Decoder Models](https://arxiv.org/abs/2206.04045)

Authors: [Michał Pietruszka](https://arxiv.org/search/cs?searchtype=author&query=Pietruszka%2C+M), [Michał Turski](https://arxiv.org/search/cs?searchtype=author&query=Turski%2C+M), [Łukasz Borchmann](https://arxiv.org/search/cs?searchtype=author&query=Borchmann%2C+Ł), [Tomasz Dwojak](https://arxiv.org/search/cs?searchtype=author&query=Dwojak%2C+T), [Gabriela Pałka](https://arxiv.org/search/cs?searchtype=author&query=Pałka%2C+G), [Karolina Szyndler](https://arxiv.org/search/cs?searchtype=author&query=Szyndler%2C+K), [Dawid Jurkiewicz](https://arxiv.org/search/cs?searchtype=author&query=Jurkiewicz%2C+D), [Łukasz Garncarek](https://arxiv.org/search/cs?searchtype=author&query=Garncarek%2C+Ł)

> The output structure of database-like tables, consisting of values structured in horizontal rows and vertical columns identifiable by name, can cover a wide range of NLP tasks. Following this constatation, we propose a framework for text-to-table neural models applicable to problems such as extraction of line items, joint entity and relation extraction, or knowledge base population. The permutation-based decoder of our proposal is a generalized sequential method that comprehends information from all cells in the table. The training maximizes the expected log-likelihood for a table's content across all random permutations of the factorization order. During the content inference, we exploit the model's ability to generate cells in any order by searching over possible orderings to maximize the model's confidence and avoid substantial error accumulation, which other sequential models are prone to. Experiments demonstrate a high practical value of the framework, which establishes state-of-the-art results on several challenging datasets, outperforming previous solutions by up to 15%.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.04045](https://arxiv.org/abs/2206.04045) [cs.CL]** |
|           | (or **[arXiv:2206.04045v1](https://arxiv.org/abs/2206.04045v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.04045Focus to learn more |





# 2022-06-08

[Return to Index](#Index)



<h2 id="2022-06-08-1">1. Tutel: Adaptive Mixture-of-Experts at Scale
</h2>

Title: [Tutel: Adaptive Mixture-of-Experts at Scale](https://arxiv.org/abs/2206.03382)

Authors: [Changho Hwang](https://arxiv.org/search/cs?searchtype=author&query=Hwang%2C+C), [Wei Cui](https://arxiv.org/search/cs?searchtype=author&query=Cui%2C+W), [Yifan Xiong](https://arxiv.org/search/cs?searchtype=author&query=Xiong%2C+Y), [Ziyue Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Z), [Ze Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Z), [Han Hu](https://arxiv.org/search/cs?searchtype=author&query=Hu%2C+H), [Zilong Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+Z), [Rafael Salas](https://arxiv.org/search/cs?searchtype=author&query=Salas%2C+R), [Jithin Jose](https://arxiv.org/search/cs?searchtype=author&query=Jose%2C+J), [Prabhat Ram](https://arxiv.org/search/cs?searchtype=author&query=Ram%2C+P), [Joe Chau](https://arxiv.org/search/cs?searchtype=author&query=Chau%2C+J), [Peng Cheng](https://arxiv.org/search/cs?searchtype=author&query=Cheng%2C+P), [Fan Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+F), [Mao Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+M), [Yongqiang Xiong](https://arxiv.org/search/cs?searchtype=author&query=Xiong%2C+Y)

> In recent years, Mixture-of-Experts (MoE) has emerged as a promising technique for deep learning that can scale the model capacity to trillion-plus parameters while reducing the computing cost via sparse computation. While MoE opens a new frontier of exceedingly large models, its implementation over thousands of GPUs has been limited due to mismatch between the dynamic nature of MoE and static parallelism/pipelining of the system. We present Tutel, a highly scalable stack design and implementation for MoE with dynamically adaptive parallelism and pipelining. Tutel delivers adaptive parallelism switching and adaptive pipelining at runtime, which achieves up to 1.74x and 2.00x single MoE layer speedup, respectively. We also propose a novel two-dimensional hierarchical algorithm for MoE communication speedup that outperforms the previous state-of-the-art up to 20.7x over 2,048 GPUs. Aggregating all techniques, Tutel finally delivers 4.96x and 5.75x speedup of a single MoE layer on 16 GPUs and 2,048 GPUs, respectively, over Fairseq: Meta's Facebook AI Research Sequence-to-Sequence Toolkit (Tutel is now partially adopted by Fairseq). Tutel source code is available in public: [this https URL](https://github.com/microsoft/tutel) . Our evaluation shows that Tutel efficiently and effectively runs a real-world MoE-based model named SwinV2-MoE, built upon Swin Transformer V2, a state-of-the-art computer vision architecture. On efficiency, Tutel accelerates SwinV2-MoE, achieving up to 1.55x and 2.11x speedup in training and inference over Fairseq, respectively. On effectiveness, the SwinV2-MoE model achieves superior accuracy in both pre-training and down-stream computer vision tasks such as COCO object detection than the counterpart dense model, indicating the readiness of Tutel for end-to-end real-world model training and inference. SwinV2-MoE is open sourced in [this https URL](https://github.com/microsoft/Swin-Transformer) .

| Subjects: | **Distributed, Parallel, and Cluster Computing (cs.DC)**; Computation and Language (cs.CL); Computer Vision and Pattern Recognition (cs.CV) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.03382](https://arxiv.org/abs/2206.03382) [cs.DC]** |
|           | (or **[arXiv:2206.03382v1](https://arxiv.org/abs/2206.03382v1) [cs.DC]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.03382Focus to learn more |





<h2 id="2022-06-08-2">2. LegoNN: Building Modular Encoder-Decoder Models
</h2>

Title: [LegoNN: Building Modular Encoder-Decoder Models](https://arxiv.org/abs/2206.03318)

Authors: [Siddharth Dalmia](https://arxiv.org/search/cs?searchtype=author&query=Dalmia%2C+S), [Dmytro Okhonko](https://arxiv.org/search/cs?searchtype=author&query=Okhonko%2C+D), [Mike Lewis](https://arxiv.org/search/cs?searchtype=author&query=Lewis%2C+M), [Sergey Edunov](https://arxiv.org/search/cs?searchtype=author&query=Edunov%2C+S), [Shinji Watanabe](https://arxiv.org/search/cs?searchtype=author&query=Watanabe%2C+S), [Florian Metze](https://arxiv.org/search/cs?searchtype=author&query=Metze%2C+F), [Luke Zettlemoyer](https://arxiv.org/search/cs?searchtype=author&query=Zettlemoyer%2C+L), [Abdelrahman Mohamed](https://arxiv.org/search/cs?searchtype=author&query=Mohamed%2C+A)

> State-of-the-art encoder-decoder models (e.g. for machine translation (MT) or speech recognition (ASR)) are constructed and trained end-to-end as an atomic unit. No component of the model can be (re-)used without the others. We describe LegoNN, a procedure for building encoder-decoder architectures with decoder modules that can be reused across various MT and ASR tasks, without the need for any fine-tuning. To achieve reusability, the interface between each encoder and decoder modules is grounded to a sequence of marginal distributions over a discrete vocabulary pre-defined by the model designer. We present two approaches for ingesting these marginals; one is differentiable, allowing the flow of gradients across the entire network, and the other is gradient-isolating. To enable portability of decoder modules between MT tasks for different source languages and across other tasks like ASR, we introduce a modality agnostic encoder which consists of a length control mechanism to dynamically adapt encoders' output lengths in order to match the expected input length range of pre-trained decoders. We present several experiments to demonstrate the effectiveness of LegoNN models: a trained language generation LegoNN decoder module from German-English (De-En) MT task can be reused with no fine-tuning for the Europarl English ASR and the Romanian-English (Ro-En) MT tasks to match or beat respective baseline models. When fine-tuned towards the target task for few thousand updates, our LegoNN models improved the Ro-En MT task by 1.5 BLEU points, and achieved 12.5% relative WER reduction for the Europarl ASR task. Furthermore, to show its extensibility, we compose a LegoNN ASR model from three modules -- each has been learned within different end-to-end trained models on three different datasets -- boosting the WER reduction to 19.5%.

| Comments: | 13 pages; Submitted to TASLP 2022                            |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2206.03318](https://arxiv.org/abs/2206.03318) [cs.CL]** |
|           | (or **[arXiv:2206.03318v1](https://arxiv.org/abs/2206.03318v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.03318Focus to learn more |





<h2 id="2022-06-08-3">3. cViL: Cross-Lingual Training of Vision-Language Models using Knowledge Distillation
</h2>

Title: [cViL: Cross-Lingual Training of Vision-Language Models using Knowledge Distillation](https://arxiv.org/abs/2206.03354)

Authors: [Kshitij Gupta](https://arxiv.org/search/cs?searchtype=author&query=Gupta%2C+K), [Devansh Gautam](https://arxiv.org/search/cs?searchtype=author&query=Gautam%2C+D), [Radhika Mamidi](https://arxiv.org/search/cs?searchtype=author&query=Mamidi%2C+R)

> Vision-and-language tasks are gaining popularity in the research community, but the focus is still mainly on English. We propose a pipeline that utilizes English-only vision-language models to train a monolingual model for a target language. We propose to extend OSCAR+, a model which leverages object tags as anchor points for learning image-text alignments, to train on visual question answering datasets in different languages. We propose a novel approach to knowledge distillation to train the model in other languages using parallel sentences. Compared to other models that use the target language in the pretraining corpora, we can leverage an existing English model to transfer the knowledge to the target language using significantly lesser resources. We also release a large-scale visual question answering dataset in Japanese and Hindi language. Though we restrict our work to visual question answering, our model can be extended to any sequence-level classification task, and it can be extended to other languages as well. This paper focuses on two languages for the visual question answering task - Japanese and Hindi. Our pipeline outperforms the current state-of-the-art models by a relative increase of 4.4% and 13.4% respectively in accuracy.

| Comments: | Accepted at ICPR 2022; 8 pages                               |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Computer Vision and Pattern Recognition (cs.CV) |
| Cite as:  | **[arXiv:2206.03354](https://arxiv.org/abs/2206.03354) [cs.CL]** |
|           | (or **[arXiv:2206.03354v1](https://arxiv.org/abs/2206.03354v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.03354Focus to learn more |







# 2022-06-07

[Return to Index](#Index)



<h2 id="2022-06-07-1">1. Rethinking the Openness of CLIP
</h2>

Title: [Rethinking the Openness of CLIP](https://arxiv.org/abs/2206.01986)

Authors: [Shuhuai Ren](https://arxiv.org/search/cs?searchtype=author&query=Ren%2C+S), [Lei Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+L), [Xuancheng Ren](https://arxiv.org/search/cs?searchtype=author&query=Ren%2C+X), [Guangxiang Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+G), [Xu Sun](https://arxiv.org/search/cs?searchtype=author&query=Sun%2C+X)

> Contrastive Language-Image Pre-training (CLIP) has demonstrated great potential in realizing open-vocabulary image classification in a matching style, because of its holistic use of natural language supervision that covers unconstrained real-world visual concepts. However, it is, in turn, also difficult to evaluate and analyze the openness of CLIP-like models, since they are in theory open to any vocabulary but the actual accuracy varies. To address the insufficiency of conventional studies on openness, we resort to an incremental view and define the extensibility, which essentially approximates the model's ability to deal with new visual concepts, by evaluating openness through vocabulary expansions. Our evaluation based on extensibility shows that CLIP-like models are hardly truly open and their performances degrade as the vocabulary expands to different degrees. Further analysis reveals that the over-estimation of openness is not because CLIP-like models fail to capture the general similarity of image and text features of novel visual concepts, but because of the confusion among competing text features, that is, they are not stable with respect to the vocabulary. In light of this, we propose to improve the openness of CLIP from the perspective of feature space by enforcing the distinguishability of text features. Our method retrieves relevant texts from the pre-training corpus to enhance prompts for inference, which boosts the extensibility and stability of CLIP even without fine-tuning.

| Comments: | 21 pages, 13 figures                                         |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2206.01986](https://arxiv.org/abs/2206.01986) [cs.CV]** |
|           | (or **[arXiv:2206.01986v1](https://arxiv.org/abs/2206.01986v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.01986Focus to learn more |





<h2 id="2022-06-07-2">2. Instance-wise Prompt Tuning for Pretrained Language Models
</h2>

Title: [Instance-wise Prompt Tuning for Pretrained Language Models](https://arxiv.org/abs/2206.01958)

Authors: [Yuezihan Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+Y), [Hao Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+H), [Junyang Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+J), [Hanyu Zhao](https://arxiv.org/search/cs?searchtype=author&query=Zhao%2C+H), [An Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+A), [Chang Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+C), [Hongxia Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+H), [Zhi Yang](https://arxiv.org/search/cs?searchtype=author&query=Yang%2C+Z), [Bin Cui](https://arxiv.org/search/cs?searchtype=author&query=Cui%2C+B)

> Prompt Learning has recently gained great popularity in bridging the gap between pretraining tasks and various downstream tasks. It freezes Pretrained Language Models (PLMs) and only tunes a few task-related parameters (prompts) for downstream tasks, greatly reducing the cost of tuning giant models. The key enabler of this is the idea of querying PLMs with task-specific knowledge implicated in prompts. This paper reveals a major limitation of existing methods that the indiscriminate prompts for all input data in a task ignore the intrinsic knowledge from input data, resulting in sub-optimal performance. We introduce Instance-wise Prompt Tuning (IPT), the first prompt learning paradigm that injects knowledge from the input data instances to the prompts, thereby providing PLMs with richer and more concrete context information. We devise a series of strategies to produce instance-wise prompts, addressing various concerns like model quality and cost-efficiency. Across multiple tasks and resource settings, IPT significantly outperforms task-based prompt learning methods, and achieves comparable performance to conventional finetuning with only 0.5% - 1.5% of tuned parameters.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.01958](https://arxiv.org/abs/2206.01958) [cs.CL]** |
|           | (or **[arXiv:2206.01958v1](https://arxiv.org/abs/2206.01958v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.01958Focus to learn more |





<h2 id="2022-06-07-3">3. Multilingual Neural Machine Translation with Deep Encoder and Multiple Shallow Decoders
</h2>

Title: [Multilingual Neural Machine Translation with Deep Encoder and Multiple Shallow Decoders](https://arxiv.org/abs/2206.02079)

Authors: [Xiang Kong](https://arxiv.org/search/cs?searchtype=author&query=Kong%2C+X), [Adithya Renduchintala](https://arxiv.org/search/cs?searchtype=author&query=Renduchintala%2C+A), [James Cross](https://arxiv.org/search/cs?searchtype=author&query=Cross%2C+J), [Yuqing Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang%2C+Y), [Jiatao Gu](https://arxiv.org/search/cs?searchtype=author&query=Gu%2C+J), [Xian Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+X)

> Recent work in multilingual translation advances translation quality surpassing bilingual baselines using deep transformer models with increased capacity. However, the extra latency and memory costs introduced by this approach may make it unacceptable for efficiency-constrained applications. It has recently been shown for bilingual translation that using a deep encoder and shallow decoder (DESD) can reduce inference latency while maintaining translation quality, so we study similar speed-accuracy trade-offs for multilingual translation. We find that for many-to-one translation we can indeed increase decoder speed without sacrificing quality using this approach, but for one-to-many translation, shallow decoders cause a clear quality drop. To ameliorate this drop, we propose a deep encoder with multiple shallow decoders (DEMSD) where each shallow decoder is responsible for a disjoint subset of target languages. Specifically, the DEMSD model with 2-layer decoders is able to obtain a 1.8x speedup on average compared to a standard transformer model with no drop in translation quality.

| Comments: | EACL 2021                                                    |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2206.02079](https://arxiv.org/abs/2206.02079) [cs.CL]** |
|           | (or **[arXiv:2206.02079v1](https://arxiv.org/abs/2206.02079v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.02079Focus to learn more |





<h2 id="2022-06-07-4">4. MorisienMT: A Dataset for Mauritian Creole Machine Translation
</h2>

Title: [MorisienMT: A Dataset for Mauritian Creole Machine Translation](https://arxiv.org/abs/2206.02421)

Authors: [Raj Dabre](https://arxiv.org/search/cs?searchtype=author&query=Dabre%2C+R), [Aneerav Sukhoo](https://arxiv.org/search/cs?searchtype=author&query=Sukhoo%2C+A)

> In this paper, we describe MorisienMT, a dataset for benchmarking machine translation quality of Mauritian Creole. Mauritian Creole (Morisien) is the lingua franca of the Republic of Mauritius and is a French-based creole language. MorisienMT consists of a parallel corpus between English and Morisien, French and Morisien and a monolingual corpus for Morisien. We first give an overview of Morisien and then describe the steps taken to create the corpora and, from it, the training and evaluation splits. Thereafter, we establish a variety of baseline models using the created parallel corpora as well as large French--English corpora for transfer learning. We release our datasets publicly for research purposes and hope that this spurs research for Morisien machine translation.

| Comments: | Work in progress! (obviously) Dataset is here: [this https URL](https://huggingface.co/datasets/prajdabre/MorisienMT) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2206.02421](https://arxiv.org/abs/2206.02421) [cs.CL]** |
|           | (or **[arXiv:2206.02421v1](https://arxiv.org/abs/2206.02421v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.02421Focus to learn more |








# 2022-06-06

[Return to Index](#Index)



<h2 id="2022-06-06-1">1. A-OKVQA: A Benchmark for Visual Question Answering using World Knowledge
</h2>

Title: [A-OKVQA: A Benchmark for Visual Question Answering using World Knowledge](https://arxiv.org/abs/2206.01718)

Authors: [Dustin Schwenk](https://arxiv.org/search/cs?searchtype=author&query=Schwenk%2C+D), [Apoorv Khandelwal](https://arxiv.org/search/cs?searchtype=author&query=Khandelwal%2C+A), [Christopher Clark](https://arxiv.org/search/cs?searchtype=author&query=Clark%2C+C), [Kenneth Marino](https://arxiv.org/search/cs?searchtype=author&query=Marino%2C+K), [Roozbeh Mottaghi](https://arxiv.org/search/cs?searchtype=author&query=Mottaghi%2C+R)

> The Visual Question Answering (VQA) task aspires to provide a meaningful testbed for the development of AI models that can jointly reason over visual and natural language inputs. Despite a proliferation of VQA datasets, this goal is hindered by a set of common limitations. These include a reliance on relatively simplistic questions that are repetitive in both concepts and linguistic structure, little world knowledge needed outside of the paired image, and limited reasoning required to arrive at the correct answer. We introduce A-OKVQA, a crowdsourced dataset composed of a diverse set of about 25K questions requiring a broad base of commonsense and world knowledge to answer. In contrast to the existing knowledge-based VQA datasets, the questions generally cannot be answered by simply querying a knowledge base, and instead require some form of commonsense reasoning about the scene depicted in the image. We demonstrate the potential of this new dataset through a detailed analysis of its contents and baseline performance measurements over a variety of state-of-the-art vision-language models. Project page: [this http URL](http://a-okvqa.allenai.org/)

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.01718](https://arxiv.org/abs/2206.01718) [cs.CV]** |
|           | (or **[arXiv:2206.01718v1](https://arxiv.org/abs/2206.01718v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.01718Focus to learn more |










# 2022-06-03

[Return to Index](#Index)



<h2 id="2022-06-03-1">1. Squeezeformer: An Efficient Transformer for Automatic Speech Recognition
</h2>

Title: [Squeezeformer: An Efficient Transformer for Automatic Speech Recognition](https://arxiv.org/abs/2206.00888)

Authors: [Sehoon Kim](https://arxiv.org/search/eess?searchtype=author&query=Kim%2C+S), [Amir Gholami](https://arxiv.org/search/eess?searchtype=author&query=Gholami%2C+A), [Albert Shaw](https://arxiv.org/search/eess?searchtype=author&query=Shaw%2C+A), [Nicholas Lee](https://arxiv.org/search/eess?searchtype=author&query=Lee%2C+N), [Karttikeya Mangalam](https://arxiv.org/search/eess?searchtype=author&query=Mangalam%2C+K), [Jitendra Malik](https://arxiv.org/search/eess?searchtype=author&query=Malik%2C+J), [Michael W. Mahoney](https://arxiv.org/search/eess?searchtype=author&query=Mahoney%2C+M+W), [Kurt Keutzer](https://arxiv.org/search/eess?searchtype=author&query=Keutzer%2C+K)

> The recently proposed Conformer model has become the de facto backbone model for various downstream speech tasks based on its hybrid attention-convolution architecture that captures both local and global features. However, through a series of systematic studies, we find that the Conformer architecture's design choices are not optimal. After reexamining the design choices for both the macro and micro-architecture of Conformer, we propose the Squeezeformer model, which consistently outperforms the state-of-the-art ASR models under the same training schemes. In particular, for the macro-architecture, Squeezeformer incorporates (i) the Temporal U-Net structure, which reduces the cost of the multi-head attention modules on long sequences, and (ii) a simpler block structure of feed-forward module, followed up by multi-head attention or convolution modules, instead of the Macaron structure proposed in Conformer. Furthermore, for the micro-architecture, Squeezeformer (i) simplifies the activations in the convolutional block, (ii) removes redundant Layer Normalization operations, and (iii) incorporates an efficient depth-wise downsampling layer to efficiently sub-sample the input signal. Squeezeformer achieves state-of-the-art results of 7.5%, 6.5%, and 6.0% word-error-rate on Librispeech test-other without external language models. This is 3.1%, 1.4%, and 0.6% better than Conformer-CTC with the same number of FLOPs. Our code is open-sourced and available online.

| Subjects: | **Audio and Speech Processing (eess.AS)**; Computation and Language (cs.CL); Sound (cs.SD) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.00888](https://arxiv.org/abs/2206.00888) [eess.AS]** |
|           | (or **[arXiv:2206.00888v1](https://arxiv.org/abs/2206.00888v1) [eess.AS]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.00888Focus to learn more |





<h2 id="2022-06-03-2">2. VL-BEiT: Generative Vision-Language Pretraining
</h2>

Title: [VL-BEiT: Generative Vision-Language Pretraining](https://arxiv.org/abs/2206.01127)

Authors: [Hangbo Bao](https://arxiv.org/search/cs?searchtype=author&query=Bao%2C+H), [Wenhui Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+W), [Li Dong](https://arxiv.org/search/cs?searchtype=author&query=Dong%2C+L), [Furu Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+F)

> We introduce a vision-language foundation model called VL-BEiT, which is a bidirectional multimodal Transformer learned by generative pretraining. Our minimalist solution conducts masked prediction on both monomodal and multimodal data with a shared Transformer. Specifically, we perform masked vision-language modeling on image-text pairs, masked language modeling on texts, and masked image modeling on images. VL-BEiT is learned from scratch with one unified pretraining task, one shared backbone, and one-stage training. Our method is conceptually simple and empirically effective. Experimental results show that VL-BEiT obtains strong results on various vision-language benchmarks, such as visual question answering, visual reasoning, and image-text retrieval. Moreover, our method learns transferable visual features, achieving competitive performance on image classification, and semantic segmentation.

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.01127](https://arxiv.org/abs/2206.01127) [cs.CV]** |
|           | (or **[arXiv:2206.01127v1](https://arxiv.org/abs/2206.01127v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.01127Focus to learn more |





<h2 id="2022-06-03-3">3. BayesFormer: Transformer with Uncertainty Estimation
</h2>

Title: [BayesFormer: Transformer with Uncertainty Estimation](https://arxiv.org/abs/2206.00826)

Authors: [Karthik Abinav Sankararaman](https://arxiv.org/search/cs?searchtype=author&query=Sankararaman%2C+K+A), [Sinong Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+S), [Han Fang](https://arxiv.org/search/cs?searchtype=author&query=Fang%2C+H)

> Transformer has become ubiquitous due to its dominant performance in various NLP and image processing tasks. However, it lacks understanding of how to generate mathematically grounded uncertainty estimates for transformer architectures. Models equipped with such uncertainty estimates can typically improve predictive performance, make networks robust, avoid over-fitting and used as acquisition function in active learning. In this paper, we introduce BayesFormer, a Transformer model with dropouts designed by Bayesian theory. We proposed a new theoretical framework to extend the approximate variational inference-based dropout to Transformer-based architectures. Through extensive experiments, we validate the proposed architecture in four paradigms and show improvements across the board: language modeling and classification, long-sequence understanding, machine translation and acquisition function for active learning.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.00826](https://arxiv.org/abs/2206.00826) [cs.CL]** |
|           | (or **[arXiv:2206.00826v1](https://arxiv.org/abs/2206.00826v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.00826Focus to learn more |





<h2 id="2022-06-03-4">4. Finding the Right Recipe for Low Resource Domain Adaptation in Neural Machine Translation
</h2>

Title: [Finding the Right Recipe for Low Resource Domain Adaptation in Neural Machine Translation](https://arxiv.org/abs/2206.01137)

Authors: [Virginia Adams](https://arxiv.org/search/cs?searchtype=author&query=Adams%2C+V), [Sandeep Subramanian](https://arxiv.org/search/cs?searchtype=author&query=Subramanian%2C+S), [Mike Chrzanowski](https://arxiv.org/search/cs?searchtype=author&query=Chrzanowski%2C+M), [Oleksii Hrinchuk](https://arxiv.org/search/cs?searchtype=author&query=Hrinchuk%2C+O), [Oleksii Kuchaiev](https://arxiv.org/search/cs?searchtype=author&query=Kuchaiev%2C+O)

> General translation models often still struggle to generate accurate translations in specialized domains. To guide machine translation practitioners and characterize the effectiveness of domain adaptation methods under different data availability scenarios, we conduct an in-depth empirical exploration of monolingual and parallel data approaches to domain adaptation of pre-trained, third-party, NMT models in settings where architecture change is impractical. We compare data centric adaptation methods in isolation and combination. We study method effectiveness in very low resource (8k parallel examples) and moderately low resource (46k parallel examples) conditions and propose an ensemble approach to alleviate reductions in original domain translation quality. Our work includes three domains: consumer electronic, clinical, and biomedical and spans four language pairs - Zh-En, Ja-En, Es-En, and Ru-En. We also make concrete recommendations for achieving high in-domain performance and release our consumer electronic and medical domain datasets for all languages and make our code publicly available.

| Subjects: | **Computation and Language (cs.CL)**; Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.01137](https://arxiv.org/abs/2206.01137) [cs.CL]** |
|           | (or **[arXiv:2206.01137v1](https://arxiv.org/abs/2206.01137v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.01137Focus to learn more |







# 2022-06-02

[Return to Index](#Index)



<h2 id="2022-06-02-1">1. VALHALLA: Visual Hallucination for Machine Translation
</h2>

Title: [VALHALLA: Visual Hallucination for Machine Translation](https://arxiv.org/abs/2206.00100)

Authors: [Yi Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+Y), [Rameswar Panda](https://arxiv.org/search/cs?searchtype=author&query=Panda%2C+R), [Yoon Kim](https://arxiv.org/search/cs?searchtype=author&query=Kim%2C+Y), [Chun-Fu](https://arxiv.org/search/cs?searchtype=author&query=Chun-Fu) (Richard)[Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen), [Rogerio Feris](https://arxiv.org/search/cs?searchtype=author&query=Feris%2C+R), [David Cox](https://arxiv.org/search/cs?searchtype=author&query=Cox%2C+D), [Nuno Vasconcelos](https://arxiv.org/search/cs?searchtype=author&query=Vasconcelos%2C+N)

> Designing better machine translation systems by considering auxiliary inputs such as images has attracted much attention in recent years. While existing methods show promising performance over the conventional text-only translation systems, they typically require paired text and image as input during inference, which limits their applicability to real-world scenarios. In this paper, we introduce a visual hallucination framework, called VALHALLA, which requires only source sentences at inference time and instead uses hallucinated visual representations for multimodal machine translation. In particular, given a source sentence an autoregressive hallucination transformer is used to predict a discrete visual representation from the input text, and the combined text and hallucinated representations are utilized to obtain the target translation. We train the hallucination transformer jointly with the translation transformer using standard backpropagation with cross-entropy losses while being guided by an additional loss that encourages consistency between predictions using either ground-truth or hallucinated visual representations. Extensive experiments on three standard translation datasets with a diverse set of language pairs demonstrate the effectiveness of our approach over both text-only baselines and state-of-the-art methods. Project page: [this http URL](http://www.svcl.ucsd.edu/projects/valhalla).

| Comments: | CVPR 2022                                                    |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2206.00100](https://arxiv.org/abs/2206.00100) [cs.CV]** |
|           | (or **[arXiv:2206.00100v1](https://arxiv.org/abs/2206.00100v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.00100Focus to learn more |





<h2 id="2022-06-02-2">2. Discovering the Hidden Vocabulary of DALLE-2
</h2>

Title: [Discovering the Hidden Vocabulary of DALLE-2](https://arxiv.org/abs/2206.00169)

Authors: [Giannis Daras](https://arxiv.org/search/cs?searchtype=author&query=Daras%2C+G), [Alexandros G. Dimakis](https://arxiv.org/search/cs?searchtype=author&query=Dimakis%2C+A+G)

> We discover that DALLE-2 seems to have a hidden vocabulary that can be used to generate images with absurd prompts. For example, it seems that \texttt{Apoploe vesrreaitais} means birds and \texttt{Contarra ccetnxniams luryca tanniounons} (sometimes) means bugs or pests. We find that these prompts are often consistent in isolation but also sometimes in combinations. We present our black-box method to discover words that seem random but have some correspondence to visual concepts. This creates important security and interpretability challenges.

| Comments: | 6 pages, 4 figures                                           |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL); Cryptography and Security (cs.CR); Computer Vision and Pattern Recognition (cs.CV) |
| Cite as:  | **[arXiv:2206.00169](https://arxiv.org/abs/2206.00169) [cs.LG]** |
|           | (or **[arXiv:2206.00169v1](https://arxiv.org/abs/2206.00169v1) [cs.LG]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.00169Focus to learn more |





<h2 id="2022-06-02-3">3. On Layer Normalizations and Residual Connections in Transformers
</h2>

Title: [On Layer Normalizations and Residual Connections in Transformers](https://arxiv.org/abs/2206.00330)

Authors: [Sho Takase](https://arxiv.org/search/cs?searchtype=author&query=Takase%2C+S), [Shun Kiyono](https://arxiv.org/search/cs?searchtype=author&query=Kiyono%2C+S), [Sosuke Kobayashi](https://arxiv.org/search/cs?searchtype=author&query=Kobayashi%2C+S), [Jun Suzuki](https://arxiv.org/search/cs?searchtype=author&query=Suzuki%2C+J)

> In the perspective of a layer normalization (LN) position, the architecture of Transformers can be categorized into two types: Post-LN and Pre-LN. Recent Transformers prefer to select Pre-LN because the training in Post-LN with deep Transformers, e.g., ten or more layers, often becomes unstable, resulting in useless models. However, in contrast, Post-LN has also consistently achieved better performance than Pre-LN in relatively shallow Transformers, e.g., six or fewer layers. This study first investigates the reason for these discrepant observations empirically and theoretically and discovers 1, the LN in Post-LN is the source of the vanishing gradient problem that mainly leads the unstable training whereas Pre-LN prevents it, and 2, Post-LN tends to preserve larger gradient norms in higher layers during the back-propagation that may lead an effective training. Exploiting the new findings, we propose a method that can equip both higher stability and effective training by a simple modification from Post-LN. We conduct experiments on a wide range of text generation tasks and demonstrate that our method outperforms Pre-LN, and stable training regardless of the shallow or deep layer settings.

| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.00330](https://arxiv.org/abs/2206.00330) [cs.LG]** |
|           | (or **[arXiv:2206.00330v1](https://arxiv.org/abs/2206.00330v1) [cs.LG]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.00330Focus to learn more |





<h2 id="2022-06-02-4">4. Optical character recognition quality affects perceived usefulness of historical newspaper clippings
</h2>

Title: [Optical character recognition quality affects perceived usefulness of historical newspaper clippings](https://arxiv.org/abs/2206.00369)

Authors: [Kimmo Kettunen](https://arxiv.org/search/cs?searchtype=author&query=Kettunen%2C+K), [Heikki Keskustalo](https://arxiv.org/search/cs?searchtype=author&query=Keskustalo%2C+H), [Sanna Kumpulainen](https://arxiv.org/search/cs?searchtype=author&query=Kumpulainen%2C+S), [Tuula Pääkkönen](https://arxiv.org/search/cs?searchtype=author&query=Pääkkönen%2C+T), [Juha Rautiainen](https://arxiv.org/search/cs?searchtype=author&query=Rautiainen%2C+J)

> Introduction. We study effect of different quality optical character recognition in interactive information retrieval with a collection of one digitized historical Finnish newspaper. Method. This study is based on the simulated interactive information retrieval work task model. Thirty-two users made searches to an article collection of Finnish newspaper Uusi Suometar 1869-1918 with ca. 1.45 million auto segmented articles. Our article search database had two versions of each article with different quality optical character recognition. Each user performed six pre-formulated and six self-formulated short queries and evaluated subjectively the top-10 results using graded relevance scale of 0-3 without knowing about the optical character recognition quality differences of the otherwise identical articles. Analysis. Analysis of the user evaluations was performed by comparing mean averages of evaluations scores in user sessions. Differences of query results were detected by analysing lengths of returned articles in pre-formulated and self-formulated queries and number of different documents retrieved overall in these two sessions. Results. The main result of the study is that improved optical character recognition quality affects perceived usefulness of historical newspaper articles positively. Conclusions. We were able to show that improvement in optical character recognition quality of documents leads to higher mean relevance evaluation scores of query results in our historical newspaper collection. To the best of our knowledge this simulated interactive user-task is the first one showing empirically that users' subjective relevance assessments are affected by a change in the quality of optically read text.

| Comments: | 21 pages, 6 figures, 2 tables, 1 appendix. arXiv admin note: substantial text overlap with [arXiv:2203.03557](https://arxiv.org/abs/2203.03557) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2206.00369](https://arxiv.org/abs/2206.00369) [cs.CL]** |
|           | (or **[arXiv:2206.00369v1](https://arxiv.org/abs/2206.00369v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.00369Focus to learn more |





<h2 id="2022-06-02-5">5. Exploring Diversity in Back Translation for Low-Resource Machine Translation
</h2>

Title: [Exploring Diversity in Back Translation for Low-Resource Machine Translation](https://arxiv.org/abs/2206.00564)

Authors: [Laurie Burchell](https://arxiv.org/search/cs?searchtype=author&query=Burchell%2C+L), [Alexandra Birch](https://arxiv.org/search/cs?searchtype=author&query=Birch%2C+A), [Kenneth Heafield](https://arxiv.org/search/cs?searchtype=author&query=Heafield%2C+K)

> Back translation is one of the most widely used methods for improving the performance of neural machine translation systems. Recent research has sought to enhance the effectiveness of this method by increasing the 'diversity' of the generated translations. We argue that the definitions and metrics used to quantify 'diversity' in previous work have been insufficient. This work puts forward a more nuanced framework for understanding diversity in training data, splitting it into lexical diversity and syntactic diversity. We present novel metrics for measuring these different aspects of diversity and carry out empirical analysis into the effect of these types of diversity on final neural machine translation model performance for low-resource English↔Turkish and mid-resource English↔Icelandic. Our findings show that generating back translation using nucleus sampling results in higher final model performance, and that this method of generation has high levels of both lexical and syntactic diversity. We also find evidence that lexical diversity is more important than syntactic for back translation performance.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.00564](https://arxiv.org/abs/2206.00564) [cs.CL]** |
|           | (or **[arXiv:2206.00564v1](https://arxiv.org/abs/2206.00564v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.00564Focus to learn more |





<h2 id="2022-06-02-6">6. Cross-View Language Modeling: Towards Unified Cross-Lingual Cross-Modal Pre-training
</h2>

Title: [Cross-View Language Modeling: Towards Unified Cross-Lingual Cross-Modal Pre-training](https://arxiv.org/abs/2206.00621)

Authors: [Yan Zeng](https://arxiv.org/search/cs?searchtype=author&query=Zeng%2C+Y), [Wangchunshu Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+W), [Ao Luo](https://arxiv.org/search/cs?searchtype=author&query=Luo%2C+A), [Xinsong Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+X)

> In this paper, we introduce Cross-View Language Modeling, a simple and effective language model pre-training framework that unifies cross-lingual cross-modal pre-training with shared architectures and objectives. Our approach is motivated by a key observation that cross-lingual and cross-modal pre-training share the same goal of aligning two different views of the same object into a common semantic space. To this end, the cross-view language modeling framework considers both multi-modal data (i.e., image-caption pairs) and multi-lingual data (i.e., parallel sentence pairs) as two different views of the same object, and trains the model to align the two views by maximizing the mutual information between them with conditional masked language modeling and contrastive learning. We pre-train CCLM, a Cross-lingual Cross-modal Language Model, with the cross-view language modeling framework. Empirical results on IGLUE, a multi-lingual multi-modal benchmark, and two multi-lingual image-text retrieval datasets show that while conceptually simpler, CCLM significantly outperforms the prior state-of-the-art with an average absolute improvement of over 10%. Notably, CCLM is the first multi-lingual multi-modal model that surpasses the translate-test performance of representative English vision-language models by zero-shot cross-lingual transfer.

| Comments: | 19 pages, 3 figures                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2206.00621](https://arxiv.org/abs/2206.00621) [cs.CL]** |
|           | (or **[arXiv:2206.00621v1](https://arxiv.org/abs/2206.00621v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.00621Focus to learn more |







# 2022-06-01

[Return to Index](#Index)



<h2 id="2022-06-01-1">1. Parameter-Efficient and Student-Friendly Knowledge Distillation
</h2>

Title: [Parameter-Efficient and Student-Friendly Knowledge Distillation](https://arxiv.org/abs/2205.15308)

Authors: [Jun Rao](https://arxiv.org/search/cs?searchtype=author&query=Rao%2C+J), [Xv Meng](https://arxiv.org/search/cs?searchtype=author&query=Meng%2C+X), [Liang Ding](https://arxiv.org/search/cs?searchtype=author&query=Ding%2C+L), [Shuhan Qi](https://arxiv.org/search/cs?searchtype=author&query=Qi%2C+S), [Dacheng Tao](https://arxiv.org/search/cs?searchtype=author&query=Tao%2C+D)

> Knowledge distillation (KD) has been extensively employed to transfer the knowledge from a large teacher model to the smaller students, where the parameters of the teacher are fixed (or partially) during training. Recent studies show that this mode may cause difficulties in knowledge transfer due to the mismatched model capacities. To alleviate the mismatch problem, teacher-student joint training methods, e.g., online distillation, have been proposed, but it always requires expensive computational cost. In this paper, we present a parameter-efficient and student-friendly knowledge distillation method, namely PESF-KD, to achieve efficient and sufficient knowledge transfer by updating relatively few partial parameters. Technically, we first mathematically formulate the mismatch as the sharpness gap between their predictive distributions, where we show such a gap can be narrowed with the appropriate smoothness of the soft label. Then, we introduce an adapter module for the teacher and only update the adapter to obtain soft labels with appropriate smoothness. Experiments on a variety of benchmarks show that PESF-KD can significantly reduce the training cost while obtaining competitive results compared to advanced online distillation methods. Code will be released upon acceptance.

| Subjects: | **Machine Learning (cs.LG)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Computer Vision and Pattern Recognition (cs.CV) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2205.15308](https://arxiv.org/abs/2205.15308) [cs.LG]** |
|           | (or **[arXiv:2205.15308v1](https://arxiv.org/abs/2205.15308v1) [cs.LG]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.15308Focus to learn more |





<h2 id="2022-06-01-2">2. ADAPT: Vision-Language Navigation with Modality-Aligned Action Prompts
</h2>

Title: [ADAPT: Vision-Language Navigation with Modality-Aligned Action Prompts](https://arxiv.org/abs/2205.15509)

Authors: [Bingqian Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+B), [Yi Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+Y), [Zicong Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+Z), [Xiwen Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+X), [Jianzhuang Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+J), [Xiaodan Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+X)

> Vision-Language Navigation (VLN) is a challenging task that requires an embodied agent to perform action-level modality alignment, i.e., make instruction-asked actions sequentially in complex visual environments. Most existing VLN agents learn the instruction-path data directly and cannot sufficiently explore action-level alignment knowledge inside the multi-modal inputs. In this paper, we propose modAlity-aligneD Action PrompTs (ADAPT), which provides the VLN agent with action prompts to enable the explicit learning of action-level modality alignment to pursue successful navigation. Specifically, an action prompt is defined as a modality-aligned pair of an image sub-prompt and a text sub-prompt, where the former is a single-view observation and the latter is a phrase like ''walk past the chair''. When starting navigation, the instruction-related action prompt set is retrieved from a pre-built action prompt base and passed through a prompt encoder to obtain the prompt feature. Then the prompt feature is concatenated with the original instruction feature and fed to a multi-layer transformer for action prediction. To collect high-quality action prompts into the prompt base, we use the Contrastive Language-Image Pretraining (CLIP) model which has powerful cross-modality alignment ability. A modality alignment loss and a sequential consistency loss are further introduced to enhance the alignment of the action prompt and enforce the agent to focus on the related prompt sequentially. Experimental results on both R2R and RxR show the superiority of ADAPT over state-of-the-art methods.

| Comments: | Accepted to CVPR 2022                                        |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2205.15509](https://arxiv.org/abs/2205.15509) [cs.CV]** |
|           | (or **[arXiv:2205.15509v1](https://arxiv.org/abs/2205.15509v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.15509Focus to learn more |





<h2 id="2022-06-01-3">3. CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers
</h2>

Title: [CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers](https://arxiv.org/abs/2205.15868)

Authors: [Wenyi Hong](https://arxiv.org/search/cs?searchtype=author&query=Hong%2C+W), [Ming Ding](https://arxiv.org/search/cs?searchtype=author&query=Ding%2C+M), [Wendi Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng%2C+W), [Xinghan Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+X), [Jie Tang](https://arxiv.org/search/cs?searchtype=author&query=Tang%2C+J)

> Large-scale pretrained transformers have created milestones in text (GPT-3) and text-to-image (DALL-E and CogView) generation. Its application to video generation is still facing many challenges: The potential huge computation cost makes the training from scratch unaffordable; The scarcity and weak relevance of text-video datasets hinder the model understanding complex movement semantics. In this work, we present 9B-parameter transformer CogVideo, trained by inheriting a pretrained text-to-image model, CogView2. We also propose multi-frame-rate hierarchical training strategy to better align text and video clips. As (probably) the first open-source large-scale pretrained text-to-video model, CogVideo outperforms all publicly available models at a large margin in machine and human evaluations.

| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2205.15868](https://arxiv.org/abs/2205.15868) [cs.CV]** |
|           | (or **[arXiv:2205.15868v1](https://arxiv.org/abs/2205.15868v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.15868Focus to learn more |





<h2 id="2022-06-01-4">4. EMS: Efficient and Effective Massively Multilingual Sentence Representation Learning
</h2>

Title: [EMS: Efficient and Effective Massively Multilingual Sentence Representation Learning](https://arxiv.org/abs/2205.15744)

Authors: [Zhuoyuan Mao](https://arxiv.org/search/cs?searchtype=author&query=Mao%2C+Z), [Chenhui Chu](https://arxiv.org/search/cs?searchtype=author&query=Chu%2C+C), [Sadao Kurohashi](https://arxiv.org/search/cs?searchtype=author&query=Kurohashi%2C+S)

> Massively multilingual sentence representation models, e.g., LASER, SBERT-distill, and LaBSE, help significantly improve cross-lingual downstream tasks. However, multiple training procedures, the use of a large amount of data, or inefficient model architectures result in heavy computation to train a new model according to our preferred languages and domains. To resolve this issue, we introduce efficient and effective massively multilingual sentence representation learning (EMS), using cross-lingual sentence reconstruction (XTR) and sentence-level contrastive learning as training objectives. Compared with related studies, the proposed model can be efficiently trained using significantly fewer parallel sentences and GPU computation resources without depending on large-scale pre-trained models. Empirical results show that the proposed model significantly yields better or comparable results with regard to bi-text mining, zero-shot cross-lingual genre classification, and sentiment classification. Ablative analyses demonstrate the effectiveness of each component of the proposed model. We release the codes for model training and the EMS pre-trained model, which supports 62 languages ([this https URL](https://github.com/Mao-KU/EMS)).

| Comments: | This work is an extension of [arXiv:2105.13856](https://arxiv.org/abs/2105.13856). This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2205.15744](https://arxiv.org/abs/2205.15744) [cs.CL]** |
|           | (or **[arXiv:2205.15744v1](https://arxiv.org/abs/2205.15744v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2205.15744Focus to learn more |






# 2022-05-31

[Return to Index](#Index)



<h2 id="2022-05-31-1">1. VLUE: A Multi-Task Benchmark for Evaluating Vision-Language Models
</h2>

Title: [VLUE: A Multi-Task Benchmark for Evaluating Vision-Language Models](https://arxiv.org/abs/2205.15237)

Authors: [Wangchunshu Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+W), [Yan Zeng](https://arxiv.org/search/cs?searchtype=author&query=Zeng%2C+Y), [Shizhe Diao](https://arxiv.org/search/cs?searchtype=author&query=Diao%2C+S), [Xinsong Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+X)

> Recent advances in vision-language pre-training (VLP) have demonstrated impressive performance in a range of vision-language (VL) tasks. However, there exist several challenges for measuring the community's progress in building general multi-modal intelligence. First, most of the downstream VL datasets are annotated using raw images that are already seen during pre-training, which may result in an overestimation of current VLP models' generalization ability. Second, recent VLP work mainly focuses on absolute performance but overlooks the efficiency-performance trade-off, which is also an important indicator for measuring progress. 
> To this end, we introduce the Vision-Language Understanding Evaluation (VLUE) benchmark, a multi-task multi-dimension benchmark for evaluating the generalization capabilities and the efficiency-performance trade-off (``Pareto SOTA'') of VLP models. We demonstrate that there is a sizable generalization gap for all VLP models when testing on out-of-distribution test sets annotated on images from a more diverse distribution that spreads across cultures. Moreover, we find that measuring the efficiency-performance trade-off of VLP models leads to complementary insights for several design choices of VLP. We release the VLUE benchmark to promote research on building vision-language models that generalize well to more diverse images and concepts unseen during pre-training, and are practical in terms of efficiency-performance trade-off.

| Comments: | ICML 2022, Benchmark website at [this https URL](https://vlue-benchmark.github.io/) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2205.15237](https://arxiv.org/abs/2205.15237) [cs.CV]** |
|           | (or **[arXiv:2205.15237v1](https://arxiv.org/abs/2205.15237v1) [cs.CV]** for this version) |



