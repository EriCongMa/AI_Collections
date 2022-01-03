# MA C.'s Daily Paper Of Interest - December, 2021

# Index


- [2022-01-03](#2022-01-03)

  - [1. ERNIE-ViLG: Unified Generative Pre-training for Bidirectional Vision-Language Generation](#2022-01-03-1)
  - [2. Deconfounded Visual Grounding](#2022-01-03-2)
  - [3. Materialized Knowledge Bases from Commonsense Transformers](#2022-01-03-3)
  - [4. ViNMT: Neural Machine Translation Tookit](#2022-01-03-4)

- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MAC-Daily-POI-index.md)





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
