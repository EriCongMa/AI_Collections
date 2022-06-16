# MA C.'s Daily Paper Of Interest - June b., 2022

# Index

- [2022-06-16](#2022-06-16)
  - [1. Coarse-to-Fine Vision-Language Pre-training with Fusion in the Backbone](#2022-06-16-1)

  - [2. A Unified Sequence Interface for Vision Tasks](#2022-06-16-2)
  
  - [3. Prefix Language Models are Unified Modal Learners](#2022-06-16-3)
  
  - [4. Human Heuristics for AI-Generated Language Are Flawed](#2022-06-16-4)
  
  - [5. MPI: Evaluating and Inducing Personality in Pre-trained Language Models](#2022-06-16-5)
  
  - [6. Emergent Abilities of Large Language Models](#2022-06-16-6)
  
- [2022-06-15](#2022-06-15)
  - [1. LST: Ladder Side-Tuning for Parameter and Memory Efficient Transfer Learning](#2022-06-15-1)

- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MAC-Daily-POI-index.md)



# 2022-06-16

[Return to Index](#Index)



<h2 id="2022-06-16-1">1. Coarse-to-Fine Vision-Language Pre-training with Fusion in the Backbone
</h2>

Title: [Coarse-to-Fine Vision-Language Pre-training with Fusion in the Backbone](https://arxiv.org/abs/2206.07643)

Authors: [Zi-Yi Dou](https://arxiv.org/search/cs?searchtype=author&query=Dou%2C+Z), [Aishwarya Kamath](https://arxiv.org/search/cs?searchtype=author&query=Kamath%2C+A), [Zhe Gan](https://arxiv.org/search/cs?searchtype=author&query=Gan%2C+Z), [Pengchuan Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+P), [Jianfeng Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+J), [Linjie Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+L), [Zicheng Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+Z), [Ce Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu%2C+C), [Yann LeCun](https://arxiv.org/search/cs?searchtype=author&query=LeCun%2C+Y), [Nanyun Peng](https://arxiv.org/search/cs?searchtype=author&query=Peng%2C+N), [Jianfeng Gao](https://arxiv.org/search/cs?searchtype=author&query=Gao%2C+J), [Lijuan Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+L)

> Vision-language (VL) pre-training has recently received considerable attention. However, most existing end-to-end pre-training approaches either only aim to tackle VL tasks such as image-text retrieval, visual question answering (VQA) and image captioning that test high-level understanding of images, or only target region-level understanding for tasks such as phrase grounding and object detection. We present FIBER (Fusion-In-the-Backbone-based transformER), a new VL model architecture that can seamlessly handle both these types of tasks. Instead of having dedicated transformer layers for fusion after the uni-modal backbones, FIBER pushes multimodal fusion deep into the model by inserting cross-attention into the image and text backbones, bringing gains in terms of memory and performance. In addition, unlike previous work that is either only pre-trained on image-text data or on fine-grained data with box-level annotations, we present a two-stage pre-training strategy that uses both these kinds of data efficiently: (i) coarse-grained pre-training based on image-text data; followed by (ii) fine-grained pre-training based on image-text-box data. We conduct comprehensive experiments on a wide range of VL tasks, ranging from VQA, image captioning, and retrieval, to phrase grounding, referring expression comprehension, and object detection. Using deep multimodal fusion coupled with the two-stage pre-training, FIBER provides consistent performance improvements over strong baselines across all tasks, often outperforming methods using magnitudes more data. Code is available at [this https URL](https://github.com/microsoft/FIBER).

| Comments: | Project Website: [this https URL](https://ashkamath.github.io/FIBER_page) |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2206.07643](https://arxiv.org/abs/2206.07643) [cs.CV]** |
|           | (or **[arXiv:2206.07643v1](https://arxiv.org/abs/2206.07643v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.07643Focus to learn more |





<h2 id="2022-06-16-2">2. A Unified Sequence Interface for Vision Tasks
</h2>

Title: [A Unified Sequence Interface for Vision Tasks](https://arxiv.org/abs/2206.07669)

Authors: [Ting Chen](https://arxiv.org/search/cs?searchtype=author&query=Chen%2C+T), [Saurabh Saxena](https://arxiv.org/search/cs?searchtype=author&query=Saxena%2C+S), [Lala Li](https://arxiv.org/search/cs?searchtype=author&query=Li%2C+L), [Tsung-Yi Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin%2C+T), [David J. Fleet](https://arxiv.org/search/cs?searchtype=author&query=Fleet%2C+D+J), [Geoffrey Hinton](https://arxiv.org/search/cs?searchtype=author&query=Hinton%2C+G)

> While language tasks are naturally expressed in a single, unified, modeling framework, i.e., generating sequences of tokens, this has not been the case in computer vision. As a result, there is a proliferation of distinct architectures and loss functions for different vision tasks. In this work we show that a diverse set of "core" computer vision tasks can also be unified if formulated in terms of a shared pixel-to-sequence interface. We focus on four tasks, namely, object detection, instance segmentation, keypoint detection, and image captioning, all with diverse types of outputs, e.g., bounding boxes or dense masks. Despite that, by formulating the output of each task as a sequence of discrete tokens with a unified interface, we show that one can train a neural network with a single model architecture and loss function on all these tasks, with no task-specific customization. To solve a specific task, we use a short prompt as task description, and the sequence output adapts to the prompt so it can produce task-specific output. We show that such a model can achieve competitive performance compared to well-established task-specific models.

| Comments: | The first three authors contributed equally                  |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2206.07669](https://arxiv.org/abs/2206.07669) [cs.CV]** |
|           | (or **[arXiv:2206.07669v1](https://arxiv.org/abs/2206.07669v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.07669Focus to learn more |





<h2 id="2022-06-16-3">3. Prefix Language Models are Unified Modal Learners
</h2>

Title: [Prefix Language Models are Unified Modal Learners](https://arxiv.org/abs/2206.07699)

Authors: [Shizhe Diao](https://arxiv.org/search/cs?searchtype=author&query=Diao%2C+S), [Wangchunshu Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+W), [Xinsong Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+X), [Jiawei Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+J)

> With the success of vision-language pre-training, we have witnessed the state-of-the-art has been pushed on multi-modal understanding and generation. However, the current pre-training paradigm is either incapable of targeting all modalities at once (e.g., text generation and image generation), or requires multi-fold well-designed tasks which significantly limits the scalability. We demonstrate that a unified modal model could be learned with a prefix language modeling objective upon text and image sequences. Thanks to the simple but powerful pre-training paradigm, our proposed model, DaVinci, is simple to train, scalable to huge data, and adaptable to a variety of downstream tasks across modalities (language / vision / vision+language), types (understanding / generation) and settings (e.g., zero-shot, fine-tuning, linear evaluation) with a single unified architecture. DaVinci achieves the competitive performance on a wide range of 26 understanding / generation tasks, and outperforms previous unified vision-language models on most tasks, including ImageNet classification (+1.6%), VQAv2 (+1.4%), COCO caption generation (BLEU@4 +1.1%, CIDEr +1.5%) and COCO image generation (IS +0.9%, FID -1.0%), at the comparable model and data scale. Furthermore, we offer a well-defined benchmark for future research by reporting the performance on different scales of the pre-training dataset on a heterogeneous and wide distribution coverage. Our results establish new, stronger baselines for future comparisons at different data scales and shed light on the difficulties of comparing VLP models more generally.

| Comments: | 22 pages, 3 figures                                          |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Computation and Language (cs.CL); Machine Learning (cs.LG) |
| Cite as:  | **[arXiv:2206.07699](https://arxiv.org/abs/2206.07699) [cs.CV]** |
|           | (or **[arXiv:2206.07699v1](https://arxiv.org/abs/2206.07699v1) [cs.CV]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.07699Focus to learn more |





<h2 id="2022-06-16-4">4. Human Heuristics for AI-Generated Language Are Flawed
</h2>

Title: [Human Heuristics for AI-Generated Language Are Flawed](https://arxiv.org/abs/2206.07271)

Authors: [Maurice Jakesch](https://arxiv.org/search/cs?searchtype=author&query=Jakesch%2C+M), [Jeffrey Hancock](https://arxiv.org/search/cs?searchtype=author&query=Hancock%2C+J), [Mor Naaman](https://arxiv.org/search/cs?searchtype=author&query=Naaman%2C+M)

> Human communication is increasingly intermixed with language generated by AI. Across chat, email, and social media, AI systems produce smart replies, autocompletes, and translations. AI-generated language is often not identified as such but poses as human language, raising concerns about novel forms of deception and manipulation. Here, we study how humans discern whether one of the most personal and consequential forms of language - a self-presentation - was generated by AI. Across six experiments, participants (N = 4,650) tried to identify self-presentations generated by state-of-the-art language models. Across professional, hospitality, and romantic settings, we find that humans are unable to identify AI-generated self-presentations. Combining qualitative analyses with language feature engineering, we find that human judgments of AI-generated language are handicapped by intuitive but flawed heuristics such as associating first-person pronouns, authentic words, or family topics with humanity. We show that these heuristics make human judgment of generated language predictable and manipulable, allowing AI systems to produce language perceived as more human than human. We conclude by discussing solutions - such as AI accents or fair use policies - to reduce the deceptive potential of generated language, limiting the subversion of human intuition.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Computers and Society (cs.CY); Human-Computer Interaction (cs.HC) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.07271](https://arxiv.org/abs/2206.07271) [cs.CL]** |
|           | (or **[arXiv:2206.07271v1](https://arxiv.org/abs/2206.07271v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.07271Focus to learn more |





<h2 id="2022-06-16-5">5. MPI: Evaluating and Inducing Personality in Pre-trained Language Models
</h2>

Title: [MPI: Evaluating and Inducing Personality in Pre-trained Language Models](https://arxiv.org/abs/2206.07550)

Authors: [Guangyuan Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang%2C+G), [Manjie Xu](https://arxiv.org/search/cs?searchtype=author&query=Xu%2C+M), [Song-Chun Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+S), [Wenjuan Han](https://arxiv.org/search/cs?searchtype=author&query=Han%2C+W), [Chi Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+C), [Yixin Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+Y)

> Originated as a philosophical quest, personality discerns how individuals differ from each other in terms of thinking, feeling, and behaving. Towards building social machines that work with humans on a daily basis, we are motivated to ask: (1) Do existing pre-trained language models possess personality, akin to their human counterpart? If so, (2) how can we evaluate them? Further, given this evaluation framework, (3) how can we induce a certain personality in a fully controllable fashion? To tackle these three questions, we propose the Machine Personality Inventory (MPI) dataset for evaluating the machine personality; MPI follows standardized personality tests, built upon the Big Five Personality Factors (Big Five) theory and personality assessment inventories. By evaluating models with MPI, we provide the first piece of evidence showing the existence of personality in pre-trained language models. We further devise a Chain Prompting method to induce the language model with a specific personality in a controllable manner, capable of producing diversified behaviors. We hope to shed light on future studies by adopting personality as the essential psychological guidance for various downstream tasks, building more human-like and in situ dialogue agents.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.07550](https://arxiv.org/abs/2206.07550) [cs.CL]** |
|           | (or **[arXiv:2206.07550v1](https://arxiv.org/abs/2206.07550v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.07550Focus to learn more |





<h2 id="2022-06-16-6">6. Emergent Abilities of Large Language Models
</h2>

Title: [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682)

Authors: [Jason Wei](https://arxiv.org/search/cs?searchtype=author&query=Wei%2C+J), [Yi Tay](https://arxiv.org/search/cs?searchtype=author&query=Tay%2C+Y), [Rishi Bommasani](https://arxiv.org/search/cs?searchtype=author&query=Bommasani%2C+R), [Colin Raffel](https://arxiv.org/search/cs?searchtype=author&query=Raffel%2C+C), [Barret Zoph](https://arxiv.org/search/cs?searchtype=author&query=Zoph%2C+B), [Sebastian Borgeaud](https://arxiv.org/search/cs?searchtype=author&query=Borgeaud%2C+S), [Dani Yogatama](https://arxiv.org/search/cs?searchtype=author&query=Yogatama%2C+D), [Maarten Bosma](https://arxiv.org/search/cs?searchtype=author&query=Bosma%2C+M), [Denny Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+D), [Donald Metzler](https://arxiv.org/search/cs?searchtype=author&query=Metzler%2C+D), [Ed H. Chi](https://arxiv.org/search/cs?searchtype=author&query=Chi%2C+E+H), [Tatsunori Hashimoto](https://arxiv.org/search/cs?searchtype=author&query=Hashimoto%2C+T), [Oriol Vinyals](https://arxiv.org/search/cs?searchtype=author&query=Vinyals%2C+O), [Percy Liang](https://arxiv.org/search/cs?searchtype=author&query=Liang%2C+P), [Jeff Dean](https://arxiv.org/search/cs?searchtype=author&query=Dean%2C+J), [William Fedus](https://arxiv.org/search/cs?searchtype=author&query=Fedus%2C+W)

> Scaling up language models has been shown to predictably improve performance and sample efficiency on a wide range of downstream tasks. This paper instead discusses an unpredictable phenomenon that we refer to as emergent abilities of large language models. We consider an ability to be emergent if it is not present in smaller models but is present in larger models. Thus, emergent abilities cannot be predicted simply by extrapolating the performance of smaller models. The existence of such emergence implies that additional scaling could further expand the range of capabilities of language models.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2206.07682](https://arxiv.org/abs/2206.07682) [cs.CL]** |
|           | (or **[arXiv:2206.07682v1](https://arxiv.org/abs/2206.07682v1) [cs.CL]** for this version) |
|           | https://doi.org/10.48550/arXiv.2206.07682Focus to learn more |






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


