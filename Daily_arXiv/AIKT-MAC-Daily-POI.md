# MA C.'s Daily Paper Of Interest - February, 2022

# Index


- [2022-02-09](#2022-02-09)

  - [1. DALL-Eval: Probing the Reasoning Skills and Social Biases of Text-to-Image Generative Transformers](#2022-02-09-1)
  
- [2022-02-08](#2022-02-08)

  - [1. Machine Translation from Signed to Spoken Languages: State of the Art and Challenges](#2022-02-08-1)
  - [2. Efficient Adapter Transfer of Self-Supervised Speech Models for Automatic Speech Recognition](#2022-02-08-2)
  - [3. Red Teaming Language Models with Language Models](#2022-02-08-3)

- [2022-02-07](#2022-02-07)

  - [1. Data Scaling Laws in NMT: The Effect of Noise and Architecture](#2022-02-07-1)
  - [2. Temporal Attention for Language Models](#2022-02-07-2)
  - [3. The Ecological Footprint of Neural Machine Translation Systems](#2022-02-07-3)

- [2022-01-28](#2022-01-28)
  - [1. Tackling data scarcity in speech translation using zero-shot multilingual machine translation techniques](#2022-01-28-1)
- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MAC-Daily-POI-index.md)



# 2022-02-09

[Return to Index](#Index)



<h2 id="2022-02-08-1">1. DALL-Eval: Probing the Reasoning Skills and Social Biases of Text-to-Image Generative Transformers
</h2>

Title: [DALL-Eval: Probing the Reasoning Skills and Social Biases of Text-to-Image Generative Transformers](https://arxiv.org/abs/2202.04053)

Authors: [Jaemin Cho](https://arxiv.org/search/cs?searchtype=author&query=Cho%2C+J), [Abhay Zala](https://arxiv.org/search/cs?searchtype=author&query=Zala%2C+A), [Mohit Bansal](https://arxiv.org/search/cs?searchtype=author&query=Bansal%2C+M)

> Generating images from textual descriptions has gained a lot of attention. Recently, DALL-E, a multimodal transformer language model, and its variants have shown high-quality text-to-image generation capabilities with a simple architecture and training objective, powered by large-scale training data and computation. However, despite the interesting image generation results, there has not been a detailed analysis on how to evaluate such models. In this work, we investigate the reasoning capabilities and social biases of such text-to-image generative transformers in detail. First, we measure four visual reasoning skills: object recognition, object counting, color recognition, and spatial relation understanding. For this, we propose PaintSkills, a diagnostic dataset and evaluation toolkit that measures these four visual reasoning skills. Second, we measure the text alignment and quality of the generated images based on pretrained image captioning, image-text retrieval, and image classification models. Third, we assess social biases in the models. For this, we suggest evaluation of gender and racial biases of text-to-image generation models based on a pretrained image-text retrieval model and human evaluation. In our experiments, we show that recent text-to-image models perform better in recognizing and counting objects than recognizing colors and understanding spatial relations, while there exists a large gap between model performances and oracle accuracy on all skills. Next, we demonstrate that recent text-to-image models learn specific gender/racial biases from web image-text pairs. We also show that our automatic evaluations of visual reasoning skills and gender bias are highly correlated with human judgments. We hope our work will help guide future progress in improving text-to-image models on visual reasoning skills and social biases. Code and data at: [this https URL](https://github.com/j-min/DallEval)

| Comments: | 20 pages, 10 figures, 13 tables                              |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computer Vision and Pattern Recognition (cs.CV)**; Artificial Intelligence (cs.AI); Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2202.04053](https://arxiv.org/abs/2202.04053) [cs.CV]** |
|           | (or **[arXiv:2202.04053v1](https://arxiv.org/abs/2202.04053v1) [cs.CV]** for this version) |







# 2022-02-08

[Return to Index](#Index)



<h2 id="2022-02-08-1">1. Machine Translation from Signed to Spoken Languages: State of the Art and Challenges
</h2>

Title: [Machine Translation from Signed to Spoken Languages: State of the Art and Challenges](https://arxiv.org/abs/2202.03086)

Authors: [Mathieu De Coster](https://arxiv.org/search/cs?searchtype=author&query=De+Coster%2C+M), [Dimitar Shterionov](https://arxiv.org/search/cs?searchtype=author&query=Shterionov%2C+D), [Mieke Van Herreweghe](https://arxiv.org/search/cs?searchtype=author&query=Van+Herreweghe%2C+M), [Joni Dambre](https://arxiv.org/search/cs?searchtype=author&query=Dambre%2C+J)

> Automatic translation from signed to spoken languages is an interdisciplinary research domain, lying on the intersection of computer vision, machine translation and linguistics. Nevertheless, research in this domain is performed mostly by computer scientists in isolation. As the domain is becoming increasingly popular - the majority of scientific papers on the topic of sign language translation have been published in the past three years - we provide an overview of the state of the art as well as some required background in the different related disciplines. We give a high-level introduction to sign language linguistics and machine translation to illustrate the requirements of automatic sign language translation. We present a systematic literature review to illustrate the state of the art in the domain and then, harking back to the requirements, lay out several challenges for future research. We find that significant advances have been made on the shoulders of spoken language machine translation research. However, current approaches are often not linguistically motivated or are not adapted to the different input modality of sign languages. We explore challenges related to the representation of sign language data, the collection of datasets, the need for interdisciplinary research and requirements for moving beyond research, towards applications. Based on our findings, we advocate for interdisciplinary research and to base future research on linguistic analysis of sign languages. Furthermore, the inclusion of deaf and hearing end users of sign language translation applications in use case identification, data collection and evaluation is of the utmost importance in the creation of useful sign language translation models. We recommend iterative, human-in-the-loop, design and development of sign language translation models.

| Subjects: | **Computation and Language (cs.CL)**                         |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2202.03086](https://arxiv.org/abs/2202.03086) [cs.CL]** |
|           | (or **[arXiv:2202.03086v1](https://arxiv.org/abs/2202.03086v1) [cs.CL]** for this version) |





<h2 id="2022-02-08-2">2. Efficient Adapter Transfer of Self-Supervised Speech Models for Automatic Speech Recognition
</h2>

Title: [Efficient Adapter Transfer of Self-Supervised Speech Models for Automatic Speech Recognition](https://arxiv.org/abs/2202.03218)

Authors: [Bethan Thomas](https://arxiv.org/search/cs?searchtype=author&query=Thomas%2C+B), [Samuel Kessler](https://arxiv.org/search/cs?searchtype=author&query=Kessler%2C+S), [Salah Karout](https://arxiv.org/search/cs?searchtype=author&query=Karout%2C+S)

> Self-supervised learning (SSL) is a powerful tool that allows learning of underlying representations from unlabeled data. Transformer based models such as wav2vec 2.0 and HuBERT are leading the field in the speech domain. Generally these models are fine-tuned on a small amount of labeled data for a downstream task such as Automatic Speech Recognition (ASR). This involves re-training the majority of the model for each task. Adapters are small lightweight modules which are commonly used in Natural Language Processing (NLP) to adapt pre-trained models to new tasks. In this paper we propose applying adapters to wav2vec 2.0 to reduce the number of parameters required for downstream ASR tasks, and increase scalability of the model to multiple tasks or languages. Using adapters we can perform ASR while training fewer than 10% of parameters per task compared to full fine-tuning with little degradation of performance. Ablations show that applying adapters into just the top few layers of the pre-trained network gives similar performance to full transfer, supporting the theory that higher pre-trained layers encode more phonemic information, and further optimizing efficiency.

| Comments: | 5 Pages, 4 figures. Accepted to ICASSP 2022                  |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**; Sound (cs.SD); Audio and Speech Processing (eess.AS) |
| Cite as:  | **[arXiv:2202.03218](https://arxiv.org/abs/2202.03218) [cs.CL]** |
|           | (or **[arXiv:2202.03218v1](https://arxiv.org/abs/2202.03218v1) [cs.CL]** for this version) |





<h2 id="2022-02-08-3">3. Red Teaming Language Models with Language Models
</h2>

Title: [Red Teaming Language Models with Language Models](https://arxiv.org/abs/2202.03286)

Authors: [Ethan Perez](https://arxiv.org/search/cs?searchtype=author&query=Perez%2C+E), [Saffron Huang](https://arxiv.org/search/cs?searchtype=author&query=Huang%2C+S), [Francis Song](https://arxiv.org/search/cs?searchtype=author&query=Song%2C+F), [Trevor Cai](https://arxiv.org/search/cs?searchtype=author&query=Cai%2C+T), [Roman Ring](https://arxiv.org/search/cs?searchtype=author&query=Ring%2C+R), [John Aslanides](https://arxiv.org/search/cs?searchtype=author&query=Aslanides%2C+J), [Amelia Glaese](https://arxiv.org/search/cs?searchtype=author&query=Glaese%2C+A), [Nat McAleese](https://arxiv.org/search/cs?searchtype=author&query=McAleese%2C+N), [Geoffrey Irving](https://arxiv.org/search/cs?searchtype=author&query=Irving%2C+G)

> Language Models (LMs) often cannot be deployed because of their potential to harm users in hard-to-predict ways. Prior work identifies harmful behaviors before deployment by using human annotators to hand-write test cases. However, human annotation is expensive, limiting the number and diversity of test cases. In this work, we automatically find cases where a target LM behaves in a harmful way, by generating test cases ("red teaming") using another LM. We evaluate the target LM's replies to generated test questions using a classifier trained to detect offensive content, uncovering tens of thousands of offensive replies in a 280B parameter LM chatbot. We explore several methods, from zero-shot generation to reinforcement learning, for generating test cases with varying levels of diversity and difficulty. Furthermore, we use prompt engineering to control LM-generated test cases to uncover a variety of other harms, automatically finding groups of people that the chatbot discusses in offensive ways, personal and hospital phone numbers generated as the chatbot's own contact info, leakage of private training data in generated text, and harms that occur over the course of a conversation. Overall, LM-based red teaming is one promising tool (among many needed) for finding and fixing diverse, undesirable LM behaviors before impacting users.

| Subjects: | **Computation and Language (cs.CL)**; Artificial Intelligence (cs.AI); Cryptography and Security (cs.CR); Machine Learning (cs.LG) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2202.03286](https://arxiv.org/abs/2202.03286) [cs.CL]** |
|           | (or **[arXiv:2202.03286v1](https://arxiv.org/abs/2202.03286v1) [cs.CL]** for this version) |







# 2022-02-07

[Return to Index](#Index)



<h2 id="2022-02-07-1">1. Data Scaling Laws in NMT: The Effect of Noise and Architecture
</h2>

Title: [Data Scaling Laws in NMT: The Effect of Noise and Architecture](https://arxiv.org/abs/2202.01994)

Authors: [Yamini Bansal](https://arxiv.org/search/cs?searchtype=author&query=Bansal%2C+Y), [Behrooz Ghorbani](https://arxiv.org/search/cs?searchtype=author&query=Ghorbani%2C+B), [Ankush Garg](https://arxiv.org/search/cs?searchtype=author&query=Garg%2C+A), [Biao Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang%2C+B), [Maxim Krikun](https://arxiv.org/search/cs?searchtype=author&query=Krikun%2C+M), [Colin Cherry](https://arxiv.org/search/cs?searchtype=author&query=Cherry%2C+C), [Behnam Neyshabur](https://arxiv.org/search/cs?searchtype=author&query=Neyshabur%2C+B), [Orhan Firat](https://arxiv.org/search/cs?searchtype=author&query=Firat%2C+O)

> In this work, we study the effect of varying the architecture and training data quality on the data scaling properties of Neural Machine Translation (NMT). First, we establish that the test loss of encoder-decoder transformer models scales as a power law in the number of training samples, with a dependence on the model size. Then, we systematically vary aspects of the training setup to understand how they impact the data scaling laws. In particular, we change the following (1) Architecture and task setup: We compare to a transformer-LSTM hybrid, and a decoder-only transformer with a language modeling loss (2) Noise level in the training distribution: We experiment with filtering, and adding iid synthetic noise. In all the above cases, we find that the data scaling exponents are minimally impacted, suggesting that marginally worse architectures or training data can be compensated for by adding more data. Lastly, we find that using back-translated data instead of parallel data, can significantly degrade the scaling exponent.

| Subjects: | **Machine Learning (cs.LG)**; Computation and Language (cs.CL) |
| --------- | ------------------------------------------------------------ |
| Cite as:  | **[arXiv:2202.01994](https://arxiv.org/abs/2202.01994) [cs.LG]** |
|           | (or **[arXiv:2202.01994v1](https://arxiv.org/abs/2202.01994v1) [cs.LG]** for this version) |





<h2 id="2022-02-07-2">2. Temporal Attention for Language Models
</h2>

Title: [Temporal Attention for Language Models](https://arxiv.org/abs/2202.02093)

Authors: [Guy D. Rosin](https://arxiv.org/search/cs?searchtype=author&query=Rosin%2C+G+D), [Kira Radinsky](https://arxiv.org/search/cs?searchtype=author&query=Radinsky%2C+K)

> Pretrained language models based on the transformer architecture have shown great success in NLP. Textual training data often comes from the web and is thus tagged with time-specific information, but most language models ignore this information. They are trained on the textual data alone, limiting their ability to generalize temporally. In this work, we extend the key component of the transformer architecture, i.e., the self-attention mechanism, and propose temporal attention - a time-aware self-attention mechanism. Temporal attention can be applied to any transformer model and requires the input texts to be accompanied with their relevant time points. It allows the transformer to capture this temporal information and create time-specific contextualized word representations. We leverage these representations for the task of semantic change detection; we apply our proposed mechanism to BERT and experiment on three datasets in different languages (English, German, and Latin) that also vary in time, size, and genre. Our proposed model achieves state-of-the-art results on all the datasets.

| Comments: | 8 pages                                                      |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2202.02093](https://arxiv.org/abs/2202.02093) [cs.CL]** |
|           | (or **[arXiv:2202.02093v1](https://arxiv.org/abs/2202.02093v1) [cs.CL]** for this version) |





<h2 id="2022-02-07-3">3. The Ecological Footprint of Neural Machine Translation Systems
</h2>

Title: [The Ecological Footprint of Neural Machine Translation Systems](https://arxiv.org/abs/2202.02170)

Authors: [Dimitar Sherionov](https://arxiv.org/search/cs?searchtype=author&query=Sherionov%2C+D), [Eva Vanmassenhove](https://arxiv.org/search/cs?searchtype=author&query=Vanmassenhove%2C+E)

> Over the past decade, deep learning (DL) has led to significant advancements in various fields of artificial intelligence, including machine translation (MT). These advancements would not be possible without the ever-growing volumes of data and the hardware that allows large DL models to be trained efficiently. Due to the large amount of computing cores as well as dedicated memory, graphics processing units (GPUs) are a more effective hardware solution for training and inference with DL models than central processing units (CPUs). However, the former is very power demanding. The electrical power consumption has economical as well as ecological implications. 
> This chapter focuses on the ecological footprint of neural MT systems. It starts from the power drain during the training of and the inference with neural MT models and moves towards the environment impact, in terms of carbon dioxide emissions. Different architectures (RNN and Transformer) and different GPUs (consumer-grate NVidia 1080Ti and workstation-grade NVidia P100) are compared. Then, the overall CO2 offload is calculated for Ireland and the Netherlands. The NMT models and their ecological impact are compared to common household appliances to draw a more clear picture. 
> The last part of this chapter analyses quantization, a technique for reducing the size and complexity of models, as a way to reduce power consumption. As quantized models can run on CPUs, they present a power-efficient inference solution without depending on a GPU.

| Comments: | 25 pages, 3 figures, 10 tables                               |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Computation and Language (cs.CL)**                         |
| Cite as:  | **[arXiv:2202.02170](https://arxiv.org/abs/2202.02170) [cs.CL]** |
|           | (or **[arXiv:2202.02170v1](https://arxiv.org/abs/2202.02170v1) [cs.CL]** for this version) |



