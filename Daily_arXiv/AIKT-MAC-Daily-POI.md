# MA C.'s Daily Paper Of Interest - February, 2022

# Index


- [2022-02-07](#2022-02-07)

  - [1. Data Scaling Laws in NMT: The Effect of Noise and Architecture](#2022-02-07-1)
  - [2. Temporal Attention for Language Models](#2022-02-07-2)
  - [3. The Ecological Footprint of Neural Machine Translation Systems](#2022-02-07-3)
  
- [2022-01-28](#2022-01-28)
  - [1. Tackling data scarcity in speech translation using zero-shot multilingual machine translation techniques](#2022-01-28-1)
- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MAC-Daily-POI-index.md)





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



