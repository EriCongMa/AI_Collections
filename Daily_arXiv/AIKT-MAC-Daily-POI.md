# MA C.'s Daily Paper Of Interest - November, 2022

# Index

- [2022-10-20](#2022-10-20)
  - [1. RankT5: Fine-Tuning T5 for Text Ranking with Ranking Losses](#2022-10-20-1)
  
- [Other Columns](https://github.com/EriCongMa/AI_Collections/blob/main/Daily_arXiv/AIKT-MAC-Daily-POI-index.md)



# 2022-10-20

[Return to Index](#Index)



<h2 id="2022-10-20-1">1. RankT5: Fine-Tuning T5 for Text Ranking with Ranking Losses
</h2>

Title: [RankT5: Fine-Tuning T5 for Text Ranking with Ranking Losses](https://arxiv.org/abs/2210.10634)

Authors: [Honglei Zhuang](https://arxiv.org/search/cs?searchtype=author&query=Zhuang%2C+H), [Zhen Qin](https://arxiv.org/search/cs?searchtype=author&query=Qin%2C+Z), [Rolf Jagerman](https://arxiv.org/search/cs?searchtype=author&query=Jagerman%2C+R), [Kai Hui](https://arxiv.org/search/cs?searchtype=author&query=Hui%2C+K), [Ji Ma](https://arxiv.org/search/cs?searchtype=author&query=Ma%2C+J), [Jing Lu](https://arxiv.org/search/cs?searchtype=author&query=Lu%2C+J), [Jianmo Ni](https://arxiv.org/search/cs?searchtype=author&query=Ni%2C+J), [Xuanhui Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang%2C+X), [Michael Bendersky](https://arxiv.org/search/cs?searchtype=author&query=Bendersky%2C+M)

> Recently, substantial progress has been made in text ranking based on pretrained language models such as BERT. However, there are limited studies on how to leverage more powerful sequence-to-sequence models such as T5. Existing attempts usually formulate text ranking as classification and rely on postprocessing to obtain a ranked list. In this paper, we propose RankT5 and study two T5-based ranking model structures, an encoder-decoder and an encoder-only one, so that they not only can directly output ranking scores for each query-document pair, but also can be fine-tuned with "pairwise" or "listwise" ranking losses to optimize ranking performances. Our experiments show that the proposed models with ranking losses can achieve substantial ranking performance gains on different public text ranking data sets. Moreover, when fine-tuned with listwise ranking losses, the ranking model appears to have better zero-shot ranking performance on out-of-domain data sets compared to the model fine-tuned with classification losses.

| Comments: | 13 pages                                                     |
| --------- | ------------------------------------------------------------ |
| Subjects: | **Information Retrieval (cs.IR)**; Computation and Language (cs.CL) |
| Cite as:  | **[arXiv:2210.10634](https://arxiv.org/abs/2210.10634) [cs.IR]** |
|           | (or **[arXiv:2210.10634v1](https://arxiv.org/abs/2210.10634v1) [cs.IR]** for this version) |
|           | https://doi.org/10.48550/arXiv.2210.10634Focus to learn more |

