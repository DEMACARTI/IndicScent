# References

The bibliographic details of entries marked ✔ were checked against the ACL Anthology, arXiv or
the dataset card while building this project (2026-10-14). Entries marked ○ are standard
references that I cite from the published record but did not re-check online in this session.
Verify ○ entries before submitting.

## Datasets

1. ✔ **MemoSen.** Eftekhar Hossain, Omar Sharif, Mohammed Moshiul Hoque. *MemoSen: A Multimodal Dataset for Sentiment Analysis of Memes.* Proceedings of the 13th Language Resources and Evaluation Conference (LREC 2022), pp. 1542–1554, Marseille. ELRA. https://aclanthology.org/2022.lrec-1.165 · Code/data link: https://github.com/eftekhar-hossain/Multimodal-Sentiment-LREC2022
2. ✔ **Memotion 3.** Shreyash Mishra, S. Suryavardan, Parth Patwa, et al. *Memotion 3: Dataset on Sentiment and Emotion Analysis of Codemixed Hindi-English Memes.* arXiv:2303.09892 (2023). Access: https://aiisc.ai/defactify2/Memotion.html (registration required).
3. ✔ **Memotion (SemEval-2020 Task 8).** Chhavi Sharma, Deepesh Bhageria, William Scott, Srinivas PYKL, Amitava Das, Tanmoy Chakraborty, Viswanath Pulabaigari, Björn Gambäck. *SemEval-2020 Task 8: Memotion Analysis – the Visuo-Lingual Metaphor!* Proceedings of SemEval-2020, pp. 759–773. https://aclanthology.org/2020.semeval-1.99
4. ✔ **CC-100** (XLM-R pre-training corpus; includes `bn_rom`, `hi_rom`, `ta_rom`, `te_rom`, `ur_rom` romanised variants). Dataset card: https://huggingface.co/datasets/statmt/cc100

## Models

5. ✔ **XLM-R.** Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, Veselin Stoyanov. *Unsupervised Cross-lingual Representation Learning at Scale.* ACL 2020, pp. 8440–8451. https://aclanthology.org/2020.acl-main.747
6. ✔ **IndicBERT / IndicNLPSuite.** Divyanshu Kakwani, Anoop Kunchukuttan, Satish Golla, Gokul N.C., Avik Bhattacharyya, Mitesh M. Khapra, Pratyush Kumar. *IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Multilingual Language Models for Indian Languages.* Findings of EMNLP 2020, pp. 4948–4961. https://aclanthology.org/2020.findings-emnlp.445
7. ○ **ResNet.** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016, pp. 770–778.
8. ○ **Transformer.** Ashish Vaswani et al. *Attention Is All You Need.* NeurIPS 2017.
9. ○ **AdamW.** Ilya Loshchilov, Frank Hutter. *Decoupled Weight Decay Regularization.* ICLR 2019.

## Explainability

10. ○ **Integrated Gradients.** Mukund Sundararajan, Ankur Taly, Qiqi Yan. *Axiomatic Attribution for Deep Networks.* ICML 2017 (PMLR 70).
11. ○ **Grad-CAM.** Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017.
12. ✔ **Erasure / occlusion for text.** Jiwei Li, Will Monroe, Dan Jurafsky. *Understanding Neural Networks through Representation Erasure.* arXiv:1612.08220 (2016).

## Evaluation methodology

13. ✔ **Significance testing in NLP.** Taylor Berg-Kirkpatrick, David Burkett, Dan Klein. *An Empirical Investigation of Statistical Significance in NLP.* EMNLP-CoNLL 2012, pp. 995–1005. https://aclanthology.org/D12-1091

## Related multimodal meme work

14. ○ Douwe Kiela et al. *The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes.* NeurIPS 2020.

## BibTeX (verified entries)

```bibtex
@inproceedings{hossain-etal-2022-memosen,
  title     = {MemoSen: A Multimodal Dataset for Sentiment Analysis of Memes},
  author    = {Hossain, Eftekhar and Sharif, Omar and Hoque, Mohammed Moshiul},
  booktitle = {Proceedings of the Thirteenth Language Resources and Evaluation Conference},
  year      = {2022}, address = {Marseille, France},
  publisher = {European Language Resources Association},
  pages     = {1542--1554}, url = {https://aclanthology.org/2022.lrec-1.165}
}
@inproceedings{conneau-etal-2020-unsupervised,
  title     = {Unsupervised Cross-lingual Representation Learning at Scale},
  author    = {Conneau, Alexis and Khandelwal, Kartikay and Goyal, Naman and Chaudhary, Vishrav and
               Wenzek, Guillaume and Guzm{\'a}n, Francisco and Grave, Edouard and Ott, Myle and
               Zettlemoyer, Luke and Stoyanov, Veselin},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  year      = {2020}, pages = {8440--8451}, doi = {10.18653/v1/2020.acl-main.747}
}
@inproceedings{kakwani-etal-2020-indicnlpsuite,
  title     = {{IndicNLPSuite}: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Multilingual
               Language Models for {I}ndian Languages},
  author    = {Kakwani, Divyanshu and Kunchukuttan, Anoop and Golla, Satish and N.C., Gokul and
               Bhattacharyya, Avik and Khapra, Mitesh M. and Kumar, Pratyush},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
  year      = {2020}, pages = {4948--4961}, doi = {10.18653/v1/2020.findings-emnlp.445}
}
@inproceedings{sharma-etal-2020-semeval,
  title     = {{S}em{E}val-2020 Task 8: Memotion Analysis- the Visuo-Lingual Metaphor!},
  author    = {Sharma, Chhavi and Bhageria, Deepesh and Scott, William and PYKL, Srinivas and Das, Amitava and
               Chakraborty, Tanmoy and Pulabaigari, Viswanath and Gamb{\"a}ck, Bj{\"o}rn},
  booktitle = {Proceedings of the Fourteenth Workshop on Semantic Evaluation},
  year      = {2020}, pages = {759--773}
}
@article{mishra2023memotion3,
  title   = {Memotion 3: Dataset on Sentiment and Emotion Analysis of Codemixed Hindi-English Memes},
  author  = {Mishra, Shreyash and Suryavardan, S. and Patwa, Parth and others},
  journal = {arXiv preprint arXiv:2303.09892}, year = {2023}
}
@article{li2016erasure,
  title   = {Understanding Neural Networks through Representation Erasure},
  author  = {Li, Jiwei and Monroe, Will and Jurafsky, Dan},
  journal = {arXiv preprint arXiv:1612.08220}, year = {2016}
}
@inproceedings{berg-kirkpatrick-etal-2012-empirical,
  title     = {An Empirical Investigation of Statistical Significance in {NLP}},
  author    = {Berg-Kirkpatrick, Taylor and Burkett, David and Klein, Dan},
  booktitle = {Proceedings of EMNLP-CoNLL 2012}, year = {2012}, pages = {995--1005}
}
```
