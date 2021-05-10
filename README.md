# DST-SC
This repository is the implementation of [Dialogue State Tracking with Explicit Slot Connection Modeling](https://www.aclweb.org/anthology/2020.acl-main.5).


## Requirements
Install requirements:
```bash
pip install -r requirements.txt
```

Other preparations:
- Unzip `dataset/multiwoz.zip`
- Download [character embedding](https://www.logos.t.u-tokyo.ac.jp/~hassy/publications/arxiv2016jmt/jmt_pre-trained_embeddings.tar.gz) and unzip `charNgram.txt` to the `embedding` folder.


## Training
To train the model in the paper, run this command:
```bash
python run.py --dataset=2.0 --gpu=0,1,2 --batch_size=6 --gas=2
```


## Evaluation
To evaluate the model, specify your saved checkpoint file in `train.py` first and run:
```bash
python run.py --dataset=2.0 --is_test=True
```


## Model
![Model](model.png)


## Results
Our model achieves the following performance on [MultiWOZ 2.0](https://arxiv.org/abs/1810.00278) and [2.1](https://arxiv.org/abs/1907.01669) dataset:
| Model  | MultiWOZ 2.0 | MultiWOZ 2.1 |
| ------ | ------------ | ------------ |
| DST-SC | 52.24%       | 49.58%       |


## Citation
If you used the datasets or code, please cite our paper:
```bibtex
@inproceedings{ouyang-etal-2020-dialogue,
    title = "Dialogue State Tracking with Explicit Slot Connection Modeling",
    author = "Ouyang, Yawen  and
      Chen, Moxin  and
      Dai, Xinyu  and
      Zhao, Yinggong  and
      Huang, Shujian  and
      Chen, Jiajun",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.5",
    doi = "10.18653/v1/2020.acl-main.5",
    pages = "34--40",
    abstract = "Recent proposed approaches have made promising progress in dialogue state tracking (DST). However, in multi-domain scenarios, ellipsis and reference are frequently adopted by users to express values that have been mentioned by slots from other domains. To handle these phenomena, we propose a Dialogue State Tracking with Slot Connections (DST-SC) model to explicitly consider slot correlations across different domains. Given a target slot, the slot connecting mechanism in DST-SC can infer its source slot and copy the source slot value directly, thus significantly reducing the difficulty of learning and reasoning. Experimental results verify the benefits of explicit slot connection modeling, and our model achieves state-of-the-art performance on MultiWOZ 2.0 and MultiWOZ 2.1 datasets.",
}
```
