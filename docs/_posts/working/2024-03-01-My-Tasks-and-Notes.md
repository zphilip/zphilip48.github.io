---
layout: page
title:  "working todo"
date:   2024-03-01
categories: working
tags: AI
---
#todo 
- [x] #task compare the nougat small model and code with the nougat big model what's the different between them.  🔼 🛫 2024-01-09 : ✅ 2024-01-21
- transformer version change to  4.34.1, then it work (static saving safe is in newer version)
- [x] #task based work output of above task let's partially loading the Chinese Bart pre-trained model data 🔼 🛫 2024-01-09 ✅ 2024-02-29
- [x] prepare one small set dataset include Chinese data for training 🔼 🛫 2024-01-09 ✅ 2024-01-21
- small set dataset --- nougat-dataset-test include - ocrpadded, ocrset
- Done. the original bart model ("facebook/mbart-large-50") already has multiple language support include chinese Bart pre-trained ,
-  languages covered
Arabic (ar_AR), Czech (cs_CZ), German (de_DE), English (en_XX), Spanish (es_XX), Estonian (et_EE), Finnish (fi_FI), French (fr_XX), Gujarati (gu_IN), Hindi (hi_IN), Italian (it_IT), Japanese (ja_XX), Kazakh (kk_KZ), Korean (ko_KR), Lithuanian (lt_LT), Latvian (lv_LV), Burmese (my_MM), Nepali (ne_NP), Dutch (nl_XX), Romanian (ro_RO), Russian (ru_RU), Sinhala (si_LK), Turkish (tr_TR), Vietnamese (vi_VN), Chinese (zh_CN), Afrikaans (af_ZA), Azerbaijani (az_AZ), Bengali (bn_IN), Persian (fa_IR), Hebrew (he_IL), Croatian (hr_HR), Indonesian (id_ID), Georgian (ka_GE), Khmer (km_KH), Macedonian (mk_MK), Malayalam (ml_IN), Mongolian (mn_MN), Marathi (mr_IN), Polish (pl_PL), Pashto (ps_AF), Portuguese (pt_XX), Swedish (sv_SE), Swahili (sw_KE), Tamil (ta_IN), Telugu (te_IN), Thai (th_TH), Tagalog (tl_XX), Ukrainian (uk_UA), Urdu (ur_PK), Xhosa (xh_ZA), Galician (gl_ES), Slovene (sl_SI)

The loading such as :  ==MBartForCausalLM.from_pretrained( "facebook/mbart-large-50" ).state_dict()==

- [ ] adding authentication function to android and pc side client. let's start this after Chinese supporting in server side. 
- [ ] adding authentication and security function to server side (multiple user supporting). let's start this after Chinese supporting in server side. 
- [x] today's work 📅 2024-01-11 , ⏫  try to modify the ocr padding method to scale the ocr png to size fit  swin-transformer , then padding.  I suspect the training error is caused by too small png 🛫 padding.  current size is 672(w)x896(h)x8.. arxiv size is 816x1056x24.. something wrong. ✅ 2024-02-29
- notice that the original code has the resize function to make the small picture to fit the 886*672 size , so using the original un-padding figure (latex ocr dataset image) to feed the current training. but still have the "repetition error"
- ==(DONE) put the original un-padding figure  and the padded figure together into the training== , it seems that the training have no "repetition error" still unclear why it is so????
	- this have the repetition error... while using bigger dataset from arXiv orignal data
	- I am think the VIT how to train the all blank image... it is lead to some error , or how it is treated , get some test ???
	- how about using all back or white image as training data 
	- how the voice to text treat the white noise? how the blank image is treated 
	- in normal ViT , the fixed size image is always required...,   how about reserve the SWIN transformer architecture, change from low level resolution to high resolution 
	- #todo Need to check the result after above change....
- [x] bleu score ✅ 2024-01-12, the bleu score is noted at [[Transformer_learning#3.2 Bleu Score]]
- [x] Pytorch-view的用法 ✅ 2024-02-29
- 在pytorch中view函数的作用为重构张量的维度，相当于numpy中resize（）的功能，但是用法可能不太一样。如下例所示

```text
>>> import torch
>>> tt1=torch.tensor([-0.3623, -0.6115,  0.7283,  0.4699,  2.3261,  0.1599])
>>> result=tt1.view(3,2)
>>> result
tensor([[-0.3623, -0.6115],
        [ 0.7283,  0.4699],
        [ 2.3261,  0.1599]])
```

1. torch.view(参数a，参数b，...)

在上面例子中参数a=3和参数b=2决定了将一维的tt1重构成3x2维的张量。

2. 有的时候会出现torch.view(-1)或者torch.view(参数a，-1)这种情况。
```text
>>> import torch
>>> tt2=torch.tensor([[-0.3623, -0.6115],
...         [ 0.7283,  0.4699],
...         [ 2.3261,  0.1599]])
>>> result=tt2.view(-1)
>>> result
tensor([-0.3623, -0.6115,  0.7283,  0.4699,  2.3261,  0.1599])
```
由上面的案例可以看到，如果是torch.view(-1)，则原张量会变成一维的结构。
```text
>>> import torch
>>> tt3=torch.tensor([[-0.3623, -0.6115],
...         [ 0.7283,  0.4699],
...         [ 2.3261,  0.1599]])
>>> result=tt3.view(2,-1)
>>> result
tensor([[-0.3623, -0.6115,  0.7283],
        [ 0.4699,  2.3261,  0.1599]])
```

由上面的案例可以看到，如果是torch.view(参数a，-1)，则表示在参数b未知，参数a已知的情况下自动补齐列向量长度，在这个例子中a=2，tt3总共由6个元素，则b=6/2=3。


**Albumentations数据增强方法 **, practice in notebook "/labs/nougat_unet/predict.ipynb", refer page [Albumentations数据增强方法_shiftscalerotate-CSDN博客](https://blog.csdn.net/qq_27039891/article/details/100795846)

- [ ] Long Table splitting into small one 
- [ ] Text emotion /sentiment split, text to voice model study...etc