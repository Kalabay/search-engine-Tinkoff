# поисковая система Тинькофф

Используются файлы:

dict.txt: https://drive.google.com/file/d/15GJsPHMxUoMpy2S9Dh1Qww6swvXMRu6h/view?usp=sharing

lemmatizer.txt: https://drive.google.com/file/d/1UtLQPMhXd2_txblt24RQLGvz9GT0g0qJ/view?usp=sharing

train.tsv: https://drive.google.com/file/d/1HNeRHrGV9BfdoB0WMUVM_CUsHeiUjGt2/view?usp=sharing

dict.txt, lemmatizer.txt сделаны самолично, train.tsv взят с kaggle: https://www.kaggle.com/sushanta/mercaridataset
#

Пример выполнения можно посмотреть, скачав execution_sample.mp4

#

Про сам поисковик:\
Для поиска по запросу достаточно просто написать строку и нажать Enter или кнопку Search.\
Затем он обработается и на экран выведется затраченное время и результаты, отсортированные по релевантности.\
Значение score (релевантность) будет указано рядом с каждым результатом.\
<img src="https://render.githubusercontent.com/render/math?math=score \in [0, 1] \quad score_1 > score_2 \Rightarrow relevance_1 > relevance_2">
