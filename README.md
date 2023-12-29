# a simple chatbot
this is a chatbot powered by [hugging face model](https://github.com/huggingface/transformers) that got trained on persona-chat dataset which makes model able to have and talk with a personality and you can change that personality. for image classification i used [indoor dataset](https://web.mit.edu/torralba/www/indoor.html) which is made of 67 classes of indoor pics.
and i added some rule-based commands like `tell me a joke` , `tell me a fun fact` and `tell me a quote`.

![light](https://github.com/Stmsmj/a-simple-chatbot/assets/96914984/6dcbd322-60b9-4877-8e5d-b5995717541f)
![dark](https://github.com/Stmsmj/a-simple-chatbot/assets/96914984/93c4f22c-9935-43f8-b469-15d1624169ad)

## requirements
there a file named `requirements.txt` and you need to have all those packages so you can pip install them one by one or 
```bash
cd a simple chatbot
pip install -r requirements.txt
```
if you see any error in pip installing just do it one by one

## instruction
the size of model for image classification was 3 Gigabyte so i couldn't upload it here so if you want this feature you need to train it yourself 🙂. you can fine tune it and got a better results just make a contribution 🤝.
if you dont want this feature just comment these lines `189,191,208-216,287,303` 
and if you want this feature you need to do these steps:
+ download [dataset](https://web.mit.edu/torralba/www/indoor.html)
+ extract it and copy `Image` folder to `a simple chatbot`
+ run `pic_separator.py`
+ run `indoor.ipynb` to build the model
+ run `interact.py`


