# Cloud Recognition

These models ground based pictures of clouds and returns what kind of cloud it is. 

## Installing dependencies  

To install the dependencies required to run the model, make sure to have pip installed and run the following commands in the command line: 

```bash
    $ pip install --user pipenv
    $ pip install -r requirements.txt
    $ pipenv install 
```
## Starting the API

To start the API run: 

``` bash
    $ fastapi dev main.py
```
and go to: 

http://127.0.0.1:8000/docs#/default/predict_cnn_predict_cnn_post