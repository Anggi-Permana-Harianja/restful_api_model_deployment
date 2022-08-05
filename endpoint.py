from fastapi import FastAPI
from pydantic import BaseModel

import joblib

from src.feature_engineering import feature_engineering
from src.feature_engineering import feature_engineering_list

app = FastAPI()

''' load trained model '''
log_reg = joblib.load('./model/model.sav')

''' create class '''
class Iris(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
 
    ''' allow mutation '''
    def __init__(self, **data) -> None:
        super().__init__(**data)
        self.__config__.allow_mutation = True
    
    ''' prevent mutation '''
    def build(self):
        self.__config__.allow_mutation = False

class IrisList(BaseModel):
    data: list[float]
        
@app.post('/predict/')
def predict(iris: Iris):
    sepal_area, petal_area = feature_engineering(iris)
    prediction = log_reg.predict([[iris.sepal_length, iris.sepal_width, 
                                  iris.petal_length, iris.petal_width,
                                  sepal_area, petal_area]])
    ''' need to convert prediction which is in numpy array into list '''
    return {'prediction': prediction.tolist()}

@app.post('/predict_list/')
def predict_list(iris_list: IrisList):
    sepal, petal = feature_engineering_list(iris_list)
    ''' merge new feature into existing features '''
    instance = iris_list.data + [sepal] + [petal]
    
    prediction = log_reg.predict([instance])
    
    return {'prediction': prediction.tolist()}