from fastapi import FastAPI
from pydantic import BaseModel
from typing import List 


app = FastAPI()


#Define Pydantic models for API 

class API_inputs(BaseModel):
    restaurant_name: str 
    date: int # example, I am not sure what to put in here 



# intialize an empty list to api 
api_list = []



#GET 
@app.get("/")
def read_root():
    return {" Hi Get test"}

#Post
@app.post("/greet")
def greet_user(name: str):
    return {'message: f"hello,{name}'}


