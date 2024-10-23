from pathlib import Path
from fastapi import FastAPI
import pandas as pd
import dill
from model.modelio import ModelInput, ModelOutput

MODEL_PATH = Path(__file__).parent.parent / "models" / "best_model_script.pkl"

with open(MODEL_PATH, 'rb') as pickle_file:
    model = dill.load(pickle_file)

app = FastAPI()

@app.post("/predict", response_model=ModelOutput)
async def predict(data: ModelInput):
    model_in = ModelInput(remote_allowed=data.remote_allowed
                        ,work_type_contract=data.work_type_contract
                        ,work_type_full_time=data.work_type_full_time
                        ,work_type_part_time=data.work_type_part_time
                        ,state=data.state
                        ,company_name=data.company_name
                        ,title=data.title
                        ,description=data.description)
 
    y_pred = MODEL.predict(
        [
            [
                data.address,
                data.area,
                data.frontage,
                data.access_road,
                data.house_direction,
                data.balcony_direction,
                data.floors,
                data.bedrooms,
                data.bathrooms,
                data.legal_status,
                data.furniture_state,
            ]
        ]
    )
    return ModelOutput(price=y_pred[0])
