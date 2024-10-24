from pathlib import Path
from fastapi import FastAPI
import joblib
from model.modelio import ModelInput,ModelOutput

MODEL_PATH = Path(__file__).parent.parent / "models" / "best_model_script.pkl"

model = joblib.load(MODEL_PATH)

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
                    ,description=data.description).df_create()
 
    y_pred = model.predict(model_in)

    return ModelOutput(salary=y_pred[0])
