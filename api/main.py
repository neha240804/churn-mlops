from fastapi import FastAPI , HTTPException
from api.schemas import CustomerData
from api.predictor import predict_customer
import traceback

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0"
)


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running."}


@app.get("/health")
def health():
    return {"status": "Healthy"}


@app.post("/predict")
def predict(customer: CustomerData):
    try:
        result = predict_customer(customer.model_dump())
        return result
    except Exception as e:
        print(traceback.format_exc())   # Shows full traceback in Render logs
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)