import joblib
import pandas as pd

# MODEL_PATH = "models_prediction/predict_flag_invoice.pkl"
MODEL_PATH = "/Users/shyamchauhan/Desktop/home/codes/project/vendor_invoice_intelligence_portal/models_prediction/predict_flag_invoice.pkl"


def load_model(model_path: str = MODEL_PATH):
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

def predict_invoice_flag(input_data):
    model = load_model()
    input_df = pd.DataFrame(input_data)
    input_df['Predicted_Flag'] = model.predict(input_df).round()
    return input_df

if __name__ == "__main__":
    sample_data = {
    "invoice_quantity": [120, 45, 300],
    "invoice_dollars": [1500, 400, 8000],
    "Freight": [60, 15, 500],
    "total_item_quantity": [118, 45, 290],
    "total_item_dollars": [1490, 390, 7800]
    }
    prediction = predict_invoice_flag(sample_data)
    print(prediction)