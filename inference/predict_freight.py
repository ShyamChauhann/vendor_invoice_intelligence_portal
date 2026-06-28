# how model performs on new data
import joblib  # to save output of models as pickle file (.pkl) and to load pkl file
import pandas as pd
MODEL_PATH = "/Users/shyamchauhan/Desktop/home/codes/project/vendor_invoice_intelligence_portal/models_prediction/predict_freight_model.pkl"

def load_model(model_path: str = MODEL_PATH):
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model

def predict_freight_cost(input_data):
    model = load_model()
    # print(type(model))
    # print(type(model))
    # print(model.keys())
    # print("path : ",MODEL_PATH)
    input_df = pd.DataFrame(input_data)
    input_df['Predicted_Freight'] = model.predict(input_df).round()
    # input_df['Predicted_Freight'] = model['model_name'].predict(input_df).round()
    return input_df

if __name__ == "__main__":
    # Example inference run (local testing)
    sample_data = {
    # "Quantity": [100, 500, 200, 10],
    "Dollars": [1500, 90000, 3100, 20]
    }

    prediction = predict_freight_cost(sample_data)
    print(prediction)