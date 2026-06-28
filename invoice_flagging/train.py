from model_evaluation import train_random_forest, evaluate_classifier
from data_preprocessing import load_invoice_data,apply_labels,split_data,scale_features
import joblib
# Feature columns
FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]
# Target column
TARGET = "flag_invoice"
def main():
    # Load data
    df = load_invoice_data()
    df = apply_labels(df)

    # Prepare data
    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)

    X_train_scaled, X_test_scaled = scale_features(
        X_train,
        X_test,
        '/Users/shyamchauhan/Desktop/home/codes/project/vendor_invoice_intelligence_portal/models_prediction/scaler.pkl'
    )
    # Train and evaluate model
    grid_search = train_random_forest(X_train_scaled, y_train)

    evaluate_classifier(
        grid_search.best_estimator_,
        X_test_scaled,
        y_test,
        "Random Forest Classifier"
    )
    # Save best model
    joblib.dump(
        grid_search.best_estimator_,
        "/Users/shyamchauhan/Desktop/home/codes/project/vendor_invoice_intelligence_portal/models_prediction/predict_flag_invoice.pkl"
    )
if __name__ == "__main__":
    main()
