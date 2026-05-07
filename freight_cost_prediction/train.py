import joblib
from pathlib import Path
from data_preprocessing import load_vendor_invoice_data,prepare_features,split_data
from model_selection import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model
)

def main():
    db_path="project/data/inventory.db"
    model_dir=Path("project/models")
    model_dir.mkdir(exist_ok=True)

    # load data
    df=load_vendor_invoice_data(db_path)

    #prepare data
    x,y=prepare_features(df)
    x_train,x_test,y_train,y_test=split_data(x,y)

    # Train model
    lr_model=train_linear_regression(x_train,y_train)
    dt_model=train_decision_tree(x_train,y_train)
    rf_model=train_random_forest(x_train,y_train)

    # evaluate models

    results=[] # creating list of results
    results.append(evaluate_model(lr_model,x_test,y_test,"Linear regression"))
    results.append(evaluate_model(dt_model,x_test,y_test,"decision tree regression"))
    results.append(evaluate_model(rf_model,x_test,y_test,"Random forest regression"))

    best_model_info=min(results,key=lambda x:x["mae"])
    best_model_name=best_model_info["model_name"]

    # select best model
    best_model={
        "Linear regression":lr_model,
        "decision tree regression":dt_model,
        "Random forest regression":rf_model
    }[best_model_name]

    # save best model
    model_path=model_dir / "predict_freight_model.pkl"
    joblib.dump(best_model,model_path)

    print(f"Best model saved : {best_model_name}")
    print(f"Model path : {model_path}")

if __name__=="__main__":
    main()
