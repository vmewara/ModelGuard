
def advise_model_performance(
    client,
    ask_modelguard_ai,
    question
):

    # Verified metrics from the original ModelGuard notebook
    model_metrics = {
        "Random Forest": {
            "accuracy": 0.7786,
            "precision": 0.5721,
            "recall": 0.6578,
            "f1_score": 0.6119,
            "roc_auc": 0.8346
        },

        "ANN": {
            "accuracy": 0.7871,
            "precision": 0.5944,
            "recall": 0.6230,
            "f1_score": 0.6084,
            "roc_auc": 0.8397
        }
    }

    context = f"""
    PROJECT: ModelGuard AI
    TASK: Customer Churn Prediction

    MODEL COMPARISON:
    {model_metrics}

    METRIC DEFINITIONS:

    Accuracy:
    Overall percentage of correct predictions.

    Precision:
    Of all customers predicted to churn,
    how many actually churned.

    Recall:
    Of all customers who actually churned,
    how many the model successfully detected.

    F1 Score:
    Balance between precision and recall.

    ROC-AUC:
    Measures how well the model separates
    churn and non-churn customers.

    IMPORTANT RULES:

    1. Use only the supplied metrics.
    2. Do not invent additional model results.
    3. Explain trade-offs between models.
    4. Do not automatically declare one model
       universally best.
    5. Consider the business importance of
       detecting customers who may churn.
    """

    return ask_modelguard_ai(
        client=client,
        prompt=question,
        context=context
    )
