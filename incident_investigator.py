
def investigate_drift(client, ask_modelguard_ai, psi_report, ks_report, alerts_df):

    # Convert monitoring tables into readable evidence
    psi_data = psi_report.to_dict(orient="records")
    ks_data = ks_report.to_dict(orient="records")
    alert_data = alerts_df.to_dict(orient="records")

    context = f"""
    PROJECT: ModelGuard AI

    DATA SOURCE:
    The current dataset is simulated by modifying the test dataset.
    MonthlyCharges was increased by 15.
    tenure was multiplied by 0.8.

    PSI RESULTS:
    {psi_data}

    KS RESULTS:
    {ks_data}

    ALERTS:
    {alert_data}

    IMPORTANT:
    These are simulated drift findings, not confirmed production incidents.
    Do not invent incident IDs or claim model performance degradation.
    """

    question = """
    Investigate the detected feature drift.

    Explain:
    1. What happened?
    2. Which features are affected?
    3. What do the PSI and KS results indicate?
    4. What are possible explanations based on the simulation?
    5. What should an ML engineer investigate next?
    6. What additional evidence is needed before retraining?

    Separate confirmed observations from hypotheses.
    """

    return ask_modelguard_ai(
        client=client,
        prompt=question,
        context=context
    )
