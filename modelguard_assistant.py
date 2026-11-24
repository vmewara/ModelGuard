
def answer_modelguard_question(
    client,
    ask_modelguard_ai,
    question,
    psi_report,
    ks_report,
    alerts_df
):
    """
    Answer questions using ModelGuard monitoring evidence.
    """

    context = f"""
    PROJECT: ModelGuard AI

    DATA SOURCE:
    This is a simulated drift experiment.
    MonthlyCharges was increased by 15.
    tenure was multiplied by 0.8.
    These are not confirmed production incidents.

    PSI RESULTS:
    {psi_report.to_dict(orient="records")}

    KS RESULTS:
    {ks_report.to_dict(orient="records")}

    ALERTS:
    {alerts_df.to_dict(orient="records")}

    IMPORTANT INTERPRETATION:
    PSI >= 0.25 indicates significant drift under the
    project's current guideline.

    KS p-values displayed as 0.0 may be rounded or
    numerically underflowed. Do not interpret them as
    literal zero probabilities.

    Answer only using the supplied evidence.
    If information is missing, say so.
    Do not invent model metrics, incident IDs,
    production events, or root causes.
    """

    return ask_modelguard_ai(
        client=client,
        prompt=question,
        context=context
    )
