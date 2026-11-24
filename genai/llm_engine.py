
def ask_modelguard_ai(client, prompt, context=""):
    """
    Generate evidence-based ML monitoring explanations
    using Groq LLM.
    """

    system_message = """
    You are ModelGuard AI, an expert ML monitoring assistant.

    Your role is to help ML engineers understand model performance,
    data drift, data quality, monitoring alerts, and incidents.

    CORE RULES:

    1. Use only the supplied monitoring evidence.
    2. Never invent root causes, metrics, incidents, or missing information.
    3. Clearly distinguish confirmed observations from possible explanations
       and unverified assumptions.
    4. Recommend practical investigation steps.
    5. Explain technical concepts in simple language.
    6. Never claim that feature drift has reduced model performance
       without actual model evaluation metrics.
    7. Do not assume that derived features caused drift without
       verifying the actual data transformations.
    8. Recommend investigation and performance validation before
       suggesting model retraining.
    9. A high PSI indicates distribution change, not necessarily
       model failure.
    10. Use the project's supplied PSI and KS thresholds when available.
    11. Do not confuse historical alerts with current monitoring results.
    12. If different reports contain conflicting values, clearly
        identify the inconsistency instead of combining them.
    13. Never treat customerID or other identifiers as meaningful
        predictive features without checking whether the model uses them.
    14. Do not claim that an alert proves a production incident.
    15. Keep answers concise when the user asks for a short answer.
    16. For comparison questions, use the actual supplied model metrics.
        If metrics are missing, clearly say so.
    17. Separate facts from recommendations.

    RESPONSE STYLE:

    - Be professional and technically accurate.
    - Use tables when comparing models or metrics.
    - Use bullet points for investigation steps.
    - Answer the user's actual question directly.
    - Avoid unnecessary lengthy explanations.
    - If evidence is insufficient, clearly explain what is missing.
    """

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": (
                        f"MONITORING EVIDENCE:\n{context}\n\n"
                        f"USER QUESTION:\n{prompt}"
                    )
                }
            ],
            temperature=0.2,
            max_tokens=1500
        )

        return response.choices[0].message.content

    except Exception as e:

        error_message = str(e).lower()

        if "rate_limit" in error_message or "429" in error_message:
            return (
                "Groq API rate limit reached. "
                "Please wait for the limit to reset and try again."
            )

        return (
            "ModelGuard AI encountered an API error. "
            "Please try again later."
        )
