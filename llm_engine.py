
def ask_modelguard_ai(client, prompt, context=""):
    """
    Generate evidence-based ML monitoring explanations.
    """

    system_message = """
    You are ModelGuard AI, an expert ML monitoring assistant.

    Rules:
    1. Use only the supplied monitoring evidence.
    2. Never invent root causes or missing metrics.
    3. Clearly distinguish observations from possible explanations.
    4. Recommend practical investigation steps.
    5. Explain technical concepts in simple language.
    """

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Evidence:\n{context}\n\nQuestion:\n{prompt}"}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content
