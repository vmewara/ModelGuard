
from genai.llm_engine import ask_modelguard_ai
from genai.incident_investigator import investigate_drift
from genai.modelguard_assistant import answer_modelguard_question
from genai.performance_advisor import advise_model_performance


class ModelGuardEngine:

    def __init__(self, client, psi_report, ks_report, alerts_df):

        self.client = client
        self.psi_report = psi_report
        self.ks_report = ks_report
        self.alerts_df = alerts_df

    def ask(self, question, feature="assistant"):

        if feature == "incident":
            return investigate_drift(
                self.client,
                ask_modelguard_ai,
                self.psi_report,
                self.ks_report,
                self.alerts_df
            )

        elif feature == "performance":
            return advise_model_performance(
                self.client,
                ask_modelguard_ai,
                question
            )

        elif feature == "assistant":
            return answer_modelguard_question(
                self.client,
                ask_modelguard_ai,
                question,
                self.psi_report,
                self.ks_report,
                self.alerts_df
            )

        else:
            return "Unknown feature. Use assistant, incident, or performance."
