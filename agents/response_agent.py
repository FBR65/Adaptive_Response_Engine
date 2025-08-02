import asyncio
import logging
from typing import Dict, Any, Optional
from a2a.client import A2AClient
from mcp import ClientSession, StdioServerParameters
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResponseEvaluation(BaseModel):
    """
    Bewertung der Antwortqualität
    """

    score: float  # Konfidenz-Score zwischen 0 und 100
    feedback: str  # Feedback zur Verbesserung
    is_sufficient: bool  # Ob die Antwort ausreichend ist


class AdaptiveResponseAgent:
    """
    Agent, der Nutzereingaben analysiert und verbessert.
    Prüft Antworten auf Genauigkeit und erzwingt ggf. neue Antworten,
    bis die Anforderungen des Nutzers zu mindestens 95% erfüllt sind.
    """

    def __init__(self, a2a_client: A2AClient):
        self.a2a_client = a2a_client
        self.mcp_client = None
        self.max_iterations = 3  # Maximale Anzahl an Iterationen
        self.quality_threshold = 95.0  # Mindestqualität in Prozent

    async def initialize_mcp_client(self):
        """
        Initialisiert den MCP-Client für die Kommunikation mit den Services
        """
        try:
            # Verbindung zu MCP-Services herstellen
            self.mcp_client = ClientSession(
                StdioServerParameters(command=["uv", "run", "mcp_main.py"])
            )
            await self.mcp_client.__aenter__()
            logger.info("MCP-Client erfolgreich initialisiert")
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung des MCP-Clients: {e}")
            raise

    async def close_mcp_client(self):
        """
        Schließt den MCP-Client
        """
        if self.mcp_client:
            await self.mcp_client.__aexit__(None, None, None)
            logger.info("MCP-Client geschlossen")

    async def generate_initial_response(self, user_input: str) -> str:
        """
        Generiert eine erste Antwort auf die Nutzereingabe

        Args:
            user_input: Die ursprüngliche Nutzereingabe

        Returns:
            Die generierte Antwort
        """
        try:
            # Verwende A2A-Client für die Antwortgenerierung
            response = await self.a2a_client.send_message(user_input)
            logger.info(f"Initiale Antwort generiert: {response}")
            return response
        except Exception as e:
            logger.error(f"Fehler bei der Generierung der initialen Antwort: {e}")
            return "Entschuldigung, ich konnte keine Antwort generieren."

    async def evaluate_response(
        self, user_input: str, response: str
    ) -> ResponseEvaluation:
        """
        Bewertet die Qualität einer Antwort

        Args:
            user_input: Die ursprüngliche Nutzereingabe
            response: Die zu bewertende Antwort

        Returns:
            ResponseEvaluation mit Score, Feedback und Suffizienz-Information
        """
        try:
            # Verwende einen LLM-basierten Ansatz zur Bewertung
            evaluation_prompt = f"""
            Bewerte die folgende Antwort im Hinblick auf Vollständigkeit und Genauigkeit im Vergleich zur ursprünglichen Nutzeranfrage.
            Gib eine Punktzahl von 1 bis 100 an und erkläre, warum du diese Bewertung gibst.
            Wenn die Antwort unvollständig ist, gib an, was fehlt.
            Wenn die Antwort falsche Informationen enthält, gib an, was falsch ist.
            
            Ursprüngliche Anfrage: {user_input}
            
            Antwort: {response}
            
            Antwort mit folgendem JSON-Format:
            {{
                "score": <Punktzahl von 1-100>,
                "feedback": "<Feedback zur Verbesserung>",
                "is_sufficient": <true/false>
            }}
            """

            # Verwende A2A-Client für die Bewertung
            evaluation_response = await self.a2a_client.send_message(evaluation_prompt)

            # Parse die JSON-Antwort
            import json

            evaluation_data = json.loads(evaluation_response)

            return ResponseEvaluation(
                score=evaluation_data["score"],
                feedback=evaluation_data["feedback"],
                is_sufficient=evaluation_data["is_sufficient"]
                and evaluation_data["score"] >= self.quality_threshold,
            )
        except Exception as e:
            logger.error(f"Fehler bei der Bewertung der Antwort: {e}")
            # Standardbewertung bei Fehler
            return ResponseEvaluation(
                score=50.0, feedback="Fehler bei der Bewertung", is_sufficient=False
            )

    async def improve_response(
        self, user_input: str, previous_response: str, feedback: str
    ) -> str:
        """
        Verbessert eine Antwort basierend auf Feedback

        Args:
            user_input: Die ursprüngliche Nutzereingabe
            previous_response: Die vorherige Antwort
            feedback: Das Feedback zur vorherigen Antwort

        Returns:
            Die verbesserte Antwort
        """
        try:
            improvement_prompt = f"""
            Generiere eine neue Antwort auf die folgende Anfrage.
            Die vorherige Antwort war unzureichend aus folgenden Gründen: {feedback}
            
            Ursprüngliche Anfrage: {user_input}
            
            Vorherige Antwort: {previous_response}
            
            Bitte gib eine verbesserte Antwort, die das Feedback berücksichtigt.
            """

            # Verwende A2A-Client für die verbesserte Antwort
            improved_response = await self.a2a_client.send_message(improvement_prompt)
            logger.info(f"Antwort verbessert: {improved_response}")
            return improved_response
        except Exception as e:
            logger.error(f"Fehler bei der Verbesserung der Antwort: {e}")
            return "Entschuldigung, ich konnte die Antwort nicht verbessern."

    async def process_user_input(self, user_input: str) -> str:
        """
        Verarbeitet die Nutzereingabe und generiert eine qualitativ hochwertige Antwort

        Args:
            user_input: Die Nutzereingabe

        Returns:
            Die finale Antwort
        """
        try:
            # Initialisiere MCP-Client
            await self.initialize_mcp_client()

            # Generiere initiale Antwort
            current_response = await self.generate_initial_response(user_input)
            logger.info(f"Initiale Antwort: {current_response}")

            # Iteriere, bis die Antwortqualität ausreichend ist oder das Iterationslimit erreicht ist
            iteration = 0
            while iteration < self.max_iterations:
                # Bewerte die aktuelle Antwort
                evaluation = await self.evaluate_response(user_input, current_response)
                logger.info(
                    f"Iteration {iteration + 1} - Bewertung: {evaluation.score}, Feedback: {evaluation.feedback}"
                )

                # Prüfe, ob die Antwort ausreichend ist
                if evaluation.is_sufficient:
                    logger.info(f"Antwort ist ausreichend mit Score {evaluation.score}")
                    break

                # Verbessere die Antwort
                current_response = await self.improve_response(
                    user_input, current_response, evaluation.feedback
                )
                iteration += 1

            if iteration == self.max_iterations:
                logger.warning("Maximale Anzahl an Iterationen erreicht")

            return current_response

        finally:
            # Schließe MCP-Client
            await self.close_mcp_client()
