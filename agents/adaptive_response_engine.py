"""
Adaptive Response Engine - Hauptmodul für das Agent System
Integriert alle Agents und MCP-Tools zu einem kohärenten System
"""

import logging
from typing import Dict, Any, Optional
import time

from .query_analysis_agent import QueryAnalysisAgent
from .response_generation_agent import ResponseGenerationAgent
from .quality_review_agent import QualityReviewAgent
from .iteration_controller import IterationController
from .a2a_coordinator import A2ACoordinator, AgentRole, MessageType

logger = logging.getLogger(__name__)


class AdaptiveResponseEngine:
    """
    Hauptklasse des Adaptive Response Engine Systems.
    Koordiniert alle Agents und stellt die primäre Schnittstelle dar.
    """

    def __init__(
        self,
        mcp_tools: Optional[Dict] = None,
        max_iterations: int = 3,
        quality_threshold: float = 95.0,
        openai_api_key: Optional[str] = None,
    ):
        """
        Initialisiert das Adaptive Response Engine System.

        Args:
            mcp_tools: Verfügbare MCP-Tools
            max_iterations: Maximale Anzahl von Iterationen
            quality_threshold: Mindestqualität für Akzeptanz
            openai_api_key: OpenAI API-Schlüssel
        """
        self.mcp_tools = mcp_tools or {}
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.openai_api_key = openai_api_key

        # Agent System Komponenten
        self.query_agent = None
        self.response_agent = None
        self.quality_agent = None
        self.iteration_controller = None
        self.a2a_coordinator = None

        # System Status
        self.initialized = False
        self.running = False
        self.performance_history = []

        logger.info("Adaptive Response Engine initialisiert")

    async def initialize(self):
        """Initialisiert alle System-Komponenten."""
        try:
            logger.info("Starte Initialisierung des Agent Systems...")

            # Initialisiere A2A Coordinator
            self.a2a_coordinator = A2ACoordinator()
            await self.a2a_coordinator.start()

            # Initialisiere Core Agents
            self.query_agent = QueryAnalysisAgent()

            self.response_agent = ResponseGenerationAgent()

            self.quality_agent = QualityReviewAgent()

            # Initialisiere Iteration Controller
            self.iteration_controller = IterationController(
                query_agent=self.query_agent,
                response_agent=self.response_agent,
                quality_agent=self.quality_agent,
                max_iterations=self.max_iterations,
                quality_threshold=self.quality_threshold,
            )

            # Registriere Agents im A2A System
            await self._register_agents()

            self.initialized = True
            self.running = True

            logger.info("Agent System erfolgreich initialisiert")

        except Exception as e:
            logger.error(f"Fehler bei Initialisierung: {e}")
            raise

    async def _register_agents(self):
        """Registriert alle Agents im A2A Coordinator."""
        try:
            # Query Analysis Agent
            self.a2a_coordinator.register_agent(
                agent_id="query_analyzer",
                agent_instance=self.query_agent,
                role=AgentRole.ANALYZER,
                capabilities=["query_analysis", "intent_detection", "query_refinement"],
                message_handler=self._handle_query_agent_message,
            )

            # Response Generation Agent
            self.a2a_coordinator.register_agent(
                agent_id="response_generator",
                agent_instance=self.response_agent,
                role=AgentRole.GENERATOR,
                capabilities=[
                    "response_generation",
                    "information_gathering",
                    "synthesis",
                ],
                message_handler=self._handle_response_agent_message,
            )

            # Quality Review Agent
            self.a2a_coordinator.register_agent(
                agent_id="quality_reviewer",
                agent_instance=self.quality_agent,
                role=AgentRole.REVIEWER,
                capabilities=[
                    "quality_evaluation",
                    "feedback_generation",
                    "assessment",
                ],
                message_handler=self._handle_quality_agent_message,
            )

            # System Coordinator
            self.a2a_coordinator.register_agent(
                agent_id="system_coordinator",
                agent_instance=self,
                role=AgentRole.COORDINATOR,
                capabilities=[
                    "system_coordination",
                    "task_orchestration",
                    "monitoring",
                ],
                message_handler=self._handle_coordinator_message,
            )

            logger.info("Alle Agents erfolgreich im A2A System registriert")

        except Exception as e:
            logger.error(f"Fehler bei Agent-Registrierung: {e}")
            raise

    async def process_query(
        self, query: str, context: Optional[Dict[str, Any]] = None, use_a2a: bool = True
    ) -> Dict[str, Any]:
        """
        Haupteingangspunkt für Query-Verarbeitung.

        Args:
            query: Die Nutzereingabe
            context: Zusätzlicher Kontext
            use_a2a: Ob A2A-Koordination verwendet werden soll

        Returns:
            Vollständiges Verarbeitungsergebnis
        """
        if not self.initialized:
            raise RuntimeError("System nicht initialisiert. Rufe initialize() auf.")

        start_time = time.time()

        try:
            logger.info(f"Starte Query-Verarbeitung: {query[:100]}...")

            if use_a2a:
                # Verwende A2A-koordinierte Verarbeitung
                result = await self._process_query_with_a2a(query, context)
            else:
                # Verwende direkte Iteration Controller Verarbeitung
                result = await self.iteration_controller.process_query(
                    query, context, self.mcp_tools
                )

            # Performance Tracking
            processing_time = time.time() - start_time
            self._track_performance(result, processing_time)

            logger.info(
                f"Query-Verarbeitung abgeschlossen. Zeit: {processing_time:.2f}s"
            )
            return result

        except Exception as e:
            logger.error(f"Fehler bei Query-Verarbeitung: {e}")
            return await self._create_error_response(
                query, str(e), time.time() - start_time
            )

    async def _process_query_with_a2a(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Verarbeitet Query mit A2A-Koordination.

        Args:
            query: Die Nutzereingabe
            context: Zusätzlicher Kontext

        Returns:
            Koordiniertes Verarbeitungsergebnis
        """
        try:
            # Koordiniere Task mit allen relevanten Agents
            coordination_result = await self.a2a_coordinator.coordinate_task(
                coordinator_id="system_coordinator",
                task_description=f"Process user query: {query[:100]}",
                required_capabilities=[
                    "query_analysis",
                    "response_generation",
                    "quality_evaluation",
                ],
                task_data={
                    "query": query,
                    "context": context,
                    "mcp_tools": self.mcp_tools,
                    "quality_threshold": self.quality_threshold,
                    "max_iterations": self.max_iterations,
                },
            )

            if coordination_result["success"]:
                # Extrahiere finales Ergebnis aus koordinierten Resultaten
                return await self._extract_final_result(coordination_result)
            else:
                # Fallback zu direkter Verarbeitung
                logger.warning(
                    "A2A-Koordination fehlgeschlagen, verwende direkten Modus"
                )
                return await self.iteration_controller.process_query(
                    query, context, self.mcp_tools
                )

        except Exception as e:
            logger.error(f"Fehler bei A2A-Verarbeitung: {e}")
            # Fallback zu direkter Verarbeitung
            return await self.iteration_controller.process_query(
                query, context, self.mcp_tools
            )

    async def _extract_final_result(
        self, coordination_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extrahiert das finale Ergebnis aus koordinierten Agent-Resultaten."""
        try:
            results = coordination_result.get("results", {})

            # Sammle Ergebnisse von allen Agents
            query_analysis = None
            response_data = None
            quality_evaluation = None

            for agent_id, result in results.items():
                if agent_id == "query_analyzer" and "analysis" in result:
                    query_analysis = result["analysis"]
                elif agent_id == "response_generator" and "response" in result:
                    response_data = result
                elif agent_id == "quality_reviewer" and "evaluation" in result:
                    quality_evaluation = result["evaluation"]

            # Konstruiere finales Ergebnis
            if response_data and quality_evaluation:
                final_result = {
                    "response": response_data["response"],
                    "quality_score": quality_evaluation.get("overall_score", 0.0),
                    "iterations": 1,  # A2A = eine koordinierte Iteration
                    "total_processing_time": coordination_result.get(
                        "processing_time", 0.0
                    ),
                    "query_analysis": query_analysis,
                    "coordination_result": coordination_result,
                    "metadata": {
                        "coordination_mode": True,
                        "agents_participated": len(results),
                        "success": quality_evaluation.get("overall_score", 0.0)
                        >= self.quality_threshold,
                    },
                }
                return final_result
            else:
                raise ValueError("Unvollständige Koordinations-Ergebnisse")

        except Exception as e:
            logger.error(f"Fehler bei Ergebnis-Extraktion: {e}")
            raise

    # A2A Message Handlers
    async def _handle_query_agent_message(self, message) -> Optional[Dict[str, Any]]:
        """Behandelt Nachrichten für den Query Analysis Agent."""
        try:
            if message.message_type == MessageType.REQUEST:
                content = message.content
                task_data = content.get("task_data", {})

                query = task_data.get("query", "")
                context = task_data.get("context", {})

                # Führe Query-Analyse durch
                analysis = await self.query_agent.analyze_query(query, context)

                return {
                    "agent_id": "query_analyzer",
                    "analysis": analysis,
                    "status": "completed",
                }

        except Exception as e:
            logger.error(f"Fehler bei Query Agent Message: {e}")
            return {"error": str(e)}

    async def _handle_response_agent_message(self, message) -> Optional[Dict[str, Any]]:
        """Behandelt Nachrichten für den Response Generation Agent."""
        try:
            if message.message_type == MessageType.REQUEST:
                content = message.content
                task_data = content.get("task_data", {})

                query = task_data.get("query", "")
                context = task_data.get("context", {})
                mcp_tools = task_data.get("mcp_tools", {})

                # Hole Query-Analyse von anderem Agent falls verfügbar
                analysis = await self._request_query_analysis(query, context)

                # Generiere Antwort
                result = await self.response_agent.generate_response(
                    query, analysis, mcp_tools, context
                )

                return {
                    "agent_id": "response_generator",
                    "response": result["response"],
                    "sources_used": result.get("sources_used", []),
                    "generation_metadata": result,
                    "status": "completed",
                }

        except Exception as e:
            logger.error(f"Fehler bei Response Agent Message: {e}")
            return {"error": str(e)}

    async def _handle_quality_agent_message(self, message) -> Optional[Dict[str, Any]]:
        """Behandelt Nachrichten für den Quality Review Agent."""
        try:
            if message.message_type == MessageType.REQUEST:
                content = message.content
                task_data = content.get("task_data", {})

                query = task_data.get("query", "")

                # Hole Response von anderem Agent
                response_data = await self._request_response_generation(
                    query, task_data
                )
                if not response_data or "error" in response_data:
                    return {"error": "Konnte keine Response für Evaluation erhalten"}

                response = response_data.get("response", "")
                generation_metadata = response_data.get("generation_metadata", {})

                # Hole Query-Analyse
                analysis = await self._request_query_analysis(
                    query, task_data.get("context", {})
                )

                # Evaluiere Qualität
                evaluation = await self.quality_agent.evaluate_response(
                    query, response, analysis, generation_metadata
                )

                return {
                    "agent_id": "quality_reviewer",
                    "evaluation": evaluation,
                    "response_evaluated": response,
                    "status": "completed",
                }

        except Exception as e:
            logger.error(f"Fehler bei Quality Agent Message: {e}")
            return {"error": str(e)}

    async def _handle_coordinator_message(self, message) -> Optional[Dict[str, Any]]:
        """Behandelt Nachrichten für den System Coordinator."""
        try:
            if message.message_type == MessageType.REQUEST:
                # System Status oder Koordinations-Anfragen
                content = message.content

                if "system_status" in content:
                    return await self.get_system_status()
                elif "performance_report" in content:
                    return await self.get_performance_report()

        except Exception as e:
            logger.error(f"Fehler bei Coordinator Message: {e}")
            return {"error": str(e)}

    async def _request_query_analysis(
        self, query: str, context: Dict
    ) -> Optional[Dict]:
        """Fordert Query-Analyse von Query Agent an."""
        try:
            response = await self.a2a_coordinator.request_response(
                "system_coordinator",
                "query_analyzer",
                {"task_data": {"query": query, "context": context}},
                timeout=30.0,
            )
            return response.get("analysis") if response else None
        except Exception as e:
            logger.error(f"Fehler bei Query-Analyse Request: {e}")
            return None

    async def _request_response_generation(
        self, query: str, task_data: Dict
    ) -> Optional[Dict]:
        """Fordert Response-Generierung von Response Agent an."""
        try:
            response = await self.a2a_coordinator.request_response(
                "system_coordinator",
                "response_generator",
                {"task_data": task_data},
                timeout=60.0,
            )
            return response
        except Exception as e:
            logger.error(f"Fehler bei Response-Generation Request: {e}")
            return None

    def _track_performance(self, result: Dict[str, Any], processing_time: float):
        """Verfolgt Performance-Metriken."""
        performance_data = {
            "timestamp": time.time(),
            "processing_time": processing_time,
            "quality_score": result.get("quality_score", 0.0),
            "iterations": result.get("iterations", 0),
            "success": result.get("metadata", {}).get("success", False),
            "sources_used": len(result.get("metadata", {}).get("sources_used", [])),
            "coordination_mode": result.get("metadata", {}).get(
                "coordination_mode", False
            ),
        }

        self.performance_history.append(performance_data)

        # Halte Historie begrenzt
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]

    async def _create_error_response(
        self, query: str, error: str, processing_time: float
    ) -> Dict[str, Any]:
        """Erstellt eine Fehler-Antwort."""
        return {
            "response": f"Entschuldigung, bei der Verarbeitung Ihrer Anfrage ist ein Fehler aufgetreten: {error}",
            "quality_score": 0.0,
            "iterations": 0,
            "total_processing_time": processing_time,
            "error": error,
            "metadata": {
                "original_query": query,
                "success": False,
                "error_occurred": True,
            },
        }

    async def get_system_status(self) -> Dict[str, Any]:
        """Gibt den aktuellen Systemstatus zurück."""
        a2a_status = (
            self.a2a_coordinator.get_system_status() if self.a2a_coordinator else {}
        )

        return {
            "initialized": self.initialized,
            "running": self.running,
            "agents": {
                "query_agent": self.query_agent is not None,
                "response_agent": self.response_agent is not None,
                "quality_agent": self.quality_agent is not None,
                "iteration_controller": self.iteration_controller is not None,
            },
            "a2a_coordinator": a2a_status,
            "configuration": {
                "max_iterations": self.max_iterations,
                "quality_threshold": self.quality_threshold,
                "mcp_tools_available": len(self.mcp_tools),
            },
            "performance": {
                "queries_processed": len(self.performance_history),
                "average_processing_time": self._calculate_average_processing_time(),
                "average_quality_score": self._calculate_average_quality(),
                "success_rate": self._calculate_success_rate(),
            },
        }

    async def get_performance_report(self) -> Dict[str, Any]:
        """Erstellt einen detaillierten Performance-Report."""
        if not self.performance_history:
            return {"message": "Keine Performance-Daten verfügbar"}

        recent_data = self.performance_history[-100:]  # Letzte 100 Queries

        a2a_report = (
            self.a2a_coordinator.get_performance_report()
            if self.a2a_coordinator
            else {}
        )

        return {
            "query_processing": {
                "total_queries": len(self.performance_history),
                "recent_queries": len(recent_data),
                "average_processing_time": sum(
                    d["processing_time"] for d in recent_data
                )
                / len(recent_data),
                "average_quality_score": sum(d["quality_score"] for d in recent_data)
                / len(recent_data),
                "average_iterations": sum(d["iterations"] for d in recent_data)
                / len(recent_data),
                "success_rate": sum(1 for d in recent_data if d["success"])
                / len(recent_data)
                * 100,
                "coordination_usage": sum(
                    1 for d in recent_data if d["coordination_mode"]
                )
                / len(recent_data)
                * 100,
            },
            "a2a_coordination": a2a_report,
            "trends": {
                "quality_trend": self._calculate_quality_trend(),
                "performance_trend": self._calculate_performance_trend(),
            },
        }

    def _calculate_average_processing_time(self) -> float:
        """Berechnet durchschnittliche Verarbeitungszeit."""
        if not self.performance_history:
            return 0.0
        return sum(d["processing_time"] for d in self.performance_history) / len(
            self.performance_history
        )

    def _calculate_average_quality(self) -> float:
        """Berechnet durchschnittliche Qualität."""
        if not self.performance_history:
            return 0.0
        return sum(d["quality_score"] for d in self.performance_history) / len(
            self.performance_history
        )

    def _calculate_success_rate(self) -> float:
        """Berechnet Erfolgsrate."""
        if not self.performance_history:
            return 0.0
        successful = sum(1 for d in self.performance_history if d["success"])
        return successful / len(self.performance_history) * 100

    def _calculate_quality_trend(self) -> str:
        """Berechnet Qualitäts-Trend."""
        if len(self.performance_history) < 20:
            return "insufficient_data"

        recent = self.performance_history[-10:]
        older = self.performance_history[-20:-10]

        recent_avg = sum(d["quality_score"] for d in recent) / len(recent)
        older_avg = sum(d["quality_score"] for d in older) / len(older)

        if recent_avg > older_avg + 5:
            return "improving"
        elif recent_avg < older_avg - 5:
            return "declining"
        else:
            return "stable"

    def _calculate_performance_trend(self) -> str:
        """Berechnet Performance-Trend."""
        if len(self.performance_history) < 20:
            return "insufficient_data"

        recent = self.performance_history[-10:]
        older = self.performance_history[-20:-10]

        recent_avg = sum(d["processing_time"] for d in recent) / len(recent)
        older_avg = sum(d["processing_time"] for d in older) / len(older)

        if recent_avg < older_avg - 1.0:  # 1 Sekunde schneller
            return "improving"
        elif recent_avg > older_avg + 1.0:  # 1 Sekunde langsamer
            return "declining"
        else:
            return "stable"

    async def shutdown(self):
        """Fährt das System ordnungsgemäß herunter."""
        logger.info("Starte System-Shutdown...")

        self.running = False

        if self.a2a_coordinator:
            await self.a2a_coordinator.stop()

        self.initialized = False
        logger.info("System-Shutdown abgeschlossen")
