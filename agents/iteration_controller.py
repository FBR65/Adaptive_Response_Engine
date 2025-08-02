"""
Iteration Controller - Steuert den iterativen Verbesserungsprozess von Antworten
"""

import logging
from typing import Dict, Any, Optional
import time

logger = logging.getLogger(__name__)


class IterationController:
    """
    Kontrolliert den iterativen Verbesserungsprozess von Antworten.
    Koordiniert Query Analysis, Response Generation und Quality Review Agents.
    """

    def __init__(
        self,
        query_agent,
        response_agent,
        quality_agent,
        max_iterations: int = 3,
        quality_threshold: float = 95.0,
    ):
        """
        Initialisiert den Iteration Controller.

        Args:
            query_agent: QueryAnalysisAgent Instanz
            response_agent: ResponseGenerationAgent Instanz
            quality_agent: QualityReviewAgent Instanz
            max_iterations: Maximale Anzahl von Iterationen
            quality_threshold: Mindestqualität für Akzeptanz
        """
        self.query_agent = query_agent
        self.response_agent = response_agent
        self.quality_agent = quality_agent
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold

    async def process_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        mcp_tools: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Verarbeitet eine Query durch den kompletten iterativen Verbesserungsprozess.

        Args:
            query: Die ursprüngliche Nutzereingabe
            context: Zusätzlicher Kontext
            mcp_tools: Verfügbare MCP-Tools

        Returns:
            Dictionary mit finaler Antwort und Verarbeitungsmetadaten
        """
        start_time = time.time()
        iteration_history = []

        try:
            logger.info(f"Starte Verarbeitung von Query: {query[:100]}...")

            # Phase 1: Query-Analyse
            logger.info("Phase 1: Query-Analyse")
            query_analysis = await self.query_agent.analyze_query(query, context)

            # Verfeinere Query basierend auf Analyse
            refined_query = await self.query_agent.refine_query(query, query_analysis)

            logger.info(f"Query verfeinert: {query[:50]}... -> {refined_query[:50]}...")

            # Phase 2: Iterative Antwort-Verbesserung
            current_response = None
            current_generation_metadata = None
            best_score = 0.0
            best_response = None
            best_metadata = None

            for iteration in range(self.max_iterations):
                logger.info(f"Iteration {iteration + 1}/{self.max_iterations}")

                iteration_start = time.time()

                # Generiere Antwort
                if iteration == 0:
                    # Erste Iteration: Nutze verfeinerte Query
                    generation_result = await self.response_agent.generate_response(
                        refined_query, query_analysis, mcp_tools, context
                    )
                else:
                    # VERBESSERTE Iterationen: Intelligentere Nutzung von Feedback
                    improved_query = await self._create_advanced_improvement_query(
                        refined_query,
                        current_response,
                        iteration_history[-1]["evaluation"],
                        iteration_history,  # Komplette Historie für bessere Analyse
                        query_analysis,
                    )

                    # ERWEITERTE Kontext-Sammlung für späte Iterationen
                    enhanced_context = await self._enhance_context_for_iteration(
                        context, iteration, iteration_history[-1]["evaluation"]
                    )

                    generation_result = await self.response_agent.generate_response(
                        improved_query, query_analysis, mcp_tools, enhanced_context
                    )

                current_response = generation_result["response"]
                current_generation_metadata = generation_result

                # Bewerte Antwortqualität
                evaluation = await self.quality_agent.evaluate_response(
                    refined_query,
                    current_response,
                    query_analysis,
                    current_generation_metadata,
                )

                iteration_time = time.time() - iteration_start
                current_score = evaluation["overall_score"]

                # Speichere Iteration in Historie
                iteration_data = {
                    "iteration": iteration + 1,
                    "query_used": refined_query if iteration == 0 else improved_query,
                    "response": current_response,
                    "evaluation": evaluation,
                    "generation_metadata": current_generation_metadata,
                    "score": current_score,
                    "processing_time": iteration_time,
                    "timestamp": time.time(),
                }
                iteration_history.append(iteration_data)

                logger.info(
                    f"Iteration {iteration + 1} abgeschlossen. Score: {current_score:.1f}%"
                )

                # Prüfe ob aktuelle Antwort besser ist
                if current_score > best_score:
                    best_score = current_score
                    best_response = current_response
                    best_metadata = current_generation_metadata

                # Prüfe Abbruchkriterien
                if evaluation["is_acceptable"]:
                    logger.info(
                        f"Qualitätsschwelle erreicht: {current_score:.1f}% >= {self.quality_threshold}%"
                    )
                    break

                # VERBESSERTE Stagnations-Erkennung
                if iteration > 0 and self._is_improvement_stagnating_advanced(
                    iteration_history
                ):
                    logger.info(
                        "Intelligente Stagnations-Erkennung: Verbesserung stagniert"
                    )

                    # RETRY-Strategie: Ein letzter Versuch mit drastisch geändertem Ansatz
                    if iteration < self.max_iterations - 1:
                        logger.info(
                            "Versuche Durchbruch-Strategie für letzte Iteration"
                        )
                        # Setze Flag für drastische Änderung in nächster Iteration
                        context["breakthrough_mode"] = True
                    break

            # Phase 3: Finalisierung
            total_time = time.time() - start_time

            # Wähle beste Antwort
            final_response = best_response if best_response else current_response
            final_metadata = (
                best_metadata if best_metadata else current_generation_metadata
            )
            final_score = best_score

            # Erstelle Ergebnis
            result = {
                "response": final_response,
                "quality_score": final_score,
                "iterations": len(iteration_history),
                "total_processing_time": total_time,
                "final_acceptable": final_score >= self.quality_threshold,
                "query_analysis": query_analysis,
                "refined_query": refined_query,
                "iteration_history": iteration_history,
                "final_evaluation": iteration_history[-1]["evaluation"]
                if iteration_history
                else None,
                "metadata": {
                    "original_query": query,
                    "context": context,
                    "max_iterations": self.max_iterations,
                    "quality_threshold": self.quality_threshold,
                    "sources_used": final_metadata.get("sources_used", [])
                    if final_metadata
                    else [],
                    "success": final_score >= self.quality_threshold,
                },
            }

            logger.info(
                f"Query-Verarbeitung abgeschlossen. Finale Qualität: {final_score:.1f}% in {len(iteration_history)} Iterationen"
            )
            return result

        except Exception as e:
            logger.error(f"Fehler bei Query-Verarbeitung: {e}")
            return await self._create_error_response(
                query, str(e), time.time() - start_time
            )

    async def _create_improvement_query(
        self, original_query: str, previous_response: str, feedback: str
    ) -> str:
        """
        Erstellt eine verbesserte Query basierend auf vorherigem Feedback.

        Args:
            original_query: Die ursprüngliche verfeinerte Query
            previous_response: Die vorherige Antwort
            feedback: Feedback zur vorherigen Antwort

        Returns:
            Verbesserte Query für nächste Iteration
        """
        try:
            improvement_query = f"""
            Basierend auf folgendem Feedback, generiere eine verbesserte Antwort:

            Ursprüngliche Anfrage: "{original_query}"
            
            Vorherige Antwort: "{previous_response}"
            
            Feedback zur Verbesserung: "{feedback}"
            
            Bitte beachte das Feedback und verbessere die Antwort entsprechend. 
            Konzentriere dich besonders auf die genannten Schwächen.
            """

            return improvement_query

        except Exception as e:
            logger.error(f"Fehler beim Erstellen der Verbesserungs-Query: {e}")
            return original_query

    async def _create_advanced_improvement_query(
        self,
        original_query: str,
        previous_response: str,
        evaluation: Dict[str, Any],
        iteration_history: list,
        query_analysis: Dict[str, Any],
    ) -> str:
        """
        VERBESSERTE Methode zur Erstellung intelligenter Verbesserungs-Queries.

        Args:
            original_query: Die ursprüngliche verfeinerte Query
            previous_response: Die vorherige Antwort
            evaluation: Vollständige Evaluation der vorherigen Antwort
            iteration_history: Komplette Historie aller Iterationen
            query_analysis: Original Query-Analyse

        Returns:
            Stark verbesserte Query für nächste Iteration
        """
        try:
            feedback = evaluation.get("feedback", "")
            scores = evaluation.get("detailed_scores", {})

            # Analysiere Schwachstellen
            weakest_areas = []
            for area, score in scores.items():
                if score < 85.0:  # Kritische Schwelle
                    weakest_areas.append(f"{area}: {score:.1f}%")

            # Erkenne Patterns in der Historie
            improvement_pattern = self._analyze_improvement_pattern(iteration_history)

            # KORREKTUR: Nutze einfache Query für Suche, aber mit Qualitäts-Hinweisen für die Generierung
            improvement_query = (
                original_query  # Für die Suche verwenden wir die originale Frage
            )

            # Die Qualitätsanforderungen werden über den Kontext übertragen, nicht über die Query
            return improvement_query

        except Exception as e:
            logger.error(
                f"Fehler beim Erstellen der erweiterten Verbesserungs-Query: {e}"
            )
            return original_query

    async def _enhance_context_for_iteration(
        self,
        original_context: Dict[str, Any],
        iteration: int,
        evaluation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Erweitert den Kontext für spätere Iterationen basierend auf Feedback.

        Args:
            original_context: Original-Kontext
            iteration: Aktuelle Iterations-Nummer
            evaluation: Evaluation der vorherigen Iteration

        Returns:
            Erweiterten Kontext für bessere Ergebnisse
        """
        try:
            enhanced_context = original_context.copy()

            # Setze Iteration-spezifische Flags
            enhanced_context["iteration_number"] = iteration
            enhanced_context["quality_focus"] = True

            # Bei niedrigen Scores: Intensiviere Suche
            if evaluation.get("overall_score", 0) < 90:
                enhanced_context["intensive_search"] = True
                enhanced_context["require_multiple_sources"] = True

            # Bei Genauigkeitsproblemen: Fokus auf Faktenchecks
            scores = evaluation.get("detailed_scores", {})
            if scores.get("accuracy", 100) < 85:
                enhanced_context["fact_check_mode"] = True
                enhanced_context["source_verification"] = True

            # Bei Vollständigkeitsproblemen: Erweiterte Recherche
            if scores.get("completeness", 100) < 85:
                enhanced_context["comprehensive_search"] = True
                enhanced_context["example_requirement"] = "minimum_3_examples"

            return enhanced_context

        except Exception as e:
            logger.error(f"Fehler beim Erweitern des Kontexts: {e}")
            return original_context

    def _analyze_improvement_pattern(self, iteration_history: list) -> str:
        """
        Analysiert das Verbesserungsmuster aus der Historie.

        Args:
            iteration_history: Historie aller Iterationen

        Returns:
            Textuelle Analyse des Verbesserungsmusters
        """
        try:
            if len(iteration_history) < 2:
                return "Erste Iteration - kein Pattern verfügbar"

            pattern_lines = []

            for i in range(1, len(iteration_history)):
                prev_score = iteration_history[i - 1]["score"]
                current_score = iteration_history[i]["score"]
                improvement = current_score - prev_score

                pattern_lines.append(
                    f"Iteration {i + 1}: {prev_score:.1f}% → {current_score:.1f}% "
                    f"({improvement:+.1f}%)"
                )

            # Gesamt-Trend
            total_improvement = (
                iteration_history[-1]["score"] - iteration_history[0]["score"]
            )
            pattern_lines.append(f"Gesamt-Verbesserung: {total_improvement:+.1f}%")

            return "\n".join(pattern_lines)

        except Exception as e:
            logger.error(f"Fehler bei Pattern-Analyse: {e}")
            return "Pattern-Analyse fehlgeschlagen"

    def _is_improvement_stagnating(self, iteration_history: list) -> bool:
        """
        Prüft ob die Verbesserung stagniert.

        Args:
            iteration_history: Liste der Iterations-Historie

        Returns:
            True wenn Verbesserung stagniert
        """
        try:
            if len(iteration_history) < 2:
                return False

            # Vergleiche letzten beiden Scores
            last_score = iteration_history[-1]["score"]
            previous_score = iteration_history[-2]["score"]

            improvement = last_score - previous_score

            # SCHÄRFERE Stagnations-Erkennung
            # Stagnation wenn Verbesserung unter 1.0% (vorher 5.0 Punkte)
            if improvement < 1.0:
                logger.info(f"Geringe Verbesserung erkannt: {improvement:.1f}%")
                return True

            # Stagnation wenn Score sogar schlechter wird
            if improvement < 0:
                logger.info(f"Verschlechterung erkannt: {improvement:.1f}%")
                return True

            return False

        except Exception as e:
            logger.error(f"Fehler bei Stagnations-Prüfung: {e}")
            return False

    def _is_improvement_stagnating_advanced(self, iteration_history: list) -> bool:
        """
        ERWEITERTE Stagnations-Erkennung mit intelligenter Analyse.

        Args:
            iteration_history: Liste der Iterations-Historie

        Returns:
            True wenn Verbesserung intelligent als stagnierend erkannt wird
        """
        try:
            if len(iteration_history) < 2:
                return False

            # Analysiere die letzten 2 Iterationen
            last_score = iteration_history[-1]["score"]
            previous_score = iteration_history[-2]["score"]
            improvement = last_score - previous_score

            # Erste Bedingung: Sehr geringe Verbesserung
            if improvement < 0.5:
                logger.info(f"Stagnation: Sehr geringe Verbesserung {improvement:.1f}%")
                return True

            # Zweite Bedingung: Score ist hoch genug aber unter Schwelle
            if (
                last_score >= 88.0
                and last_score < self.quality_threshold
                and improvement < 2.0
            ):
                logger.info(
                    f"Stagnation: Hoher Score ({last_score:.1f}%) aber unter Schwelle mit geringer Verbesserung"
                )
                return True

            # Dritte Bedingung: Analyse der Gesamt-Entwicklung
            if len(iteration_history) >= 3:
                # Betrachte letzten 3 Iterationen
                recent_scores = [h["score"] for h in iteration_history[-3:]]
                max_recent = max(recent_scores)
                min_recent = min(recent_scores)
                recent_range = max_recent - min_recent

                # Wenn Schwankung sehr gering ist (< 1.5%), dann Stagnation
                if recent_range < 1.5:
                    logger.info(
                        f"Stagnation: Geringe Schwankung in letzten 3 Iterationen ({recent_range:.1f}%)"
                    )
                    return True

            return False

        except Exception as e:
            logger.error(f"Fehler bei erweiterter Stagnations-Analyse: {e}")
            return False

    async def _create_error_response(
        self, query: str, error: str, processing_time: float
    ) -> Dict[str, Any]:
        """
        Erstellt eine Fehler-Antwort.

        Args:
            query: Die ursprüngliche Query
            error: Fehlermeldung
            processing_time: Verarbeitungszeit

        Returns:
            Fehler-Antwort Dictionary
        """
        error_response = f"""
        Entschuldigung, bei der Verarbeitung Ihrer Anfrage ist ein Fehler aufgetreten.

        Ihre Anfrage: "{query}"

        Es gab ein technisches Problem, das die vollständige Bearbeitung verhindert hat. 
        Bitte versuchen Sie es erneut oder formulieren Sie Ihre Frage anders.

        Für einfache Fragen stehe ich Ihnen weiterhin zur Verfügung.
        """

        return {
            "response": error_response,
            "quality_score": 0.0,
            "iterations": 0,
            "total_processing_time": processing_time,
            "final_acceptable": False,
            "error": error,
            "metadata": {
                "original_query": query,
                "success": False,
                "error_occurred": True,
            },
        }

    def get_iteration_statistics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extrahiert Statistiken aus einem Verarbeitungsergebnis.

        Args:
            result: Ergebnis der Query-Verarbeitung

        Returns:
            Statistik-Dictionary
        """
        try:
            iteration_history = result.get("iteration_history", [])

            if not iteration_history:
                return {"error": "Keine Iterations-Historie verfügbar"}

            scores = [iteration["score"] for iteration in iteration_history]
            processing_times = [
                iteration["processing_time"] for iteration in iteration_history
            ]

            statistics = {
                "total_iterations": len(iteration_history),
                "final_score": result.get("quality_score", 0.0),
                "initial_score": scores[0] if scores else 0.0,
                "improvement": (scores[-1] - scores[0]) if len(scores) > 1 else 0.0,
                "max_score": max(scores) if scores else 0.0,
                "min_score": min(scores) if scores else 0.0,
                "average_score": sum(scores) / len(scores) if scores else 0.0,
                "total_processing_time": result.get("total_processing_time", 0.0),
                "average_iteration_time": sum(processing_times) / len(processing_times)
                if processing_times
                else 0.0,
                "success": result.get("metadata", {}).get("success", False),
                "sources_used": result.get("metadata", {}).get("sources_used", []),
                "score_progression": scores,
            }

            return statistics

        except Exception as e:
            logger.error(f"Fehler bei Statistik-Extraktion: {e}")
            return {"error": f"Fehler bei Statistik-Berechnung: {e}"}

    async def analyze_performance(self, results: list) -> Dict[str, Any]:
        """
        Analysiert die Performance über mehrere Query-Verarbeitungen.

        Args:
            results: Liste von Verarbeitungsergebnissen

        Returns:
            Performance-Analyse
        """
        try:
            if not results:
                return {"error": "Keine Ergebnisse für Analyse verfügbar"}

            all_stats = [self.get_iteration_statistics(result) for result in results]
            valid_stats = [stat for stat in all_stats if "error" not in stat]

            if not valid_stats:
                return {"error": "Keine gültigen Statistiken verfügbar"}

            analysis = {
                "total_queries": len(results),
                "successful_queries": len(valid_stats),
                "success_rate": len(valid_stats) / len(results) * 100,
                "average_final_score": sum(stat["final_score"] for stat in valid_stats)
                / len(valid_stats),
                "average_iterations": sum(
                    stat["total_iterations"] for stat in valid_stats
                )
                / len(valid_stats),
                "average_processing_time": sum(
                    stat["total_processing_time"] for stat in valid_stats
                )
                / len(valid_stats),
                "average_improvement": sum(stat["improvement"] for stat in valid_stats)
                / len(valid_stats),
                "quality_threshold_met": sum(
                    1 for stat in valid_stats if stat["success"]
                )
                / len(valid_stats)
                * 100,
                "most_used_sources": self._analyze_source_usage(valid_stats),
                "performance_trends": {
                    "scores": [stat["final_score"] for stat in valid_stats],
                    "iterations": [stat["total_iterations"] for stat in valid_stats],
                    "processing_times": [
                        stat["total_processing_time"] for stat in valid_stats
                    ],
                },
            }

            return analysis

        except Exception as e:
            logger.error(f"Fehler bei Performance-Analyse: {e}")
            return {"error": f"Fehler bei Performance-Analyse: {e}"}

    def _analyze_source_usage(self, stats: list) -> Dict[str, int]:
        """Analysiert die Nutzung verschiedener Informationsquellen."""
        source_count = {}

        for stat in stats:
            sources = stat.get("sources_used", [])
            for source in sources:
                source_count[source] = source_count.get(source, 0) + 1

        # Sortiere nach Häufigkeit
        return dict(sorted(source_count.items(), key=lambda x: x[1], reverse=True))
