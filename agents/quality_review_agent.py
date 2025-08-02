"""
Quality Review Agent - Bewertet die Qualität von generierten Antworten
"""

import logging
from typing import Dict, Any, List
import requests
import json
import os

logger = logging.getLogger(__name__)


class QualityReviewAgent:
    """
    Agent zur Bewertung der Qualität von generierten Antworten.
    Prüft Vollständigkeit, Genauigkeit, Relevanz und Kohärenz.
    """

    def __init__(self):
        """Initialisiert den Quality Review Agent mit OpenAI-kompatibler API."""
        # OpenAI-kompatible API Konfiguration (ollama, vllm, etc.)
        self.api_base = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        self.api_key = os.getenv(
            "OPENAI_API_KEY", "ollama"
        )  # ollama braucht keinen echten key
        self.model = os.getenv("OPENAI_MODEL", "qwen2.5:latest")
        self.quality_threshold = float(os.getenv("QUALITY_THRESHOLD", "95.0"))

        logger.info(f"QualityReviewAgent - API: {self.api_base}, Model: {self.model}")

    def _call_openai_compatible_api(
        self, messages: list, max_tokens: int = 1000
    ) -> str:
        """Ruft OpenAI-kompatible API auf (ollama, vllm, etc.)"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }

            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.1,
            }

            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"API Call Error: {e}")
            return None

    async def evaluate_response(
        self,
        query: str,
        response: str,
        analysis: Dict[str, Any],
        generation_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Bewertet die Qualität einer generierten Antwort.

        Args:
            query: Die ursprüngliche Nutzereingabe
            response: Die generierte Antwort
            analysis: Query-Analyse-Ergebnis
            generation_metadata: Metadaten der Antwortgenerierung

        Returns:
            Dictionary mit Qualitätsbewertung und Feedback
        """
        try:
            logger.info(f"Bewerte Antwortqualität für Query: {query[:50]}...")

            # Führe verschiedene Qualitätsprüfungen durch
            completeness_score = await self._evaluate_completeness(
                query, response, analysis
            )
            accuracy_score = await self._evaluate_accuracy(
                response, generation_metadata
            )
            relevance_score = await self._evaluate_relevance(query, response)
            coherence_score = await self._evaluate_coherence(response)

            # Berechne Gesamtbewertung
            overall_score = self._calculate_overall_score(
                completeness_score, accuracy_score, relevance_score, coherence_score
            )

            # Generiere spezifisches Feedback
            feedback = await self._generate_feedback(
                query,
                response,
                {
                    "completeness": completeness_score,
                    "accuracy": accuracy_score,
                    "relevance": relevance_score,
                    "coherence": coherence_score,
                    "overall": overall_score,
                },
            )

            is_acceptable = overall_score >= self.quality_threshold

            evaluation_result = {
                "overall_score": overall_score,
                "is_acceptable": is_acceptable,
                "scores": {
                    "completeness": completeness_score,
                    "accuracy": accuracy_score,
                    "relevance": relevance_score,
                    "coherence": coherence_score,
                },
                "feedback": feedback,
                "recommendations": self._generate_recommendations(
                    completeness_score, accuracy_score, relevance_score, coherence_score
                ),
                "metadata": {
                    "threshold": self.quality_threshold,
                    "generation_metadata": generation_metadata,
                },
            }

            logger.info(
                f"Qualitätsbewertung abgeschlossen: {overall_score:.1f}% (Schwelle: {self.quality_threshold}%)"
            )
            return evaluation_result

        except Exception as e:
            logger.error(f"Fehler bei der Qualitätsbewertung: {e}")
            return self._create_fallback_evaluation(query, response, str(e))

    async def _evaluate_completeness(
        self, query: str, response: str, analysis: Dict[str, Any]
    ) -> float:
        """
        Bewertet die Vollständigkeit der Antwort.

        Args:
            query: Die ursprüngliche Anfrage
            response: Die generierte Antwort
            analysis: Query-Analyse

        Returns:
            Vollständigkeitsscore (0-100)
        """
        try:
            intent = analysis.get("intent", "unbekannt")
            complexity = analysis.get("complexity", 1)

            completeness_prompt = f"""
            Bewerte die Vollständigkeit der folgenden Antwort SEHR STRENG im Hinblick auf die ursprüngliche Anfrage.

            Ursprüngliche Anfrage: "{query}"
            Intent: {intent}
            Komplexität: {complexity}/3

            Antwort: "{response}"

            ULTRA-STRENGE Bewertungskriterien (95%+ Standard):
            1. Werden ALLE Aspekte der Anfrage vollständig und tiefgreifend behandelt?
            2. Sind mindestens 3-5 konkrete, spezifische und aktuelle Beispiele enthalten?
            3. Sind echte URLs vorhanden (NIEMALS [URL_1], [URL_2] oder example.com Platzhalter)?
            4. Ist die Sprache grammatikalisch perfekt und professionell?
            5. Ist die Informationstiefe wissenschaftlich fundiert und präzise?
            6. Sind komplexe Zusammenhänge und Nuancen erklärt?
            7. Gibt es praktische Handlungsempfehlungen oder Lösungsansätze?
            8. Ist die Struktur logisch und gut gegliedert?

            KRITISCHE URL-BEWERTUNG:
            - URLs wie [URL_1], [URL_2], [URL_3] sind Platzhalter und führen zu MASSIVEM Punktabzug (-20 Punkte)!
            - Nur echte, vollständige URLs (z.B. https://www.nature.com/...) sind akzeptabel
            - Fehlende URLs sind besser als Platzhalter-URLs

            ULTRA-STRENGE Bewertung von 0-100, wobei nur Exzellenz akzeptiert wird:
            - 95-100: PERFEKT - Alle 8 Kriterien erfüllt, wissenschaftliche Präzision, ECHTE URLs, konkrete Beispiele, perfekte Sprache
            - 90-94: SEHR GUT - 7/8 Kriterien erfüllt, hohe Qualität, ECHTE URLs, gute Beispiele
            - 85-89: GUT - 6/8 Kriterien erfüllt, akzeptable Qualität, meist ECHTE URLs
            - 80-84: BEFRIEDIGEND - 5/8 Kriterien erfüllt, grundlegende Qualität, evtl. URL-Probleme
            - 70-79: AUSREICHEND - 4/8 Kriterien erfüllt, oberflächlich, URL-Platzhalter vorhanden
            - 60-69: MANGELHAFT - 3/8 Kriterien erfüllt, wichtige Mängel, viele URL-Platzhalter
            - 0-59: UNGENÜGEND - Weniger als 3/8 Kriterien erfüllt, inakzeptabel

            Antwort nur mit der Zahl (0-100):
            """

            response_content = self._call_openai_compatible_api(
                messages=[
                    {
                        "role": "system",
                        "content": "Du bist ein Experte für die Bewertung von Antwortqualität. Antworte nur mit einer Zahl zwischen 0 und 100.",
                    },
                    {"role": "user", "content": completeness_prompt},
                ],
                max_tokens=10,
            )

            if not response_content:
                raise Exception("API-Aufruf fehlgeschlagen")

            score_text = response_content.strip()

            # Extrahiere numerischen Score
            import re

            score_match = re.search(r"\d+", score_text)
            if score_match:
                score = float(score_match.group())
                return min(max(score, 0), 100)  # Clamp zwischen 0 und 100

            return 50.0  # Fallback-Score

        except Exception as e:
            logger.error(f"Fehler bei Vollständigkeitsbewertung: {e}")
            return 50.0

    async def _evaluate_accuracy(
        self, response: str, generation_metadata: Dict[str, Any]
    ) -> float:
        """
        Bewertet die Genauigkeit der Antwort basierend auf verwendeten Quellen.

        Args:
            response: Die generierte Antwort
            generation_metadata: Metadaten der Antwortgenerierung

        Returns:
            Genauigkeitsscore (0-100)
        """
        try:
            sources_used = generation_metadata.get("sources_used", [])
            information_completeness = generation_metadata.get(
                "information_completeness", 0.5
            )

            # Basis-Score - DEUTLICH strenger
            base_score = 40.0  # Reduziert von 60

            # VIEL höhere Anforderungen für Quellenbonusse
            if "rag" in sources_used:
                base_score += 20.0  # RAG ist vertrauenswürdig - erhöht
            if "web_search" in sources_used:
                base_score += 15.0  # Web-Suche für Aktualität - erhöht
            if "time" in sources_used:
                base_score += 5.0  # Zeitinformationen

            # SCHÄRFERE Bewertung der Informationsvollständigkeit
            # Nur bei sehr hoher Vollständigkeit gibt es Bonus
            if information_completeness >= 0.9:
                completeness_bonus = 25.0
            elif information_completeness >= 0.8:
                completeness_bonus = 15.0
            elif information_completeness >= 0.7:
                completeness_bonus = 5.0
            else:
                completeness_bonus = 0.0  # Kein Bonus bei schlechter Vollständigkeit

            base_score += completeness_bonus

            # HÄRTE Strafen für Widersprüche
            contradiction_penalty = await self._check_for_contradictions(response)
            base_score -= contradiction_penalty * 1.5  # Verstärkte Strafe

            # ZUSÄTZLICHE Überprüfung: Fake-Quellen erkennen und stark bestrafen
            fake_source_penalty = await self._check_for_fake_sources(response)
            base_score -= fake_source_penalty

            return min(max(base_score, 0), 100)

        except Exception as e:
            logger.error(f"Fehler bei Genauigkeitsbewertung: {e}")
            return 70.0

    async def _evaluate_relevance(self, query: str, response: str) -> float:
        """
        Bewertet die Relevanz der Antwort zur ursprünglichen Anfrage.

        Args:
            query: Die ursprüngliche Anfrage
            response: Die generierte Antwort

        Returns:
            Relevanzscore (0-100)
        """
        try:
            relevance_prompt = f"""
            Bewerte die Relevanz der folgenden Antwort SEHR STRENG zur ursprünglichen Anfrage.

            Anfrage: "{query}"
            Antwort: "{response}"

            ULTRA-STRENGE Relevanzbewertung (95%+ Standard):
            1. Beantwortet die Antwort DIREKT und VOLLSTÄNDIG die gestellte Frage?
            2. Sind ALLE Informationen in der Antwort hochrelevant für die Anfrage?
            3. Gibt es KEINE irrelevanten oder ablenkenden Informationen?
            4. Ist der Fokus präzise auf das gerichtet, was der Nutzer wissen will?
            5. Sind alle genannten Beispiele und Details DIREKT anwendbar?
            6. Wird die spezifische Intention der Anfrage perfekt erfasst?

            ULTRA-STRENGE Bewertung von 0-100:
            - 95-100: PERFEKT - Absolute Relevanz, jedes Wort ist wichtig, perfekte Fokussierung
            - 90-94: SEHR GUT - Fast perfekte Relevanz, minimal irrelevante Details
            - 85-89: GUT - Hohe Relevanz, aber einige unwichtige Informationen
            - 80-84: BEFRIEDIGEND - Grundsätzlich relevant, aber deutliche Abschweifungen
            - 70-79: AUSREICHEND - Teilweise relevant, aber wichtige Aspekte fehlen
            - 60-69: MANGELHAFT - Viele irrelevante Informationen, Fokus unklar
            - 0-59: UNGENÜGEND - Antwort verfehlt die Anfrage völlig
            3. Enthält die Antwort irrelevante oder ablenkende Informationen?
            4. Ist der Fokus der Antwort EXAKT auf die Frage ausgerichtet?
            5. Werden spezifische Aspekte der Frage ignoriert?

            STRENGE Bewertung von 0-100, wobei:
            - 90-100: Perfekt relevant, jedes Wort direkt auf den Punkt, vollständig fokussiert
            - 80-89: Sehr relevant, fast alle Punkte direkt angesprochen
            - 70-79: Relevant, aber einige unwichtige Zusatzinformationen
            - 60-69: Grundsätzlich relevant, aber wichtige Aspekte oberflächlich
            - 50-59: Teilweise relevant, wichtige Kernaspekte verfehlt
            - 30-49: Wenig relevant, geht an der Kernfrage vorbei
            - 0-29: Irrelevant oder völlig am Thema vorbei

            Antwort nur mit der Zahl (0-100):
            """

            response_content = self._call_openai_compatible_api(
                messages=[
                    {
                        "role": "system",
                        "content": "Du bist ein Experte für Relevanz-Bewertung. Antworte nur mit einer Zahl zwischen 0 und 100.",
                    },
                    {"role": "user", "content": relevance_prompt},
                ],
                max_tokens=10,
            )

            if not response_content:
                raise Exception("API-Aufruf fehlgeschlagen")

            score_text = response_content.strip()

            import re

            score_match = re.search(r"\d+", score_text)
            if score_match:
                score = float(score_match.group())
                return min(max(score, 0), 100)

            return 70.0

        except Exception as e:
            logger.error(f"Fehler bei Relevanzbewertung: {e}")
            return 70.0

    async def _evaluate_coherence(self, response: str) -> float:
        """
        Bewertet die Kohärenz und Verständlichkeit der Antwort.

        Args:
            response: Die generierte Antwort

        Returns:
            Kohärenzscore (0-100)
        """
        try:
            # Basis-Checks für Kohärenz
            base_score = 80.0

            # Längen-Check
            if len(response) < 50:
                base_score -= 20.0  # Zu kurz
            elif len(response) > 3000:
                base_score -= 10.0  # Möglicherweise zu lang

            # Struktur-Check (einfache Heuristiken)
            sentences = response.split(".")
            if len(sentences) < 2:
                base_score -= 15.0  # Zu wenige Sätze

            # Prüfe auf wiederholende Phrasen
            words = response.lower().split()
            unique_words = set(words)
            repetition_ratio = len(words) / len(unique_words) if unique_words else 1
            if repetition_ratio > 2.0:
                base_score -= 10.0  # Zu viele Wiederholungen

            # LLM-basierte Kohärenz-Bewertung für komplexere Checks
            coherence_score = await self._llm_coherence_check(response)

            # Gewichteter Durchschnitt
            final_score = (base_score * 0.4) + (coherence_score * 0.6)

            return min(max(final_score, 0), 100)

        except Exception as e:
            logger.error(f"Fehler bei Kohärenzbewertung: {e}")
            return 75.0

    async def _llm_coherence_check(self, response: str) -> float:
        """LLM-basierte Kohärenz-Prüfung."""
        try:
            coherence_prompt = f"""
            Bewerte die Kohärenz und Verständlichkeit des folgenden Textes.

            Text: "{response}"

            Bewertungskriterien:
            1. Ist der Text logisch strukturiert?
            2. Sind die Sätze klar und verständlich?
            3. Gibt es einen roten Faden?
            4. Ist die Sprache angemessen?
            5. Sind Übergänge zwischen Themen smooth?

            Gib eine Bewertung von 0-100 zurück.
            Antwort nur mit der Zahl (0-100):
            """

            response_content = self._call_openai_compatible_api(
                messages=[
                    {
                        "role": "system",
                        "content": "Du bist ein Experte für Textkohärenz. Antworte nur mit einer Zahl zwischen 0 und 100.",
                    },
                    {"role": "user", "content": coherence_prompt},
                ],
                max_tokens=10,
            )

            if not response_content:
                raise Exception("API-Aufruf fehlgeschlagen")

            score_text = response_content.strip()

            import re

            score_match = re.search(r"\d+", score_text)
            if score_match:
                return min(max(float(score_match.group()), 0), 100)

            return 75.0

        except Exception as e:
            logger.error(f"Fehler bei LLM-Kohärenz-Check: {e}")
            return 75.0

    async def _check_for_contradictions(self, response: str) -> float:
        """
        Prüft auf Widersprüche in der Antwort.

        Args:
            response: Die zu prüfende Antwort

        Returns:
            Penalty-Score (0-30)
        """
        try:
            # Einfache Widerspruchs-Prüfung
            contradiction_indicators = [
                ("ja", "nein"),
                ("wahr", "falsch"),
                ("korrekt", "inkorrekt"),
                ("möglich", "unmöglich"),
            ]

            response_lower = response.lower()
            penalty = 0.0

            for positive, negative in contradiction_indicators:
                if positive in response_lower and negative in response_lower:
                    penalty += 5.0

            # Prüfe auf numerische Widersprüche (sehr vereinfacht)
            import re

            numbers = re.findall(r"\d+", response)
            if len(numbers) > 1 and len(set(numbers)) != len(numbers):
                # Gleiche Zahlen können unterschiedliche Dinge bedeuten, also nur leichte Penalty
                penalty += 2.0

            return min(penalty, 30.0)

        except Exception as e:
            logger.error(f"Fehler bei Widerspruchs-Check: {e}")
            return 0.0

    async def _check_for_fake_sources(self, response: str) -> float:
        """
        Prüft auf Fake-Quellen und bestraft diese hart.

        Args:
            response: Die zu prüfende Antwort

        Returns:
            Penalty-Score (0-50)
        """
        try:
            import re

            # Bekannte Fake-Quelle Patterns
            fake_patterns = [
                r"example\.com",
                r"test\.com",
                r"sample\.org",
                r"dummy\.net",
                r"placeholder\.",
                r"lorem\.ipsum",
                r"https://example",
                r"http://test",
                r"www\.example",
                r"https://www\.example",
                r"(Quelle: Beispiel)",
                r"(beispielhafte Quelle)",
                r"Muster-URL",
                r"Platzhalter-Link",
            ]

            penalty = 0.0
            response_lower = response.lower()

            for pattern in fake_patterns:
                matches = re.findall(pattern, response_lower, re.IGNORECASE)
                penalty += len(matches) * 15.0  # Harte Strafe pro Fake-Quelle

            # Prüfe auf typische Fake-Content-Indikatoren
            fake_content_indicators = [
                "lorem ipsum",
                "beispieltext",
                "platzhaltertext",
                "dummy content",
                "test content",
                "mustertext",
            ]

            for indicator in fake_content_indicators:
                if indicator in response_lower:
                    penalty += 10.0

            return min(penalty, 50.0)  # Maximum 50 Punkte Abzug

        except Exception as e:
            logger.error(f"Fehler bei Fake-Quellen-Check: {e}")
            return 0.0

    def _calculate_overall_score(
        self, completeness: float, accuracy: float, relevance: float, coherence: float
    ) -> float:
        """
        ULTRA-STRENGE Gesamtbewertung für 95%+ Standard.

        Args:
            completeness: Vollständigkeitsscore
            accuracy: Genauigkeitsscore
            relevance: Relevanzscore
            coherence: Kohärenzscore

        Returns:
            Gesamtscore (0-100)
        """
        # VERSCHÄRFTE Gewichtung - Vollständigkeit und Genauigkeit sind kritisch
        weights = {
            "completeness": 0.40,  # Erhöht - Vollständigkeit ist kritisch
            "accuracy": 0.40,  # Erhöht - Genauigkeit ist kritisch
            "relevance": 0.15,  # Reduziert aber immer noch wichtig
            "coherence": 0.05,  # Reduziert - wichtig aber nicht kritisch
        }

        # Berechne gewichteten Durchschnitt
        overall_score = (
            completeness * weights["completeness"]
            + accuracy * weights["accuracy"]
            + relevance * weights["relevance"]
            + coherence * weights["coherence"]
        )

        # ZUSÄTZLICHE Härte: Wenn ein kritischer Bereich unter 85% liegt, starke Strafe
        critical_threshold = 85.0
        if completeness < critical_threshold or accuracy < critical_threshold:
            penalty = (critical_threshold - min(completeness, accuracy)) * 0.5
            overall_score -= penalty

        # ZUSÄTZLICHE Härte: Alle Bereiche müssen mindestens 80% haben für hohe Scores
        min_score = min(completeness, accuracy, relevance, coherence)
        if min_score < 80.0 and overall_score > 90.0:
            overall_score = 89.0  # Deckelung bei schwacher Einzelwertung

        return round(max(overall_score, 0), 1)

    async def _generate_feedback(
        self, query: str, response: str, scores: Dict[str, float]
    ) -> str:
        """
        Generiert spezifisches Feedback zur Antwortqualität.

        Args:
            query: Die ursprüngliche Anfrage
            response: Die generierte Antwort
            scores: Dictionary mit Einzelbewertungen

        Returns:
            Strukturiertes Feedback
        """
        try:
            feedback_prompt = f"""
            Generiere konstruktives Feedback für die folgende Antwort:

            Ursprüngliche Anfrage: "{query}"
            Antwort: "{response}"

            Bewertungen:
            - Vollständigkeit: {scores["completeness"]:.1f}/100
            - Genauigkeit: {scores["accuracy"]:.1f}/100
            - Relevanz: {scores["relevance"]:.1f}/100
            - Kohärenz: {scores["coherence"]:.1f}/100
            - Gesamt: {scores["overall"]:.1f}/100

            Erstelle spezifisches, konstruktives Feedback, das:
            1. Die Stärken der Antwort hervorhebt
            2. Verbesserungsmöglichkeiten konkret benennt
            3. Handlungsempfehlungen gibt
            4. Präzise und hilfreich ist

            Feedback:
            """

            response_content = self._call_openai_compatible_api(
                messages=[
                    {
                        "role": "system",
                        "content": "Du bist ein Experte für konstruktives Feedback zu Textqualität.",
                    },
                    {"role": "user", "content": feedback_prompt},
                ],
                max_tokens=500,
            )

            if not response_content:
                raise Exception("API-Aufruf fehlgeschlagen")

            feedback = response_content.strip()
            return feedback

        except Exception as e:
            logger.error(f"Fehler bei Feedback-Generierung: {e}")
            return self._generate_simple_feedback(scores)

    def _generate_simple_feedback(self, scores: Dict[str, float]) -> str:
        """Generiert einfaches Feedback basierend auf Scores."""
        feedback_parts = []

        if scores["overall"] >= 90:
            feedback_parts.append("Hervorragende Antwort!")
        elif scores["overall"] >= 70:
            feedback_parts.append("Gute Antwort mit Verbesserungspotential.")
        else:
            feedback_parts.append("Die Antwort benötigt deutliche Verbesserungen.")

        # Spezifisches Feedback zu niedrigen Scores
        if scores["completeness"] < 70:
            feedback_parts.append(
                "Die Antwort ist unvollständig und sollte alle Aspekte der Anfrage abdecken."
            )

        if scores["accuracy"] < 70:
            feedback_parts.append(
                "Die Genauigkeit sollte durch bessere Quellennutzung verbessert werden."
            )

        if scores["relevance"] < 70:
            feedback_parts.append(
                "Die Antwort sollte fokussierter auf die ursprüngliche Anfrage eingehen."
            )

        if scores["coherence"] < 70:
            feedback_parts.append(
                "Die Struktur und Klarheit der Antwort sollten verbessert werden."
            )

        return " ".join(feedback_parts)

    def _generate_recommendations(
        self, completeness: float, accuracy: float, relevance: float, coherence: float
    ) -> List[str]:
        """
        Generiert konkrete Verbesserungsempfehlungen.

        Returns:
            Liste von Verbesserungsempfehlungen
        """
        recommendations = []

        if completeness < 80:
            recommendations.append(
                "Alle Aspekte der ursprünglichen Anfrage vollständig behandeln"
            )

        if accuracy < 80:
            recommendations.append(
                "Zusätzliche oder verlässlichere Informationsquellen nutzen"
            )

        if relevance < 80:
            recommendations.append(
                "Fokus auf die Kernfrage beibehalten und irrelevante Informationen entfernen"
            )

        if coherence < 80:
            recommendations.append("Struktur und Klarheit der Antwort verbessern")

        if not recommendations:
            recommendations.append("Qualität beibehalten und ggf. Details ergänzen")

        return recommendations

    def _create_fallback_evaluation(
        self, query: str, response: str, error: str
    ) -> Dict[str, Any]:
        """
        Erstellt eine Fallback-Bewertung bei Fehlern.

        Args:
            query: Die ursprüngliche Anfrage
            response: Die generierte Antwort
            error: Fehlermeldung

        Returns:
            Fallback-Bewertung
        """
        # Einfache heuristische Bewertung
        base_score = 50.0

        if len(response) > 100:
            base_score += 10.0
        if query.lower() in response.lower():
            base_score += 15.0

        return {
            "overall_score": base_score,
            "is_acceptable": base_score >= self.quality_threshold,
            "scores": {
                "completeness": base_score,
                "accuracy": base_score - 10,
                "relevance": base_score + 10,
                "coherence": base_score,
            },
            "feedback": f"Automatische Bewertung aufgrund von Fehler: {error}",
            "recommendations": ["Manuelle Überprüfung der Antwort empfohlen"],
            "error": error,
            "metadata": {"fallback": True, "threshold": self.quality_threshold},
        }
