"""
A2A Coordinator - Agent-to-Agent Protocol Implementation
Koordiniert Kommunikation zwischen verschiedenen Agents
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Typen von Agent-zu-Agent Nachrichten."""

    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    STATUS = "status"
    ERROR = "error"
    COORDINATION = "coordination"


class AgentRole(Enum):
    """Rollen von Agents im System."""

    COORDINATOR = "coordinator"
    ANALYZER = "analyzer"
    GENERATOR = "generator"
    REVIEWER = "reviewer"
    SPECIALIST = "specialist"
    MONITOR = "monitor"


@dataclass
class A2AMessage:
    """Struktur einer Agent-zu-Agent Nachricht."""

    id: str
    sender: str
    receiver: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: float
    priority: int = 1  # 1=niedrig, 5=hoch
    requires_response: bool = False
    correlation_id: Optional[str] = None


class A2ACoordinator:
    """
    Koordiniert die Kommunikation zwischen verschiedenen Agents.
    Implementiert das Agent-to-Agent Protocol für erweiterte Koordination.
    """

    def __init__(self):
        """Initialisiert den A2A Coordinator."""
        self.agents = {}
        self.message_handlers = {}
        self.message_queue = asyncio.Queue()
        self.active_conversations = {}
        self.routing_table = {}
        self.performance_metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "response_times": [],
            "error_count": 0,
        }
        self.running = False

    async def start(self):
        """Startet den A2A Coordinator."""
        self.running = True
        logger.info("A2A Coordinator gestartet")

        # Starte Message Processing Loop
        asyncio.create_task(self._process_messages())

    async def stop(self):
        """Stoppt den A2A Coordinator."""
        self.running = False
        logger.info("A2A Coordinator gestoppt")

    def register_agent(
        self,
        agent_id: str,
        agent_instance: Any,
        role: AgentRole,
        capabilities: List[str],
        message_handler: Optional[Callable] = None,
    ):
        """
        Registriert einen Agent im A2A System.

        Args:
            agent_id: Eindeutige Agent-ID
            agent_instance: Agent-Instanz
            role: Rolle des Agents
            capabilities: Liste der Agent-Fähigkeiten
            message_handler: Handler für eingehende Nachrichten
        """
        self.agents[agent_id] = {
            "instance": agent_instance,
            "role": role,
            "capabilities": capabilities,
            "status": "active",
            "registered_at": time.time(),
        }

        if message_handler:
            self.message_handlers[agent_id] = message_handler

        # Update Routing Table
        self._update_routing_table()

        logger.info(f"Agent {agent_id} registriert mit Rolle {role.value}")

    def unregister_agent(self, agent_id: str):
        """Entfernt einen Agent aus dem A2A System."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            if agent_id in self.message_handlers:
                del self.message_handlers[agent_id]
            self._update_routing_table()
            logger.info(f"Agent {agent_id} deregistriert")

    async def send_message(
        self,
        sender_id: str,
        receiver_id: str,
        message_type: MessageType,
        content: Dict[str, Any],
        priority: int = 1,
        requires_response: bool = False,
        correlation_id: Optional[str] = None,
    ) -> str:
        """
        Sendet eine Nachricht zwischen Agents.

        Args:
            sender_id: Sender Agent-ID
            receiver_id: Empfänger Agent-ID
            message_type: Typ der Nachricht
            content: Nachrichteninhalt
            priority: Nachrichtenpriorität
            requires_response: Ob eine Antwort erwartet wird
            correlation_id: Korrelations-ID für Request-Response

        Returns:
            Message-ID
        """
        message_id = f"{sender_id}_{receiver_id}_{int(time.time() * 1000)}"

        message = A2AMessage(
            id=message_id,
            sender=sender_id,
            receiver=receiver_id,
            message_type=message_type,
            content=content,
            timestamp=time.time(),
            priority=priority,
            requires_response=requires_response,
            correlation_id=correlation_id,
        )

        await self.message_queue.put(message)
        self.performance_metrics["messages_sent"] += 1

        logger.debug(
            f"Nachricht {message_id} von {sender_id} an {receiver_id} gesendet"
        )
        return message_id

    async def broadcast_message(
        self,
        sender_id: str,
        message_type: MessageType,
        content: Dict[str, Any],
        target_roles: Optional[List[AgentRole]] = None,
        priority: int = 1,
    ) -> List[str]:
        """
        Broadcastet eine Nachricht an mehrere Agents.

        Args:
            sender_id: Sender Agent-ID
            message_type: Typ der Nachricht
            content: Nachrichteninhalt
            target_roles: Ziel-Rollen (None = alle)
            priority: Nachrichtenpriorität

        Returns:
            Liste der Message-IDs
        """
        message_ids = []

        for agent_id, agent_info in self.agents.items():
            if agent_id == sender_id:
                continue

            if target_roles and agent_info["role"] not in target_roles:
                continue

            message_id = await self.send_message(
                sender_id, agent_id, message_type, content, priority
            )
            message_ids.append(message_id)

        logger.info(f"Broadcast von {sender_id} an {len(message_ids)} Agents")
        return message_ids

    async def request_response(
        self,
        sender_id: str,
        receiver_id: str,
        content: Dict[str, Any],
        timeout: float = 30.0,
    ) -> Optional[Dict[str, Any]]:
        """
        Sendet Request und wartet auf Response.

        Args:
            sender_id: Sender Agent-ID
            receiver_id: Empfänger Agent-ID
            content: Request-Inhalt
            timeout: Timeout in Sekunden

        Returns:
            Response-Inhalt oder None bei Timeout
        """
        correlation_id = f"req_{int(time.time() * 1000)}"

        # Sende Request
        await self.send_message(
            sender_id,
            receiver_id,
            MessageType.REQUEST,
            content,
            priority=3,
            requires_response=True,
            correlation_id=correlation_id,
        )

        # Warte auf Response
        start_time = time.time()
        while time.time() - start_time < timeout:
            if correlation_id in self.active_conversations:
                response = self.active_conversations[correlation_id]
                del self.active_conversations[correlation_id]

                # Performance Tracking
                response_time = time.time() - start_time
                self.performance_metrics["response_times"].append(response_time)

                return response

            await asyncio.sleep(0.1)

        logger.warning(f"Request {correlation_id} timeout nach {timeout}s")
        return None

    async def coordinate_task(
        self,
        coordinator_id: str,
        task_description: str,
        required_capabilities: List[str],
        task_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Koordiniert eine komplexe Aufgabe zwischen mehreren Agents.

        Args:
            coordinator_id: Koordinator Agent-ID
            task_description: Beschreibung der Aufgabe
            required_capabilities: Benötigte Fähigkeiten
            task_data: Aufgaben-Daten

        Returns:
            Koordinations-Ergebnis
        """
        logger.info(f"Starte Task-Koordination: {task_description}")

        # Finde geeignete Agents
        suitable_agents = self._find_agents_by_capabilities(required_capabilities)

        if not suitable_agents:
            return {
                "success": False,
                "error": "Keine geeigneten Agents gefunden",
                "required_capabilities": required_capabilities,
            }

        coordination_id = f"coord_{int(time.time() * 1000)}"
        results = {}

        try:
            # Sende Koordinations-Nachrichten
            coordination_tasks = []

            for agent_id in suitable_agents:
                coordination_content = {
                    "coordination_id": coordination_id,
                    "task_description": task_description,
                    "task_data": task_data,
                    "role_assignment": self._assign_role_for_task(
                        agent_id, required_capabilities
                    ),
                }

                task = self.request_response(
                    coordinator_id, agent_id, coordination_content, timeout=60.0
                )
                coordination_tasks.append((agent_id, task))

            # Sammle Ergebnisse
            for agent_id, task in coordination_tasks:
                try:
                    result = await task
                    if result:
                        results[agent_id] = result
                except Exception as e:
                    logger.error(f"Fehler bei Agent {agent_id}: {e}")
                    results[agent_id] = {"error": str(e)}

            # Aggregiere Ergebnisse
            final_result = await self._aggregate_coordination_results(
                coordination_id, results, task_description
            )

            logger.info(
                f"Task-Koordination abgeschlossen: {len(results)} Agents beteiligt"
            )
            return final_result

        except Exception as e:
            logger.error(f"Fehler bei Task-Koordination: {e}")
            return {
                "success": False,
                "error": str(e),
                "coordination_id": coordination_id,
            }

    async def _process_messages(self):
        """Verarbeitet Nachrichten aus der Queue."""
        while self.running:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                await self._handle_message(message)
                self.performance_metrics["messages_received"] += 1

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Fehler bei Message-Processing: {e}")
                self.performance_metrics["error_count"] += 1

    async def _handle_message(self, message: A2AMessage):
        """Behandelt eine einzelne Nachricht."""
        try:
            receiver_id = message.receiver

            # Prüfe ob Empfänger existiert
            if receiver_id not in self.agents:
                logger.warning(f"Empfänger {receiver_id} nicht gefunden")
                return

            # Route zu Message Handler
            if receiver_id in self.message_handlers:
                handler = self.message_handlers[receiver_id]
                response = await handler(message)

                # Handle Response falls erwartet
                if message.requires_response and response:
                    if message.correlation_id:
                        self.active_conversations[message.correlation_id] = response
                    else:
                        # Sende Response zurück
                        await self.send_message(
                            receiver_id,
                            message.sender,
                            MessageType.RESPONSE,
                            response,
                            correlation_id=message.id,
                        )
            else:
                logger.warning(f"Kein Message Handler für Agent {receiver_id}")

        except Exception as e:
            logger.error(f"Fehler beim Behandeln der Nachricht {message.id}: {e}")

    def _update_routing_table(self):
        """Aktualisiert die Routing-Tabelle basierend auf registrierten Agents."""
        self.routing_table = {}

        for agent_id, agent_info in self.agents.items():
            capabilities = agent_info["capabilities"]
            role = agent_info["role"]

            # Route nach Fähigkeiten
            for capability in capabilities:
                if capability not in self.routing_table:
                    self.routing_table[capability] = []
                self.routing_table[capability].append(agent_id)

            # Route nach Rolle
            role_key = f"role:{role.value}"
            if role_key not in self.routing_table:
                self.routing_table[role_key] = []
            self.routing_table[role_key].append(agent_id)

    def _find_agents_by_capabilities(self, capabilities: List[str]) -> List[str]:
        """Findet Agents basierend auf benötigten Fähigkeiten."""
        suitable_agents = set()

        for capability in capabilities:
            if capability in self.routing_table:
                suitable_agents.update(self.routing_table[capability])

        return list(suitable_agents)

    def _assign_role_for_task(
        self, agent_id: str, required_capabilities: List[str]
    ) -> str:
        """Weist einem Agent eine Rolle für eine spezifische Aufgabe zu."""
        agent_info = self.agents.get(agent_id, {})
        agent_capabilities = agent_info.get("capabilities", [])

        # Finde beste Übereinstimmung
        matching_capabilities = set(agent_capabilities) & set(required_capabilities)

        if not matching_capabilities:
            return "support"

        # Bestimme Rolle basierend auf Fähigkeiten
        if "analysis" in matching_capabilities:
            return "analyzer"
        elif "generation" in matching_capabilities:
            return "generator"
        elif "review" in matching_capabilities:
            return "reviewer"
        else:
            return "specialist"

    async def _aggregate_coordination_results(
        self, coordination_id: str, results: Dict[str, Any], task_description: str
    ) -> Dict[str, Any]:
        """Aggregiert Ergebnisse einer koordinierten Aufgabe."""
        try:
            successful_results = {k: v for k, v in results.items() if "error" not in v}

            error_results = {k: v for k, v in results.items() if "error" in v}

            aggregated_result = {
                "coordination_id": coordination_id,
                "task_description": task_description,
                "success": len(successful_results) > 0,
                "agents_participated": len(results),
                "agents_successful": len(successful_results),
                "agents_failed": len(error_results),
                "results": successful_results,
                "errors": error_results,
                "aggregated_data": {},
                "timestamp": time.time(),
            }

            # Aggregiere erfolgreiche Ergebnisse
            if successful_results:
                # Sammle alle Daten
                all_data = []
                for result in successful_results.values():
                    if isinstance(result, dict) and "data" in result:
                        all_data.append(result["data"])

                aggregated_result["aggregated_data"] = {
                    "combined_results": all_data,
                    "count": len(all_data),
                    "sources": list(successful_results.keys()),
                }

            return aggregated_result

        except Exception as e:
            logger.error(f"Fehler bei Result-Aggregation: {e}")
            return {
                "coordination_id": coordination_id,
                "success": False,
                "error": f"Aggregation failed: {e}",
                "raw_results": results,
            }

    def get_system_status(self) -> Dict[str, Any]:
        """Gibt den aktuellen Systemstatus zurück."""
        return {
            "coordinator_running": self.running,
            "registered_agents": len(self.agents),
            "active_conversations": len(self.active_conversations),
            "queue_size": self.message_queue.qsize(),
            "performance_metrics": self.performance_metrics.copy(),
            "agents": {
                agent_id: {
                    "role": info["role"].value,
                    "capabilities": info["capabilities"],
                    "status": info["status"],
                }
                for agent_id, info in self.agents.items()
            },
            "routing_table_size": len(self.routing_table),
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """Erstellt einen Performance-Report."""
        metrics = self.performance_metrics
        response_times = metrics["response_times"]

        report = {
            "total_messages_sent": metrics["messages_sent"],
            "total_messages_received": metrics["messages_received"],
            "total_errors": metrics["error_count"],
            "error_rate": metrics["error_count"]
            / max(metrics["messages_received"], 1)
            * 100,
            "response_time_stats": {
                "count": len(response_times),
                "average": sum(response_times) / len(response_times)
                if response_times
                else 0,
                "min": min(response_times) if response_times else 0,
                "max": max(response_times) if response_times else 0,
            },
            "system_health": "good" if metrics["error_count"] < 10 else "degraded",
            "agent_activity": {
                agent_id: info["status"] for agent_id, info in self.agents.items()
            },
        }

        return report
