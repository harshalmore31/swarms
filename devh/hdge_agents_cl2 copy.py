import json
import logging
import datetime
import random  # For simulating tool results
from dataclasses import dataclass, field, InitVar, MISSING
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AssetClass(str, Enum):
    """Defines the asset classes that agents can specialize in."""
    BITCOIN = "Bitcoin"
    STOCKS = "Stocks"
    FOREX = "Forex"

class Action(str, Enum):
    """Defines the possible trading actions."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    HEDGE = "HEDGE"  # Added for hedging strategies
    REBALANCE = "REBALANCE"  # Added for portfolio rebalancing
    STOP_LOSS = "STOP_LOSS"  # Added for risk management
    TAKE_PROFIT = "TAKE_PROFIT"  # Added for profit taking
    RISK_ASSESSMENT = "RISK_ASSESSMENT"  # Added for risk analysis

class MarketCondition(str, Enum):
    """Defines possible market conditions that might trigger special conferences."""
    NORMAL = "NORMAL"
    VOLATILE = "VOLATILE"
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    EXTREME_VOLATILITY = "EXTREME_VOLATILITY"
    BLACK_SWAN = "BLACK_SWAN"  # Rare, unpredictable events

@dataclass
class MarketData:
    """Represents market data for a specific asset class."""
    asset_class: AssetClass
    price: float
    timestamp: datetime.datetime
    volume: float = 0.0
    open_price: float = 0.0
    high_price: float = 0.0
    low_price: float = 0.0
    close_price: float = 0.0
    news: List[Dict[str, str]] = field(default_factory=list)
    technical_indicators: Dict[str, float] = field(default_factory=dict)
    market_sentiment: float = 0.0  # -1.0 to 1.0
    volatility: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Converts to dictionary for serialization."""
        return {
            "asset_class": self.asset_class,
            "price": self.price,
            "timestamp": self.timestamp.isoformat(),
            "volume": self.volume,
            "open_price": self.open_price,
            "high_price": self.high_price,
            "low_price": self.low_price,
            "close_price": self.close_price,
            "news": self.news,
            "technical_indicators": self.technical_indicators,
            "market_sentiment": self.market_sentiment,
            "volatility": self.volatility
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Creates a MarketData instance from a dictionary."""
        data_copy = data.copy()
        data_copy["timestamp"] = datetime.datetime.fromisoformat(data_copy["timestamp"])
        return cls(**data_copy)

# Instead of using inheritance for memory entries which causes field ordering issues,
# we'll create standalone dataclasses with all required fields explicitly defined

@dataclass
class MarketInfoMemory:
    """Represents a Market Information Memory (MMI) entry."""
    timestamp: datetime.datetime
    asset_class: AssetClass
    market_data: Dict[str, Any]
    analysis_summary: str
    importance: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Converts to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "asset_class": self.asset_class,
            "market_data": self.market_data,
            "analysis_summary": self.analysis_summary,
            "importance": self.importance
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Creates a memory entry from a dictionary."""
        data_copy = data.copy()
        data_copy["timestamp"] = datetime.datetime.fromisoformat(data_copy["timestamp"])
        return cls(**data_copy)

@dataclass
class InvestmentReflectionMemory:
    """Represents an Investment Reflection Memory (MIR) entry."""
    timestamp: datetime.datetime
    asset_class: AssetClass
    action_taken: Action
    reasoning: str
    importance: float = 1.0
    outcome: Optional[str] = None
    profit_loss: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "asset_class": self.asset_class,
            "action_taken": self.action_taken,
            "reasoning": self.reasoning,
            "importance": self.importance,
            "outcome": self.outcome,
            "profit_loss": self.profit_loss
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Creates a memory entry from a dictionary."""
        data_copy = data.copy()
        data_copy["timestamp"] = datetime.datetime.fromisoformat(data_copy["timestamp"])
        return cls(**data_copy)

@dataclass
class GeneralExperienceMemory:
    """Represents a General Experience Memory (MGE) entry."""
    timestamp: datetime.datetime
    conference_type: str  # BAC, ESC, EMC
    key_insights: List[str]
    importance: float = 1.0
    budget_allocation: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "conference_type": self.conference_type,
            "key_insights": self.key_insights,
            "importance": self.importance,
            "budget_allocation": self.budget_allocation
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Creates a memory entry from a dictionary."""
        data_copy = data.copy()
        data_copy["timestamp"] = datetime.datetime.fromisoformat(data_copy["timestamp"])
        return cls(**data_copy)

@dataclass
class AgentMemory:
    """Represents an agent's memory with typed entries."""
    mmi: List[MarketInfoMemory] = field(default_factory=list)  # Market Information Memory
    mir: List[InvestmentReflectionMemory] = field(default_factory=list)  # Investment Reflection Memory
    mge: List[GeneralExperienceMemory] = field(default_factory=list)  # General Experience Memory

    def add_market_info(self, info: MarketInfoMemory):
        """Add a market information memory entry."""
        self.mmi.append(info)

    def add_investment_reflection(self, reflection: InvestmentReflectionMemory):
        """Add an investment reflection memory entry."""
        self.mir.append(reflection)

    def add_general_experience(self, experience: GeneralExperienceMemory):
        """Add a general experience memory entry."""
        self.mge.append(experience)

    def retrieve_relevant_market_info(self, asset_class: AssetClass, limit: int = 5) -> List[MarketInfoMemory]:
        """Retrieve relevant market information memories for a specific asset class."""
        relevant_info = [info for info in self.mmi if info.asset_class == asset_class]
        relevant_info.sort(key=lambda x: (x.importance, x.timestamp), reverse=True)
        return relevant_info[:limit]

    def retrieve_relevant_reflections(self, asset_class: AssetClass, limit: int = 3) -> List[InvestmentReflectionMemory]:
        """Retrieve relevant investment reflections for a specific asset class."""
        relevant_reflections = [ref for ref in self.mir if ref.asset_class == asset_class]
        relevant_reflections.sort(key=lambda x: (x.importance, x.timestamp), reverse=True)
        return relevant_reflections[:limit]

    def retrieve_latest_conference_insights(self, conference_type: str, limit: int = 3) -> List[GeneralExperienceMemory]:
        """Retrieve the latest conference insights of a specific type."""
        relevant_experiences = [exp for exp in self.mge if exp.conference_type == conference_type]
        relevant_experiences.sort(key=lambda x: x.timestamp, reverse=True)
        return relevant_experiences[:limit]

    def to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """Converts the memory to a dictionary for serialization."""
        return {
            "mmi": [entry.to_dict() for entry in self.mmi],
            "mir": [entry.to_dict() for entry in self.mir],
            "mge": [entry.to_dict() for entry in self.mge]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, List[Dict[str, Any]]]):
        """Creates a Memory instance from a dictionary."""
        memory = cls()

        for entry in data.get("mmi", []):
            memory.mmi.append(MarketInfoMemory.from_dict(entry))

        for entry in data.get("mir", []):
            memory.mir.append(InvestmentReflectionMemory.from_dict(entry))

        for entry in data.get("mge", []):
            memory.mge.append(GeneralExperienceMemory.from_dict(entry))

        return memory

class AnalystAgent:
    """Base class for specialized analyst agents (Bitcoin, Stocks, Forex)."""
    def __init__(
        self,
        agent_name: str,
        asset_class: AssetClass,
        model_name: str = "gemini/gemini-2.0-flash", # Or "gpt-4" if you prefer and have API key setup
        tools: Optional[List[Callable]] = None,
        saved_state_path: Optional[str] = None,
        initial_allocation: float = 100000.0,
        **kwargs
    ):
        self.agent_name = agent_name
        self.asset_class = asset_class
        self.model_name = model_name
        self.tools = tools or []
        self.saved_state_path = saved_state_path
        self.budget_allocation = initial_allocation
        self.portfolio_value = initial_allocation
        self.positions = {}  # Current positions
        self.performance_history = []  # Track performance over time
        self.memory = AgentMemory()
        self.system_prompt = self._get_system_prompt()

        if saved_state_path:
            self.load_state()

    def _get_system_prompt(self) -> str:
        """Generates the system prompt for the analyst agent."""
        return f"""You are a financial analyst specializing in {self.asset_class}.
        Your role is to analyze market data, use provided tools, consult your memory, and make informed investment decisions.
        Respond in JSON format whenever possible.

        You have three types of memory:
        - MMI (Market Information Memory): Basic market data and summaries
        - MIR (Investment Reflection Memory): Records of past actions and reflections
        - MGE (General Experience Memory): Accumulated investment knowledge

        You have access to the following tools:
        - technical_indicator_analysis: Analyze technical indicators
        - market_dynamics_annotation: Annotate market dynamics
        - news_analysis: Analyze financial news
        - risk_assessment: Evaluate potential risks
        - hedging_strategy: Develop hedging strategies
        - stop_loss_calculation: Calculate optimal stop-loss levels
        - trend_analysis: Analyze market trends
        - correlation_analysis: Analyze correlations between assets

        You can take the following actions:
        - BUY: Purchase an asset
        - SELL: Sell an asset
        - HOLD: Maintain current position
        - HEDGE: Implement a hedging strategy
        - REBALANCE: Rebalance portfolio
        - STOP_LOSS: Set stop-loss orders
        - TAKE_PROFIT: Set take-profit orders
        - RISK_ASSESSMENT: Perform detailed risk analysis

        Your goal is to maximize returns while managing risk through effective hedging strategies.
        """

    def run(self, prompt: str) -> str:
        """Simulates running the prompt through an LLM."""
        # In a real implementation, replace this with actual LLM API calls (e.g., OpenAI, Gemini API)
        # Example using OpenAI (requires openai library and API key setup):
        # try:
        #     response = openai.ChatCompletion.create(
        #         model=self.model_name,
        #         messages=[
        #             {"role": "system", "content": self.system_prompt},
        #             {"role": "user", "content": prompt}
        #         ],
        #         temperature=0.7,
        #         max_tokens=1000
        #     )
        #     return response.choices[0].message.content
        # except Exception as e:
        #     logger.error(f"Error calling LLM API: {e}")
        #     return json.dumps({"error": str(e)})

        logger.info(f"[{self.agent_name}] Processing prompt (Simulated LLM): {prompt[:100]}...")

        # Simulate different responses based on the agent type and prompt content (PLACEHOLDER)
        if "decision" in prompt.lower():
            return json.dumps({
                "action": "BUY" if "bitcoin" in prompt.lower() else "HOLD",
                "reason": f"Based on market analysis for {self.asset_class}",
                "confidence": 0.85,
                "hedging_strategy": "Consider options to protect downside"
            })
        elif "report" in prompt.lower():
            return json.dumps({
                "asset_class": self.asset_class,
                "profit_situation": f"{self.portfolio_value - self.budget_allocation:.2f}",
                "budget_expectation": self.budget_allocation * 1.2,
                "reasoning": f"Based on positive outlook for {self.asset_class}"
            })
        elif "analyze" in prompt.lower():
            return json.dumps({
                "trend": "BULLISH" if self.asset_class == AssetClass.BITCOIN else "NEUTRAL",
                "volatility": "HIGH" if self.asset_class == AssetClass.BITCOIN else "MEDIUM",
                "risk_level": "MEDIUM",
                "recommended_action": "BUY"
            })

        # Default response
        return json.dumps({
            "response": f"Analysis for {self.asset_class}",
            "timestamp": datetime.datetime.now().isoformat()
        })

    def analyze_market(self, market_data: MarketData) -> Dict[str, Any]:
        """Analyzes the market using tools and LLM."""
        try:
            tool_results = {}
            for tool in self.tools:
                tool_results[tool.__name__] = tool(market_data)

            # Retrieve relevant memories
            relevant_mmi = self.memory.retrieve_relevant_market_info(market_data.asset_class)
            relevant_mir = self.memory.retrieve_relevant_reflections(market_data.asset_class)

            # Format the memories for inclusion in the prompt
            mmi_str = "\n".join([f"- {m.analysis_summary}" for m in relevant_mmi])
            mir_str = "\n".join([f"- Action: {m.action_taken}, Outcome: {m.outcome}, P/L: {m.profit_loss}" for m in relevant_mir])

            prompt = f"""Analyze the following market data for {market_data.asset_class}:
            Price: {market_data.price}
            Volume: {market_data.volume}
            Volatility: {market_data.volatility}
            Market Sentiment: {market_data.market_sentiment}

            Technical Indicators: {market_data.technical_indicators}

            Recent News:
            {[news['headline'] for news in market_data.news]}

            Tool Analysis Results:
            {json.dumps(tool_results, indent=2)}

            Recent Market Memories:
            {mmi_str}

            Recent Investment Reflections:
            {mir_str}

            Based on this information, provide a comprehensive market analysis.
            Include trend direction, key support/resistance levels, risk assessment,
            and potential catalysts for price movement.
            """

            response = self.run(prompt)
            analysis_result = self._safe_json_loads(response)

            # Store market information in memory
            new_mmi = MarketInfoMemory(
                timestamp=datetime.datetime.now(),
                importance=8.0 if market_data.volatility > 0.5 else 5.0,
                asset_class=market_data.asset_class,
                market_data=market_data.to_dict(),
                analysis_summary=analysis_result.get("summary", str(analysis_result))
            )
            self.memory.add_market_info(new_mmi)

            return analysis_result
        except Exception as e:
            logger.error(f"Error in analyze_market for {self.agent_name}: {e}")
            return {"error": str(e)}

    def make_decision(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Makes an investment decision based on the analysis."""
        try:
            # Retrieve relevant investment reflections
            relevant_reflections = self.memory.retrieve_relevant_reflections(self.asset_class)
            reflection_str = "\n".join([
                f"- Action: {r.action_taken}, Result: {r.outcome}, P/L: {r.profit_loss}, Reasoning: {r.reasoning}"
                for r in relevant_reflections
            ])

            # Get insights from recent conferences
            bac_insights = self.memory.retrieve_latest_conference_insights("BAC")
            esc_insights = self.memory.retrieve_latest_conference_insights("ESC")
            emc_insights = self.memory.retrieve_latest_conference_insights("EMC")

            bac_str = "\n".join([f"- {insight}" for exp in bac_insights for insight in exp.key_insights])
            esc_str = "\n".join([f"- {insight}" for exp in esc_insights for insight in exp.key_insights])
            emc_str = "\n".join([f"- {insight}" for exp in emc_insights for insight in exp.key_insights])

            prompt = f"""Based on the following analysis, make an investment decision for {self.asset_class}:

            Analysis:
            {json.dumps(analysis_result, indent=2)}

            Current Portfolio Value: {self.portfolio_value}
            Current Budget Allocation: {self.budget_allocation}
            Current Positions: {json.dumps(self.positions, indent=2)}

            Past Investment Reflections:
            {reflection_str}

            Recent Budget Conference Insights:
            {bac_str}

            Recent Experience Sharing Insights:
            {esc_str}

            Recent Emergency Conference Insights:
            {emc_str}

            Your task is to decide the most appropriate action (BUY, SELL, HOLD, HEDGE, etc.)
            and provide detailed reasoning, risk assessment, and hedging strategies if applicable.

            Consider both fundamental and technical factors, as well as lessons from past experiences.
            Focus on maximizing returns while implementing effective hedging to protect against downside risk.
            """

            response = self.run(prompt)
            decision = self._safe_json_loads(response)

            # Store the decision reflection in memory
            new_reflection = InvestmentReflectionMemory(
                timestamp=datetime.datetime.now(),
                importance=7.0,
                asset_class=self.asset_class,
                action_taken=decision.get("action", "UNKNOWN"),
                reasoning=decision.get("reasoning", "No reasoning provided"),
                outcome=None,  # To be updated later with actual outcome
                profit_loss=None  # To be updated later
            )
            self.memory.add_investment_reflection(new_reflection)

            # Update positions based on decision (simplified)
            if "action" in decision:
                action = decision["action"]
                if action == "BUY":
                    # Simplified position update
                    self.positions[self.asset_class] = self.positions.get(self.asset_class, 0) + 1
                elif action == "SELL" and self.asset_class in self.positions:
                    if self.positions[self.asset_class] > 0:
                        self.positions[self.asset_class] -= 1

            return decision
        except Exception as e:
            logger.error(f"Error in make_decision for {self.agent_name}: {e}")
            return {"error": str(e)}

    def generate_report(self) -> Dict[str, Any]:
        """Generates a report for the budget allocation conference."""
        try:
            # Get recent market information
            recent_market_info = self.memory.retrieve_relevant_market_info(self.asset_class)
            market_info_str = "\n".join([f"- {info.analysis_summary}" for info in recent_market_info])

            # Get recent reflections
            recent_reflections = self.memory.retrieve_relevant_reflections(self.asset_class)
            reflections_str = "\n".join([
                f"- Action: {ref.action_taken}, Reasoning: {ref.reasoning}"
                for ref in recent_reflections
            ])

            prompt = f"""Generate a concise report for the budget allocation conference.
            Focus on your current performance, market outlook, and budget requirements.

            Asset Class: {self.asset_class}
            Current Portfolio Value: {self.portfolio_value}
            Current Budget Allocation: {self.budget_allocation}
            Current Positions: {json.dumps(self.positions, indent=2)}

            Recent Market Information:
            {market_info_str}

            Recent Investment Decisions:
            {reflections_str}

            Your report should include:
            1. Current profit/loss situation
            2. Market outlook for the next period
            3. Budget expectation with clear reasoning
            4. Proposed investment and hedging strategies
            5. Key risks and mitigation plans
            """

            response = self.run(prompt)
            report = self._safe_json_loads(response)

            return report
        except Exception as e:
            logger.error(f"Error in generate_report for {self.agent_name}: {e}")
            return {"error": str(e)}

    def update_portfolio_value(self, new_value: float, reason: str = "Market movement"):
        """Updates the portfolio value and records performance."""
        old_value = self.portfolio_value
        self.portfolio_value = new_value
        change = new_value - old_value
        change_percent = (change / old_value) * 100 if old_value else 0

        # Record performance
        self.performance_history.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "old_value": old_value,
            "new_value": new_value,
            "change": change,
            "change_percent": change_percent,
            "reason": reason
        })

        logger.info(f"[{self.agent_name}] Portfolio updated: {old_value:.2f} -> {new_value:.2f} ({change_percent:.2f}%) - {reason}")

        # Update relevant reflection outcomes if this is due to a trade
        if "trade" in reason.lower():
            for reflection in reversed(self.memory.mir):
                if reflection.outcome is None:
                    reflection.outcome = "Profitable" if change > 0 else "Loss"
                    reflection.profit_loss = change
                    reflection.importance = min(10.0, reflection.importance + abs(change_percent) / 10)
                    break

    def process_conference_insights(self, conference_type: str, insights: List[str], budget_allocation: Optional[float] = None):
        """Processes insights from a conference and updates memory."""
        new_experience = GeneralExperienceMemory(
            timestamp=datetime.datetime.now(),
            importance=8.0 if conference_type == "EMC" else 6.0,
            conference_type=conference_type,
            key_insights=insights,
            budget_allocation=budget_allocation
        )
        self.memory.add_general_experience(new_experience)

        # Update budget allocation if provided
        if budget_allocation is not None:
            old_allocation = self.budget_allocation
            self.budget_allocation = budget_allocation
            logger.info(f"[{self.agent_name}] Budget allocation updated: {old_allocation:.2f} -> {budget_allocation:.2f}")

    def simulate_hedging_strategy(self, market_data: MarketData) -> Dict[str, Any]:
        """Simulates implementing a hedging strategy based on market conditions."""
        if not self.tools:
            return {"result": "No tools available for hedging strategy"}

        # Here we'd use tools like risk_assessment and hedging_strategy
        tools_to_use = [tool for tool in self.tools if tool.__name__ in ["risk_assessment", "hedging_strategy"]]

        tool_results = {}
        for tool in tools_to_use:
            tool_results[tool.__name__] = tool(market_data)

        prompt = f"""
        Develop a hedging strategy for your {self.asset_class} portfolio to protect against market volatility.

        Current Market Data:
        {market_data.to_dict()}

        Tool Analysis:
        {json.dumps(tool_results, indent=2)}

        Current Positions:
        {json.dumps(self.positions, indent=2)}

        Recommend specific hedging techniques such as:
        - Options strategies (puts, covered calls)
        - Futures contracts
        - Inverse ETFs or positions
        - Stop-loss/take-profit orders
        - Position sizing adjustments

        Provide a complete hedging plan with specific actions, allocations, and timing.
        """

        response = self.run(prompt)
        strategy = self._safe_json_loads(response)

        # Store the hedging strategy in memory
        new_reflection = InvestmentReflectionMemory(
            timestamp=datetime.datetime.now(),
            importance=9.0,  # Hedging strategies are high importance
            asset_class=self.asset_class,
            action_taken="HEDGE",
            reasoning=strategy.get("reasoning", "Hedging due to market volatility"),
            outcome=None,  # To be updated later
            profit_loss=None  # To be updated later
        )
        self.memory.add_investment_reflection(new_reflection)

        return strategy

    def _safe_json_loads(self, json_str: str) -> Dict:
        """Safely loads JSON, handling potential errors and variations."""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON: {json_str[:200]}..., attempting to extract.")
            try:
                # Try to extract a JSON object using regex
                import re
                match = re.search(r"\{.*\}", json_str, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
                else:
                    return {"error": f"Invalid JSON format", "raw_response": json_str[:200]}
            except Exception:
                return {"error": f"Invalid JSON format", "raw_response": json_str[:200]}

    def save_state(self):
        """Saves the agent's state to a JSON file."""
        if not self.saved_state_path:
            logger.warning(f"No saved_state_path provided for {self.agent_name}, state not saved")
            return

        try:
            filepath = Path(self.saved_state_path)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            state = {
                "agent_name": self.agent_name,
                "asset_class": self.asset_class,
                "budget_allocation": self.budget_allocation,
                "portfolio_value": self.portfolio_value,
                "positions": self.positions,
                "performance_history": self.performance_history,
                "memory": self.memory.to_dict()
            }

            with open(filepath, "w") as f:
                json.dump(state, f, indent=2)

            logger.info(f"Saved state for {self.agent_name} to {filepath}")
        except Exception as e:
            logger.error(f"Error saving state for {self.agent_name}: {e}")

    def load_state(self):
        """Loads the agent's state from a JSON file."""
        if not self.saved_state_path:
            logger.warning(f"No saved_state_path provided for {self.agent_name}, no state loaded")
            return

        try:
            filepath = Path(self.saved_state_path)
            if filepath.exists():
                with open(filepath, "r") as f:
                    state = json.load(f)

                self.agent_name = state.get("agent_name", self.agent_name)
                self.asset_class = state.get("asset_class", self.asset_class)
                self.budget_allocation = state.get("budget_allocation", self.budget_allocation)
                self.portfolio_value = state.get("portfolio_value", self.portfolio_value)
                self.positions = state.get("positions", self.positions)
                self.performance_history = state.get("performance_history", self.performance_history)

                if "memory" in state:
                    self.memory = AgentMemory.from_dict(state["memory"])

                logger.info(f"Loaded state for {self.agent_name} from {filepath}")
        except Exception as e:
            logger.error(f"Error loading state for {self.agent_name}: {e}")


class BitcoinAnalyst(AnalystAgent):
    """Specialized analyst for Bitcoin."""
    def __init__(
        self,
        model_name: str = "gemini/gemini-2.0-flash", # Or "gpt-4"
        tools: Optional[List[Callable]] = None,
        saved_state_path: Optional[str] = None,
        initial_allocation: float = 100000.0
    ):
        super().__init__(
            agent_name="Bitcoin Analyst",
            asset_class=AssetClass.BITCOIN,
            model_name=model_name,
            tools=tools,
            saved_state_path=saved_state_path,
            initial_allocation=initial_allocation
        )

    def _get_system_prompt(self) -> str:
        """Customized system prompt for Bitcoin analyst."""
        base_prompt = super()._get_system_prompt()
        return base_prompt + """
        As a Bitcoin analyst, pay special attention to:
        - On-chain metrics and network activity
        - Institutional adoption and regulatory developments
        - Technical patterns specific to crypto markets
        - Mining difficulty and hash rate trends
        - Bitcoin dominance and correlation with altcoins

        Your goal is to identify optimal entry and exit points while maintaining effective
        hedging strategies against Bitcoin's high volatility.
        """


class StocksAnalyst(AnalystAgent):
    """Specialized analyst for Stocks."""
    def __init__(
        self,
        model_name: str = "gemini/gemini-2.0-flash", # Or "gpt-4"
        tools: Optional[List[Callable]] = None,
        saved_state_path: Optional[str] = None,
        initial_allocation: float = 100000.0
    ):
        super().__init__(
            agent_name="Stocks Analyst",
            asset_class=AssetClass.STOCKS,
            model_name=model_name,
            tools=tools,
            saved_state_path=saved_state_path,
            initial_allocation=initial_allocation
        )

    def _get_system_prompt(self) -> str:
        """Customized system prompt for Stocks analyst."""
        base_prompt = super()._get_system_prompt()
        return base_prompt + """
        As a Stocks analyst, focus on:
        - Sector rotations and industry trends
        - Earnings reports and corporate guidance
        - Macroeconomic indicators and Fed policy
        - Market breadth and sentiment indicators
        - Options activity and institutional positioning

        Your goal is to identify undervalued opportunities while maintaining proper
        sector diversification and implementing tactical hedging strategies.
        """

    def analyze_sector_rotation(self, sector_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyzes sector rotation trends to guide investment decisions."""
        prompt = f"""
        Analyze the following sector performance data:
        {json.dumps(sector_data, indent=2)}

        Identify:
        1. Leading and lagging sectors
        2. Sectors showing momentum shift
        3. Defensive vs cyclical rotation patterns
        4. Correlation to broader economic indicators
        5. Recommended sector allocations

        Provide actionable insights for portfolio positioning.
        """

        response = self.run(prompt)
        return self._safe_json_loads(response)


class ForexAnalyst(AnalystAgent):
    """Specialized analyst for Forex markets."""
    def __init__(
        self,
        model_name: str = "gemini/gemini-2.0-flash", # Or "gpt-4"
        tools: Optional[List[Callable]] = None,
        saved_state_path: Optional[str] = None,
        initial_allocation: float = 100000.0
    ):
        super().__init__(
            agent_name="Forex Analyst",
            asset_class=AssetClass.FOREX,
            model_name=model_name,
            tools=tools,
            saved_state_path=saved_state_path,
            initial_allocation=initial_allocation
        )

    def _get_system_prompt(self) -> str:
        """Customized system prompt for Forex analyst."""
        base_prompt = super()._get_system_prompt()
        return base_prompt + """
        As a Forex analyst, focus on:
        - Interest rate differentials and central bank policies
        - Economic data releases and their impact on currency pairs
        - Geopolitical events affecting currency valuations
        - Technical patterns specific to forex markets
        - Carry trade opportunities and funding currency dynamics

        Your goal is to identify high-probability currency trades while implementing
        robust risk management and hedging strategies to protect against volatile moves.
        """

    def analyze_central_bank_policy(self, policy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyzes central bank policy developments and their impact on forex markets."""
        prompt = f"""
        Analyze the following central bank policy developments:
        {json.dumps(policy_data, indent=2)}

        Assess:
        1. Expected interest rate path for major central banks
        2. Forward guidance and policy language changes
        3. Balance sheet operations and quantitative policies
        4. Impact on respective currencies and carry trades
        5. Potential currency pair trades based on policy divergence

        Provide a comprehensive analysis with risk-adjusted trade recommendations.
        """

        response = self.run(prompt)
        return self._safe_json_loads(response)


class HedgeFundManager:
    """Manages a team of specialized analyst agents and coordinates their activities."""
    def __init__(
        self,
        model_name: str = "gemini/gemini-2.0-flash", # Or "gpt-4"
        bitcoin_allocation: float = 50000.0,
        stocks_allocation: float = 100000.0,
        forex_allocation: float = 50000.0,
        saved_state_path: Optional[str] = None
    ):
        self.model_name = model_name
        self.saved_state_path = saved_state_path
        self.base_path = Path(saved_state_path).parent if saved_state_path else Path("./agent_states") # Changed default path
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Initialize analysts
        self.bitcoin_analyst = BitcoinAnalyst(
            model_name=model_name,
            tools=self._get_bitcoin_tools(),
            saved_state_path=str(self.base_path / "bitcoin_analyst.json"),
            initial_allocation=bitcoin_allocation
        )

        self.stocks_analyst = StocksAnalyst(
            model_name=model_name,
            tools=self._get_stocks_tools(),
            saved_state_path=str(self.base_path / "stocks_analyst.json"),
            initial_allocation=stocks_allocation
        )

        self.forex_analyst = ForexAnalyst(
            model_name=model_name,
            tools=self._get_forex_tools(),
            saved_state_path=str(self.base_path / "forex_analyst.json"),
            initial_allocation=forex_allocation
        )

        self.analysts = [self.bitcoin_analyst, self.stocks_analyst, self.forex_analyst]
        self.total_portfolio_value = sum(analyst.portfolio_value for analyst in self.analysts)
        self.market_condition = MarketCondition.NORMAL
        self.conference_history = []

        # Load state if available
        if saved_state_path:
            self.load_state()

    def _get_bitcoin_tools(self) -> List[Callable]:
        """Returns tools for bitcoin analysis."""
        # Placeholder tool functions - replace with actual implementations
        def technical_indicator_analysis_btc(market_data: MarketData):
            # Simulate technical indicator analysis for Bitcoin
            return {"RSI": random.uniform(30, 70), "MACD": random.uniform(-1, 1)}

        def news_analysis_btc(market_data: MarketData):
            # Simulate news sentiment analysis for Bitcoin
            return {"sentiment_score": random.uniform(-0.5, 0.8), "keywords": ["crypto", "regulation"]}

        return [technical_indicator_analysis_btc, news_analysis_btc]

    def _get_stocks_tools(self) -> List[Callable]:
        """Returns tools for stock analysis."""
        # Placeholder tool functions - replace with actual implementations
        def technical_indicator_analysis_stocks(market_data: MarketData):
            # Simulate technical indicator analysis for Stocks
            return {"SMA_50": random.uniform(100, 150), "EMA_20": random.uniform(120, 160)}

        def market_dynamics_annotation_stocks(market_data: MarketData):
            # Simulate market dynamics annotation for Stocks
            return {"volume_change": random.uniform(-10, 20), "volatility_index": random.uniform(15, 25)}

        return [technical_indicator_analysis_stocks, market_dynamics_annotation_stocks]

    def _get_forex_tools(self) -> List[Callable]:
        """Returns tools for forex analysis."""
        # Placeholder tool functions - replace with actual implementations
        def risk_assessment_forex(market_data: MarketData):
            # Simulate risk assessment for Forex
            return {"risk_score": random.uniform(0.3, 0.7), "currency_pair": market_data.asset_class}

        def hedging_strategy_forex(market_data: MarketData):
            # Simulate hedging strategy recommendation for Forex
            return {"strategy_type": "Carry Trade", "recommendation": "Consider hedging with options"}

        return [risk_assessment_forex, hedging_strategy_forex]


    def assess_market_condition(self, market_data: Dict[str, MarketData]) -> MarketCondition:
        """Assesses overall market conditions to determine if special conferences are needed."""
        try:
            # Extract volatility measures
            volatilities = [data.volatility for data in market_data.values()]
            avg_volatility = sum(volatilities) / len(volatilities) if volatilities else 0

            # Extract sentiment measures
            sentiments = [data.market_sentiment for data in market_data.values()]
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

            # Determine market condition
            if any(v > 0.8 for v in volatilities):
                return MarketCondition.EXTREME_VOLATILITY
            elif avg_volatility > 0.6:
                return MarketCondition.VOLATILE
            elif avg_sentiment > 0.7:
                return MarketCondition.BULLISH
            elif avg_sentiment < -0.7:
                return MarketCondition.BEARISH
            else:
                return MarketCondition.NORMAL

        except Exception as e:
            logger.error(f"Error assessing market condition: {e}")
            return MarketCondition.NORMAL

    def run_budget_allocation_conference(self) -> Dict[str, Any]:
        """Runs a budget allocation conference (BAC) with all analysts."""
        try:
            logger.info("Starting Budget Allocation Conference (BAC)")

            # Get reports from all analysts
            reports = {}
            for analyst in self.analysts:
                reports[analyst.asset_class] = analyst.generate_report()

            # Calculate current portfolio values
            self.total_portfolio_value = sum(analyst.portfolio_value for analyst in self.analysts)
            allocations = {analyst.asset_class: analyst.portfolio_value for analyst in self.analysts}

            # Create the budget allocation prompt
            prompt = f"""
            You are the chairperson of a Budget Allocation Conference (BAC) for a hedge fund.
            Review the reports from each asset specialist and decide on budget allocations:

            Current Total Portfolio Value: {self.total_portfolio_value}
            Current Market Condition: {self.market_condition.value}

            Analyst Reports:
            {json.dumps(reports, indent=2)}

            Current Allocations:
            {json.dumps(allocations, indent=2)}

            Make allocation decisions based on:
            1. Recent performance and ROI
            2. Market outlook and opportunities
            3. Risk management and diversification
            4. Current market conditions ({self.market_condition.value})

            Return the new budget allocations as percentages and absolute values.
            Provide clear reasoning for each allocation decision.

            The sum of allocations must equal the total portfolio value.
            """

            # Simulate running the prompt through LLM
            # In real implementation, use proper API call
            response = self._simulate_conference_llm(prompt)
            result = self._safe_json_loads(response)

            # Update budget allocations based on conference
            if "allocations" in result:
                for asset_class, allocation in result["allocations"].items():
                    if isinstance(allocation, dict) and "value" in allocation:
                        value = allocation["value"]
                    elif isinstance(allocation, (int, float)):
                        value = allocation
                    else:
                        continue

                    for analyst in self.analysts:
                        if analyst.asset_class == asset_class:
                            # Process conference insights for each analyst
                            insights = result.get("reasoning", {}).get(asset_class, [])
                            if isinstance(insights, str):
                                insights = [insights]
                            elif not isinstance(insights, list):
                                insights = []

                            analyst.process_conference_insights("BAC", insights, value)

            # Record conference
            self.conference_history.append({
                "type": "BAC",
                "timestamp": datetime.datetime.now().isoformat(),
                "market_condition": self.market_condition.value,
                "result": result
            })

            # Save state after conference
            self.save_state()

            logger.info("Budget Allocation Conference (BAC) completed")
            return result
        except Exception as e:
            logger.error(f"Error during budget allocation conference: {e}")
            return {"error": str(e), "status": "failed"}

    def run_experience_sharing_conference(self) -> Dict[str, Any]:
        """Runs an experience sharing conference (ESC) with all analysts."""
        try:
            logger.info("Starting Experience Sharing Conference (ESC)")

            # Gather recent experiences from each analyst
            experiences = {}
            for analyst in self.analysts:
                asset_class = analyst.asset_class
                reflections = analyst.memory.retrieve_relevant_reflections(asset_class, limit=5)
                experiences[asset_class] = [
                    {
                        "action": r.action_taken,
                        "reasoning": r.reasoning,
                        "outcome": r.outcome,
                        "profit_loss": r.profit_loss,
                        "timestamp": r.timestamp.isoformat()
                    }
                    for r in reflections
                ]

            prompt = f"""
            You are facilitating an Experience Sharing Conference (ESC) for a hedge fund.
            Analysts have shared their recent trading experiences and decisions.

            Market Condition: {self.market_condition.value}

            Analyst Experiences:
            {json.dumps(experiences, indent=2)}

            Your task is to:
            1. Identify successful patterns and strategies from each asset class
            2. Extract lessons from unsuccessful trades
            3. Find correlations between assets and market conditions
            4. Recommend best practices and risk management techniques
            5. Synthesize key insights that each analyst can apply

            Structure your response with specific insights for each analyst.
            Focus on actionable, evidence-based recommendations.
            """

            response = self._simulate_conference_llm(prompt)
            result = self._safe_json_loads(response)

            # Process insights for each analyst
            for analyst in self.analysts:
                asset_insights = result.get(str(analyst.asset_class), []) or result.get("insights", {}).get(str(analyst.asset_class), [])

                # Format insights properly
                if isinstance(asset_insights, str):
                    asset_insights = [asset_insights]
                elif isinstance(asset_insights, dict):
                    asset_insights = [f"{k}: {v}" for k, v in asset_insights.items()]
                elif not isinstance(asset_insights, list):
                    asset_insights = []

                # Process insights
                analyst.process_conference_insights("ESC", asset_insights)

            # Record conference
            self.conference_history.append({
                "type": "ESC",
                "timestamp": datetime.datetime.now().isoformat(),
                "market_condition": self.market_condition.value,
                "result": result
            })

            # Save state after conference
            self.save_state()

            logger.info("Experience Sharing Conference (ESC) completed")
            return result
        except Exception as e:
            logger.error(f"Error during experience sharing conference: {e}")
            return {"error": str(e), "status": "failed"}

    def run_emergency_market_conference(self, trigger_reason: str) -> Dict[str, Any]:
        """Runs an emergency market conference (EMC) for unusual market conditions."""
        try:
            logger.info(f"Starting Emergency Market Conference (EMC) - Trigger: {trigger_reason}")

            # Get current positions and analyses from all analysts
            positions = {}
            analyses = {}
            for analyst in self.analysts:
                asset_class = analyst.asset_class
                positions[asset_class] = analyst.positions

                # Get most recent market info memory for analysis
                recent_info = analyst.memory.retrieve_relevant_market_info(asset_class, limit=1)
                if recent_info:
                    analyses[asset_class] = recent_info[0].analysis_summary
                else:
                    analyses[asset_class] = "No recent analysis available"

            prompt = f"""
            You are leading an Emergency Market Conference (EMC) due to unusual market conditions.

            Trigger Reason: {trigger_reason}
            Market Condition: {self.market_condition.value}

            Current Positions:
            {json.dumps(positions, indent=2)}

            Recent Analyses:
            {json.dumps(analyses, indent=2)}

            Your task is to develop an immediate action plan:
            1. Assess the severity and potential impact of the current situation
            2. Recommend immediate hedging strategies for each asset class
            3. Identify potential opportunities created by the market disruption
            4. Determine if any emergency rebalancing is required
            5. Develop a communication plan and monitoring framework

            Provide specific instructions for each analyst with a focus on risk mitigation.
            Include both defensive and opportunistic strategies where appropriate.
            """

            response = self._simulate_conference_llm(prompt)
            result = self._safe_json_loads(response)

            # Process emergency insights for each analyst
            for analyst in self.analysts:
                # Extract relevant action items and insights
                asset_insights = result.get(str(analyst.asset_class), []) or result.get("actions", {}).get(str(analyst.asset_class), [])

                # Format insights properly
                if isinstance(asset_insights, str):
                    asset_insights = [asset_insights]
                elif isinstance(asset_insights, dict):
                    asset_insights = [f"{k}: {v}" for k, v in asset_insights.items()]
                elif not isinstance(asset_insights, list):
                    asset_insights = []

                # Additional insights from general recommendations
                general_insights = result.get("general_recommendations", [])
                if isinstance(general_insights, str):
                    general_insights = [general_insights]
                elif isinstance(general_insights, dict):
                    general_insights = [f"{k}: {v}" for k, v in general_insights.items()]
                elif not isinstance(general_insights, list):
                    general_insights = []

                # Combine insights
                all_insights = asset_insights + general_insights

                # Process emergency insights with high importance
                analyst.process_conference_insights("EMC", all_insights)

            # Record emergency conference
            self.conference_history.append({
                "type": "EMC",
                "timestamp": datetime.datetime.now().isoformat(),
                "market_condition": self.market_condition.value,
                "trigger_reason": trigger_reason,
                "result": result
            })

            # Save state after emergency conference
            self.save_state()

            logger.info("Emergency Market Conference (EMC) completed")
            return result
        except Exception as e:
            logger.error(f"Error during emergency market conference: {e}")
            return {"error": str(e), "status": "failed"}

    def process_market_update(self, market_data: Dict[AssetClass, MarketData]) -> Dict[str, Any]:
        """Processes market updates for all asset classes."""
        try:
            logger.info("Processing market updates for all asset classes")

            # Check if market condition has changed
            new_market_condition = self.assess_market_condition(market_data)
            condition_changed = new_market_condition != self.market_condition

            if condition_changed:
                logger.info(f"Market condition changed: {self.market_condition.value} -> {new_market_condition.value}")
                self.market_condition = new_market_condition

            # Process market data for each analyst
            results = {}
            for analyst in self.analysts:
                if analyst.asset_class in market_data:
                    asset_data = market_data[analyst.asset_class]
                    analysis = analyst.analyze_market(asset_data)
                    decision = analyst.make_decision(analysis)

                    results[analyst.asset_class] = {
                        "analysis": analysis,
                        "decision": decision
                    }
                    # Simulate portfolio value update based on decision (very basic)
                    if "action" in decision and decision["action"] == "BUY":
                        analyst.update_portfolio_value(analyst.portfolio_value * 1.01, reason="Trade: BUY") # 1% increase
                    elif "action" in decision and decision["action"] == "SELL":
                        analyst.update_portfolio_value(analyst.portfolio_value * 0.99, reason="Trade: SELL") # 1% decrease


            # Update total portfolio value
            self.total_portfolio_value = sum(analyst.portfolio_value for analyst in self.analysts)

            # Check if we need to trigger emergency conference
            if self.market_condition in [MarketCondition.EXTREME_VOLATILITY, MarketCondition.BLACK_SWAN]:
                self.run_emergency_market_conference(f"Detected {self.market_condition.value} market condition")

            # Save state after processing
            self.save_state()

            return {
                "results": results,
                "market_condition": self.market_condition.value,
                "total_portfolio_value": self.total_portfolio_value,
                "emergency_triggered": self.market_condition in [MarketCondition.EXTREME_VOLATILITY, MarketCondition.BLACK_SWAN]
            }
        except Exception as e:
            logger.error(f"Error during market update processing: {e}")
            return {"error": str(e), "status": "failed"}

    def _simulate_conference_llm(self, prompt: str) -> str:
        """Simulates running the conference prompt through an LLM."""
        # In a real implementation, replace this with actual LLM API calls.
        # This is a placeholder for demonstration.
        logger.info(f"Processing conference prompt (Simulated LLM): {prompt[:100]}...")

        if "Budget Allocation Conference" in prompt:
            return json.dumps({
                "allocations": {
                    AssetClass.BITCOIN: {"value": 60000, "percentage": 30},
                    AssetClass.STOCKS: {"value": 100000, "percentage": 50},
                    AssetClass.FOREX: {"value": 40000, "percentage": 20}
                },
                "reasoning": {
                    AssetClass.BITCOIN: ["Bullish momentum increasing", "Institutional adoption growing"],
                    AssetClass.STOCKS: ["Stable performance", "Diversified exposure needed"],
                    AssetClass.FOREX: ["Hedging opportunity", "Interest rate divergence trading"]
                },
                "summary": "Increased Bitcoin allocation due to bullish momentum, maintained stocks for stability."
            })
        elif "Experience Sharing Conference" in prompt:
            return json.dumps({
                "insights": {
                    AssetClass.BITCOIN: ["Momentum trading effective", "Stop losses critical for volatility"],
                    AssetClass.STOCKS: ["Sector rotation strategy working well", "Earnings season requires caution"],
                    AssetClass.FOREX: ["Central bank policy divergence profitable", "Risk management improvements needed"]
                },
                "common_patterns": ["Risk management critical across assets", "Correlation between BTC and tech stocks increasing"],
                "recommendations": ["Implement cross-asset hedging", "Improve exit strategy timing"]
            })
        elif "Emergency Market Conference" in prompt:
            return json.dumps({
                "severity": "High",
                "actions": {
                    AssetClass.BITCOIN: ["Implement 15% hedge via options", "Set stop losses at -5%"],
                    AssetClass.STOCKS: ["Rotate to defensive sectors", "Increase cash position to 20%"],
                    AssetClass.FOREX: ["Close high-risk positions", "Shift to safe-haven currencies"]
                },
                "general_recommendations": ["Increase communication frequency", "Monitor liquidity indicators"],
                "monitoring_framework": ["Daily volatility checks", "Increased reporting schedule"]
            })

        # Default generic response
        return json.dumps({
            "result": "Conference completed",
            "recommendations": ["Generic recommendation 1", "Generic recommendation 2"],
            "summary": "Conference summary would appear here"
        })

    def _safe_json_loads(self, json_str: str) -> Dict:
        """Safely loads JSON, handling potential errors and variations."""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON: {json_str[:200]}..., attempting to extract.")
            try:
                # Try to extract a JSON object using regex
                import re
                match = re.search(r"\{.*\}", json_str, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
                else:
                    return {"error": f"Invalid JSON format", "raw_response": json_str[:200]}
            except Exception:
                return {"error": f"Invalid JSON format", "raw_response": json_str[:200]}

    def save_state(self):
        """Saves the manager's state to a JSON file."""
        if not self.saved_state_path:
            logger.warning("No saved_state_path provided for HedgeFundManager, state not saved")
            return

        try:
            filepath = Path(self.saved_state_path)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Save individual analyst states
            for analyst in self.analysts:
                analyst.save_state()

            # Save manager state
            state = {
                "total_portfolio_value": self.total_portfolio_value,
                "market_condition": self.market_condition.value,
                "conference_history": self.conference_history
            }

            with open(filepath, "w") as f:
                json.dump(state, f, indent=2)

            logger.info(f"Saved state for HedgeFundManager to {filepath}")
        except Exception as e:
            logger.error(f"Error saving state for HedgeFundManager: {e}")

    def load_state(self):
        """Loads the manager's state from a JSON file."""
        if not self.saved_state_path:
            logger.warning("No saved_state_path provided for HedgeFundManager, no state loaded")
            return

        try:
            filepath = Path(self.saved_state_path)
            if filepath.exists():
                with open(filepath, "r") as f:
                    state = json.load(f)

                self.total_portfolio_value = state.get("total_portfolio_value", self.total_portfolio_value)

                if "market_condition" in state:
                    try:
                        self.market_condition = MarketCondition(state["market_condition"])
                    except ValueError:
                        self.market_condition = MarketCondition.NORMAL

                self.conference_history = state.get("conference_history", [])

                # Load individual analyst states (they handle their own loading)
                for analyst in self.analysts:
                    analyst.load_state()

                logger.info(f"Loaded state for HedgeFundManager from {filepath}")
        except Exception as e:
            logger.error(f"Error loading state for HedgeFundManager: {e}")


# --- Example Usage ---
if __name__ == "__main__":
    try:
        # Create HedgeFundManager instance
        manager = HedgeFundManager(
            model_name="gemini/gemini-2.0-flash", # Or "gpt-4"
            saved_state_path="./agent_states/hedge_fund_manager.json",
            bitcoin_allocation=70000.0,
            stocks_allocation=120000.0,
            forex_allocation=60000.0
        )
        # Load saved state if available
        manager.load_state()

        # Simulate market data updates
        market_data_updates = {
            AssetClass.BITCOIN: MarketData(
                asset_class=AssetClass.BITCOIN,
                price=62000.0,
                timestamp=datetime.datetime.now(),
                volatility=0.75,
                market_sentiment=0.8,
                news=[{"headline": "Bitcoin price surges after ETF approval"}]
            ),
            AssetClass.STOCKS: MarketData(
                asset_class=AssetClass.STOCKS,
                price=4600.0,
                timestamp=datetime.datetime.now(),
                volatility=0.5,
                market_sentiment=0.6,
                news=[{"headline": "Tech sector leads market rally"}]
            ),
            AssetClass.FOREX: MarketData(
                asset_class=AssetClass.FOREX,
                price=1.19,
                timestamp=datetime.datetime.now(),
                volatility=0.6,
                market_sentiment=0.2,
                news=[{"headline": "Dollar strengthens on positive economic data"}]
            )
        }

        # Process market update and run trading cycle
        update_result = manager.process_market_update(market_data_updates)
        logger.info(f"Market Update Result:\n{json.dumps(update_result, indent=2)}")

        # Run conferences periodically (e.g., monthly BAC, weekly ESC)
        manager.run_budget_allocation_conference()
        manager.run_experience_sharing_conference()

        # Simulate extreme market event triggering EMC
        if manager.market_condition == MarketCondition.EXTREME_VOLATILITY:
            manager.run_emergency_market_conference(trigger_reason="Extreme Volatility Detected")

        # Save state after simulation
        manager.save_state()
        print(f"Current Total Portfolio Value: ${manager.total_portfolio_value:.2f}")


    except Exception as e:
        logger.error(f"Example execution failed: {e}")
