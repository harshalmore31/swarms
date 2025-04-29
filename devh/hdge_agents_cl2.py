import json
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import datetime

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

@dataclass
class MemoryEntry:
    """Base class for memory entries with common fields."""
    timestamp: datetime.datetime
    importance: float = 1.0  # Scale of 1-10 for retrieval prioritization
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Creates a memory entry from a dictionary."""
        data_copy = data.copy()
        data_copy["timestamp"] = datetime.datetime.fromisoformat(data_copy["timestamp"])
        return cls(**data_copy)

@dataclass
class MarketInfoMemory(MemoryEntry):
    """Represents a Market Information Memory (MMI) entry."""
    asset_class: AssetClass
    market_data: Dict[str, Any]
    analysis_summary: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts to dictionary for serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            "asset_class": self.asset_class,
            "market_data": self.market_data,
            "analysis_summary": self.analysis_summary
        })
        return base_dict

@dataclass
class InvestmentReflectionMemory(MemoryEntry):
    """Represents an Investment Reflection Memory (MIR) entry."""
    asset_class: AssetClass
    action_taken: Action
    reasoning: str
    outcome: Optional[str] = None
    profit_loss: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts to dictionary for serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            "asset_class": self.asset_class,
            "action_taken": self.action_taken,
            "reasoning": self.reasoning,
            "outcome": self.outcome,
            "profit_loss": self.profit_loss
        })
        return base_dict

@dataclass
class GeneralExperienceMemory(MemoryEntry):
    """Represents a General Experience Memory (MGE) entry."""
    conference_type: str  # BAC, ESC, EMC
    key_insights: List[str]
    budget_allocation: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts to dictionary for serialization."""
        base_dict = super().to_dict()
        base_dict.update({
            "conference_type": self.conference_type,
            "key_insights": self.key_insights,
            "budget_allocation": self.budget_allocation
        })
        return base_dict

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
        model_name: str = "gemini/gemini-2.0-flash",
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
        # In a real implementation, this would call the LLM API
        # For demonstration purposes, we'll return a placeholder response
        logger.info(f"[{self.agent_name}] Processing prompt: {prompt[:100]}...")
        
        # Simulate different responses based on the agent type and prompt content
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
        model_name: str = "gpt-4", 
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
        model_name: str = "gpt-4", 
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
        - Market