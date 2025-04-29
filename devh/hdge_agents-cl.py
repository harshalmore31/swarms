import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import datetime

from swarms import Agent
from swarms.structs.conversation import Conversation

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AssetClass(str, Enum):
    """Defines the asset classes that agents can specialize in."""
    BITCOIN = "Bitcoin"
    STOCKS = "Stocks"
    FOREX = "Forex"

class ActionType(str, Enum):
    """Defines the possible actions agents can take."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    HEDGE = "HEDGE"
    DIVERSIFY = "DIVERSIFY"
    REBALANCE = "REBALANCE"
    RISK_ASSESSMENT = "RISK_ASSESSMENT"
    STOP_LOSS = "STOP_LOSS"

@dataclass
class MarketData:
    """Represents market data for a specific asset class."""
    asset_class: AssetClass
    price: float
    change_percent: float
    volume: float
    timestamp: datetime.datetime
    volatility: float
    news: List[Dict[str, str]]
    technical_indicators: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts market data to dictionary format."""
        return {
            "asset_class": self.asset_class,
            "price": self.price,
            "change_percent": self.change_percent,
            "volume": self.volume,
            "timestamp": self.timestamp.isoformat(),
            "volatility": self.volatility,
            "news": self.news,
            "technical_indicators": self.technical_indicators
        }

@dataclass
class AgentMemory:
    """Represents an agent's memory with three distinct types."""
    # Market Information Memory - stores market data points and basic analysis
    mmi: List[Dict[str, Any]] = field(default_factory=list)
    
    # Investment Reflection Memory - stores past decisions and their outcomes
    mir: List[Dict[str, Any]] = field(default_factory=list)
    
    # General Experience Memory - stores knowledge from conferences and insights
    mge: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the memory to a dictionary for serialization."""
        return {
            "mmi": self.mmi,
            "mir": self.mir,
            "mge": self.mge
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMemory':
        """Creates a Memory instance from a dictionary."""
        return cls(**data)

    def add_market_info(self, info: Dict[str, Any]) -> None:
        """Adds a new market information entry to MMI."""
        timestamp = datetime.datetime.now().isoformat()
        entry = {**info, "timestamp": timestamp}
        self.mmi.append(entry)
        # Keep MMI size manageable by removing oldest entries if too large
        if len(self.mmi) > 100:
            self.mmi.pop(0)

    def add_investment_reflection(self, reflection: Dict[str, Any]) -> None:
        """Adds a new investment reflection to MIR."""
        timestamp = datetime.datetime.now().isoformat()
        entry = {**reflection, "timestamp": timestamp}
        self.mir.append(entry)
        # Keep MIR size manageable
        if len(self.mir) > 50:
            self.mir.pop(0)

    def add_general_experience(self, experience: Dict[str, Any]) -> None:
        """Adds a new general experience entry to MGE."""
        timestamp = datetime.datetime.now().isoformat()
        entry = {**experience, "timestamp": timestamp}
        self.mge.append(entry)
        # MGE can grow larger as it represents accumulated knowledge

    def get_recent_market_info(self, count: int = 5) -> List[Dict[str, Any]]:
        """Returns the most recent market information entries."""
        return self.mmi[-count:] if self.mmi else []

    def get_relevant_reflections(self, query: str, count: int = 3) -> List[Dict[str, Any]]:
        """Returns relevant investment reflections based on a query."""
        # In a real implementation, this would use a vector database or other retrieval method
        # For simplicity, we just return the most recent entries
        return self.mir[-count:] if self.mir else []

    def get_relevant_experiences(self, query: str, count: int = 3) -> List[Dict[str, Any]]:
        """Returns relevant general experiences based on a query."""
        # In a real implementation, this would use a vector database or other retrieval method
        # For simplicity, we just return the most recent entries
        return self.mge[-count:] if self.mge else []

class PortfolioPosition:
    """Represents a position in the portfolio."""
    def __init__(
        self,
        asset_class: AssetClass,
        entry_price: float,
        quantity: float,
        entry_timestamp: datetime.datetime,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ):
        self.asset_class = asset_class
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_timestamp = entry_timestamp
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.hedge_positions: List[Dict[str, Any]] = []
    
    def add_hedge(
        self,
        hedge_type: str,
        quantity: float,
        price: float
    ) -> None:
        """Adds a hedge position to protect this position."""
        self.hedge_positions.append({
            "hedge_type": hedge_type,
            "quantity": quantity,
            "price": price,
            "timestamp": datetime.datetime.now()
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the position to a dictionary."""
        return {
            "asset_class": self.asset_class,
            "entry_price": self.entry_price,
            "quantity": self.quantity,
            "entry_timestamp": self.entry_timestamp.isoformat(),
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "hedge_positions": self.hedge_positions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PortfolioPosition':
        """Creates a Position instance from a dictionary."""
        entry_timestamp = datetime.datetime.fromisoformat(data.pop("entry_timestamp"))
        hedge_positions = data.pop("hedge_positions", [])
        position = cls(entry_timestamp=entry_timestamp, **data)
        position.hedge_positions = hedge_positions
        return position

class Portfolio:
    """Represents the investment portfolio with positions and performance tracking."""
    def __init__(self, initial_cash: float = 1000000.0):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions: Dict[str, PortfolioPosition] = {}
        self.trade_history: List[Dict[str, Any]] = []
        self.performance_history: List[Dict[str, Any]] = []
        self.record_performance()
    
    def execute_trade(
        self,
        asset_class: AssetClass,
        action: ActionType,
        quantity: float,
        price: float,
        reason: str
    ) -> Dict[str, Any]:
        """Executes a trade and updates the portfolio."""
        timestamp = datetime.datetime.now()
        position_key = asset_class.value
        
        # Record the trade
        trade = {
            "asset_class": asset_class,
            "action": action,
            "quantity": quantity,
            "price": price,
            "timestamp": timestamp.isoformat(),
            "reason": reason
        }
        self.trade_history.append(trade)
        
        # Update positions and cash
        if action == ActionType.BUY:
            cost = price * quantity
            if cost > self.cash:
                logger.warning(f"Insufficient funds for trade: {trade}")
                return {"success": False, "error": "Insufficient funds"}
            
            self.cash -= cost
            
            if position_key in self.positions:
                # Update existing position
                existing_pos = self.positions[position_key]
                total_value = (existing_pos.entry_price * existing_pos.quantity) + cost
                total_quantity = existing_pos.quantity + quantity
                # Calculate new average entry price
                new_entry_price = total_value / total_quantity
                
                self.positions[position_key] = PortfolioPosition(
                    asset_class=asset_class,
                    entry_price=new_entry_price,
                    quantity=total_quantity,
                    entry_timestamp=existing_pos.entry_timestamp,
                    stop_loss=existing_pos.stop_loss,
                    take_profit=existing_pos.take_profit
                )
            else:
                # Create new position
                self.positions[position_key] = PortfolioPosition(
                    asset_class=asset_class,
                    entry_price=price,
                    quantity=quantity,
                    entry_timestamp=timestamp
                )
        
        elif action == ActionType.SELL:
            if position_key not in self.positions:
                logger.warning(f"No position found for {asset_class} to sell")
                return {"success": False, "error": "No position found"}
            
            existing_pos = self.positions[position_key]
            if existing_pos.quantity < quantity:
                logger.warning(f"Insufficient quantity for sale: {trade}")
                return {"success": False, "error": "Insufficient quantity"}
            
            # Update cash
            self.cash += price * quantity
            
            # Update position
            remaining_quantity = existing_pos.quantity - quantity
            if remaining_quantity > 0:
                existing_pos.quantity = remaining_quantity
            else:
                # Remove position if fully sold
                del self.positions[position_key]
        
        elif action == ActionType.HEDGE:
            if position_key not in self.positions:
                logger.warning(f"No position found for {asset_class} to hedge")
                return {"success": False, "error": "No position found"}
            
            # Apply hedging (this is simplified, real hedging would be more complex)
            cost = price * quantity * 0.1  # Assume hedging costs 10% of position value
            if cost > self.cash:
                logger.warning(f"Insufficient funds for hedging: {trade}")
                return {"success": False, "error": "Insufficient funds for hedging"}
            
            self.cash -= cost
            self.positions[position_key].add_hedge("options", quantity, price)
        
        # Record performance after trade
        self.record_performance()
        
        return {"success": True, "trade": trade}
    
    def set_stop_loss(
        self,
        asset_class: AssetClass,
        stop_loss_price: float
    ) -> bool:
        """Sets a stop loss for a position."""
        position_key = asset_class.value
        if position_key not in self.positions:
            return False
        
        self.positions[position_key].stop_loss = stop_loss_price
        return True
    
    def set_take_profit(
        self,
        asset_class: AssetClass,
        take_profit_price: float
    ) -> bool:
        """Sets a take profit level for a position."""
        position_key = asset_class.value
        if position_key not in self.positions:
            return False
        
        self.positions[position_key].take_profit = take_profit_price
        return True
    
    def check_stop_loss_take_profit(
        self,
        market_data: MarketData
    ) -> List[Dict[str, Any]]:
        """Checks if any stop loss or take profit levels have been triggered."""
        asset_class = market_data.asset_class
        position_key = asset_class.value
        executed_trades = []
        
        if position_key in self.positions:
            position = self.positions[position_key]
            current_price = market_data.price
            
            # Check stop loss
            if position.stop_loss and current_price <= position.stop_loss:
                trade = self.execute_trade(
                    asset_class=asset_class,
                    action=ActionType.SELL,
                    quantity=position.quantity,
                    price=current_price,
                    reason="Stop loss triggered"
                )
                if trade["success"]:
                    executed_trades.append(trade["trade"])
            
            # Check take profit
            elif position.take_profit and current_price >= position.take_profit:
                trade = self.execute_trade(
                    asset_class=asset_class,
                    action=ActionType.SELL,
                    quantity=position.quantity,
                    price=current_price,
                    reason="Take profit triggered"
                )
                if trade["success"]:
                    executed_trades.append(trade["trade"])
        
        return executed_trades
    
    def record_performance(self) -> None:
        """Records the current portfolio performance."""
        total_value = self.cash
        for position_key, position in self.positions.items():
            # In a real system, you'd get current prices from market data
            # For simplicity, we use entry price
            position_value = position.entry_price * position.quantity
            total_value += position_value
        
        timestamp = datetime.datetime.now()
        performance = {
            "timestamp": timestamp.isoformat(),
            "cash": self.cash,
            "position_value": total_value - self.cash,
            "total_value": total_value,
            "return_pct": ((total_value / self.initial_cash) - 1) * 100
        }
        self.performance_history.append(performance)
    
    def calculate_allocation(self) -> Dict[str, float]:
        """Calculates the current portfolio allocation."""
        total_value = self.cash
        for position_key, position in self.positions.items():
            position_value = position.entry_price * position.quantity
            total_value += position_value
        
        allocation = {"cash": (self.cash / total_value) * 100}
        for position_key, position in self.positions.items():
            position_value = position.entry_price * position.quantity
            allocation[position_key] = (position_value / total_value) * 100
        
        return allocation
    
    def get_positions_summary(self) -> List[Dict[str, Any]]:
        """Returns a summary of current positions."""
        positions_summary = []
        for position_key, position in self.positions.items():
            position_value = position.entry_price * position.quantity
            positions_summary.append({
                "asset_class": position.asset_class,
                "quantity": position.quantity,
                "entry_price": position.entry_price,
                "current_value": position_value,
                "entry_date": position.entry_timestamp.date().isoformat(),
                "has_hedge": len(position.hedge_positions) > 0
            })
        
        return positions_summary
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the portfolio to a dictionary for serialization."""
        positions_dict = {k: v.to_dict() for k, v in self.positions.items()}
        return {
            "cash": self.cash,
            "initial_cash": self.initial_cash,
            "positions": positions_dict,
            "trade_history": self.trade_history,
            "performance_history": self.performance_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Portfolio':
        """Creates a Portfolio instance from a dictionary."""
        portfolio = cls(initial_cash=data["initial_cash"])
        portfolio.cash = data["cash"]
        
        # Reconstruct positions
        positions_dict = data.get("positions", {})
        portfolio.positions = {
            k: PortfolioPosition.from_dict(v) for k, v in positions_dict.items()
        }
        
        portfolio.trade_history = data.get("trade_history", [])
        portfolio.performance_history = data.get("performance_history", [])
        
        return portfolio

class AnalystAgent(Agent):
    """Base class for specialized analyst agents (Bitcoin, Stocks, Forex)."""
    def __init__(
        self,
        agent_name: str,
        asset_class: AssetClass,
        model_name: str = "gemini/gemini-2.0-flash",
        tools: Optional[List[Callable]] = None,
        saved_state_path: Optional[str] = None,
        **kwargs
    ):
        self.asset_class = asset_class
        system_prompt = self._get_system_prompt()
        super().__init__(
            agent_name=agent_name,
            system_prompt=system_prompt,
            model_name=model_name,
            tools=tools,
            saved_state_path=saved_state_path,
            **kwargs
        )
        self.memory = AgentMemory()
        self.conversation = Conversation()
        self.portfolio = None  # Set by HedgeFundSwarm
        self.allocation = 0.0  # Current budget allocation

    def _get_system_prompt(self) -> str:
        """Generates the system prompt for the analyst agent."""
        return f"""You are an expert financial analyst specializing in {self.asset_class}.
        Your role is to analyze market data, news, and trends to make informed investment decisions.
        You're part of a hedge fund team with other analysts, each specializing in different asset classes.
        
        Your responsibilities include:
        1. Analyzing market data using technical indicators, patterns, and news
        2. Making trading decisions (BUY, SELL, HOLD, HEDGE)
        3. Setting stop-loss and take-profit levels
        4. Generating reports for budget allocation conferences
        5. Sharing investment experiences with other analysts
        6. Responding to extreme market conditions
        
        You have access to three types of memory:
        - MMI (Market Information Memory): Stores market data and basic analysis
        - MIR (Investment Reflection Memory): Records past decisions and their outcomes
        - MGE (General Experience Memory): Knowledge from conferences and accumulated experience
        
        As a {self.asset_class} expert, you should be particularly attentive to factors specific to this asset class.
        
        Always format your analysis and decisions in JSON format.
        """

    def analyze_market(self, market_data: MarketData) -> Dict[str, Any]:
        """Analyzes the market using tools and LLM."""
        try:
            # Use tools for technical analysis
            tool_results = {}
            if self.tools:
                for tool in self.tools:
                    tool_results[tool.__name__] = tool(market_data)
            
            # Retrieve relevant memories
            recent_market_info = self.memory.get_recent_market_info(5)
            relevant_reflections = self.memory.get_relevant_reflections(
                f"{market_data.asset_class} {market_data.price} {market_data.volatility}"
            )
            
            # Check current portfolio positions
            positions_summary = []
            if self.portfolio:
                positions_summary = self.portfolio.get_positions_summary()
                # Filter for this asset class
                positions_summary = [
                    pos for pos in positions_summary 
                    if pos["asset_class"] == self.asset_class
                ]
            
            # Prepare prompt for LLM
            prompt = f"""Analyze the following market data for {self.asset_class}:
            
            Market Data:
            {json.dumps(market_data.to_dict(), indent=2)}
            
            Technical Analysis Results:
            {json.dumps(tool_results, indent=2)}
            
            Recent Market Information (from memory):
            {json.dumps(recent_market_info, indent=2)}
            
            Relevant Past Reflections:
            {json.dumps(relevant_reflections, indent=2)}
            
            Current Positions:
            {json.dumps(positions_summary, indent=2)}
            
            Current Budget Allocation: ${self.allocation}
            
            Based on this information, provide a comprehensive analysis of the market situation.
            Include:
            1. Key market indicators and what they suggest
            2. Market sentiment based on news
            3. Technical analysis interpretation
            4. Risk assessment
            5. Correlation with other markets (if relevant)
            
            Format your response as a JSON object with the following structure:
            {{
                "market_summary": "Brief summary of current market situation",
                "key_indicators": [{"indicator": "name", "value": "value", "interpretation": "what it means"}],
                "market_sentiment": "bullish/bearish/neutral with explanation",
                "technical_analysis": "interpretation of technical indicators",
                "risk_assessment": "low/medium/high with explanation",
                "correlation_analysis": "relationship with other markets",
                "recommendation": "overall trading recommendation"
            }}
            """
            
            # Get analysis from LLM
            response = self.run(prompt)
            self.conversation.add(role=self.agent_name, content=response)
            
            # Parse JSON response
            analysis_result = self._safe_json_loads(response)
            
            # Update memory with market information
            self.memory.add_market_info({
                "asset_class": market_data.asset_class,
                "price": market_data.price,
                "timestamp": market_data.timestamp.isoformat(),
                "volatility": market_data.volatility,
                "analysis": analysis_result
            })
            
            return analysis_result

        except Exception as e:
            logger.error(f"Error in analyze_market for {self.agent_name}: {e}")
            return {"error": str(e)}

    def make_decision(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Makes an investment decision based on the analysis."""
        try:
            # Get current positions and allocation
            positions_summary = []
            allocation = 0.0
            if self.portfolio:
                positions_summary = self.portfolio.get_positions_summary()
                # Filter for this asset class
                positions_summary = [
                    pos for pos in positions_summary 
                    if pos["asset_class"] == self.asset_class
                ]
                allocation = self.allocation
            
            # Retrieve relevant past reflections
            relevant_reflections = self.memory.get_relevant_reflections(
                f"{self.asset_class} decision {analysis_result.get('market_sentiment', '')}"
            )
            
            # Prepare prompt for LLM
            prompt = f"""Based on the following market analysis, make an investment decision:
            
            Analysis Result:
            {json.dumps(analysis_result, indent=2)}
            
            Current Positions:
            {json.dumps(positions_summary, indent=2)}
            
            Current Budget Allocation: ${allocation}
            
            Past Investment Reflections:
            {json.dumps(relevant_reflections, indent=2)}
            
            Based on this information, make a detailed investment decision. Consider:
            1. Whether to BUY, SELL, HOLD, or HEDGE
            2. The appropriate position size (as a percentage of your allocation)
            3. Stop-loss and take-profit levels if applicable
            4. Reasoning behind your decision
            5. Potential risks and how to mitigate them
            
            Format your response as a JSON object with the following structure:
            {{
                "action": "BUY/SELL/HOLD/HEDGE",
                "asset_class": "{self.asset_class}",
                "quantity_percent": 10-100,
                "reasoning": "detailed explanation of decision",
                "stop_loss_percent": percentage below entry for stop loss (if applicable),
                "take_profit_percent": percentage above entry for take profit (if applicable),
                "risk_mitigation": "specific strategies to mitigate identified risks",
                "confidence": 1-10
            }}
            """
            
            # Get decision from LLM
            response = self.run(prompt)
            self.conversation.add(role=self.agent_name, content=response)
            
            # Parse JSON response
            decision = self._safe_json_loads(response)
            
            # Add reflection to memory
            self.memory.add_investment_reflection({
                "decision": decision,
                "analysis": analysis_result,
                "context": {
                    "asset_class": self.asset_class,
                    "allocation": allocation,
                    "has_position": len(positions_summary) > 0
                }
            })
            
            return decision
        
        except Exception as e:
            logger.error(f"Error in make_decision for {self.agent_name}: {e}")
            return {"error": str(e)}

    def execute_decision(
        self,
        decision: Dict[str, Any],
        market_data: MarketData
    ) -> Dict[str, Any]:
        """Executes the investment decision on the portfolio."""
        if not self.portfolio:
            return {"error": "No portfolio assigned to agent"}
        
        try:
            action = decision.get("action")
            if not action:
                return {"error": "No action specified in decision"}
            
            # Convert action to ActionType
            try:
                action_type = ActionType(action)
            except ValueError:
                return {"error": f"Invalid action type: {action}"}
            
            # Calculate quantity based on percentage of allocation
            quantity_percent = decision.get("quantity_percent", 0)
            quantity = 0.0
            
            if action_type in [ActionType.BUY, ActionType.HEDGE]:
                # Calculate quantity based on allocation and price
                amount_to_use = (self.allocation * quantity_percent / 100)
                quantity = amount_to_use / market_data.price
            elif action_type == ActionType.SELL:
                # Get current position
                for pos in self.portfolio.get_positions_summary():
                    if pos["asset_class"] == self.asset_class:
                        # Sell percentage of current position
                        quantity = pos["quantity"] * (quantity_percent / 100)
                        break
            
            if quantity <= 0 and action_type != ActionType.HOLD:
                return {"error": "Invalid quantity calculated", "quantity": quantity}
            
            # Execute the trade
            result = {"success": True, "action": action}
            
            if action_type == ActionType.HOLD:
                # No trade needed
                result["message"] = "Holding position, no trade executed"
            else:
                # Execute trade on portfolio
                trade_result = self.portfolio.execute_trade(
                    asset_class=market_data.asset_class,
                    action=action_type,
                    quantity=quantity,
                    price=market_data.price,
                    reason=decision.get("reasoning", "No reasoning provided")
                )
                result.update(trade_result)
                
                # Set stop loss and take profit if provided
                if trade_result.get("success") and action_type == ActionType.BUY:
                    stop_loss_percent = decision.get("stop_loss_percent")
                    take_profit_percent = decision.get("take_profit_percent")
                    
                    if stop_loss_percent:
                        stop_price = market_data.price * (1 - stop_loss_percent / 100)
                        self.portfolio.set_stop_loss(market_data.asset_class, stop_price)
                    
                    if take_profit_percent:
                        take_profit_price = market_data.price * (1 + take_profit_percent / 100)
                        self.portfolio.set_take_profit(market_data.asset_class, take_profit_price)
            
            # Add outcome to memory
            self.memory.add_investment_reflection({
                "decision": decision,
                "outcome": result,
                "market_data": market_data.to_dict()
            })
            
            return result
        
        except Exception as e:
            logger.error(f"Error in execute_decision for {self.agent_name}: {e}")
            return {"error": str(e)}

    def generate_report(self) -> Dict[str, Any]:
        """Generates a report for the budget allocation conference."""
        try:
            # Get performance data
            positions_summary = []
            recent_trades = []
            allocation = 0.0
            performance = {"return_pct": 0}
            
            if self.portfolio:
                positions_summary = [
                    pos for pos in self.portfolio.get_positions_summary() 
                    if pos["asset_class"] == self.asset_class
                ]
                allocation = self.allocation
                
                # Get recent trades for this asset class
                recent_trades = [
                    trade for trade in self.portfolio.trade_history[-10:]
                    if trade["asset_class"] == self.asset_class
                ]
                
                # Calculate performance for this asset class
                # This is simplified - in a real system you'd calculate proper P&L
                if positions_summary:
                    position_value = sum(pos["current_value"] for pos in positions_summary)
                    cost_basis = sum(pos["entry_price"] * pos["quantity"] for pos in positions_summary)
                    if cost_basis > 0:
                        performance = {"return_pct": ((position_value / cost_basis) - 1) * 100}
            
            # Get relevant market memories
            recent_market_info = self.memory.get_recent_market_info(3)
            
            # Get relevant reflections
            relevant_reflections = self.memory.get_relevant_reflections(f"{self.asset_class} performance", 3)
            
            # Prepare prompt for LLM
            prompt = f"""Generate a concise report for the upcoming budget allocation conference.
            
            Current Asset Class: {self.asset_class}
            Current Allocation: ${allocation}
            Current Positions: {json.dumps(positions_summary, indent=2)}
            Recent Trades: {json.dumps(recent_trades, indent=2)}
            Performance: {json.dumps(performance, indent=2)}
            Recent Market Information: {json.dumps(recent_market_info, indent=2)}
            Investment Reflections: {json.dumps(relevant_reflections, indent=2)}
            
            Based on this information, create a comprehensive report that includes:
            1. Current profit/loss situation
            2. Market outlook for your asset class
            3. Expected budget allocation request (with justification)
            4. Planned strategy for the next period
            5. Key risks and opportunities
            
            Format your response as a JSON object with the following structure:
            {{
                "asset_class": "{self.asset_class}",
                "current_situation": "summary of current positions and performance",
                "market_outlook": "forecast for the coming period",
                "budget_request": requested amount in dollars,
                "budget_justification": "detailed explanation for requested amount",
                "strategy_overview": "planned approach for next period",
                "key_risks": ["risk1", "risk2", ...],
                "key_opportunities": ["opportunity1", "opportunity2", ...],
                "confidence_level": 1-10
            }}
            """
            
            # Get report from LLM
            response = self.run(prompt)
            self.conversation.add(role=self.agent_name, content=response)
            
            # Parse JSON response
            report = self._safe_json_loads(response)
            
            # Add to memory
            self.memory.add_general_experience({
                "report_type": "budget_allocation",
                "report": report,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            return report
        
        except Exception as e:
            logger.error(f"Error in generate_report for {self.agent_name}: {e}")
            return {"error": str(e)}
    
    def attend_conference(self, reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Processes insights from other analysts' reports during a conference."""
        try:
            # Filter out own report
            other_reports = [r for r in reports if r.get("asset_class") != self.asset_class]
            
            # Prepare prompt for LLM
            prompt = f"""You are attending a budget allocation conference with other analysts.
            Review these reports from other asset class specialists and extract insights
            that could be relevant for your {self.asset_class} strategy:
            
            Other Analysts' Reports:
            {json.dumps(other_reports, indent=2)}
            
            Based on these reports, identify:
            1. Cross-market correlations or patterns
            2. Insights that might affect your asset class
            3. Risk factors mentioned by other analysts
            4. Opportunities for hedging or diversification
            5. New strategies you might adopt
            
            Format your response as a JSON object with the following structure:
            {{
                "cross_market_insights": ["insight1", "insight2", ...],
                "relevant_factors_for_{self.asset_class}": ["factor1", "factor2", ...],
                "risk_considerations": ["risk1", "risk2", ...],
                "hedging_opportunities": ["opportunity1", "opportunity2", ...],
                "strategy_adaptations": ["adaptation1", "adaptation2", ...],
                "overall_market_sentiment": "bullish/bearish/neutral"
            }}
            """
            
            # Get insights from LLM
            response = self.run(prompt)
            self.conversation.add(role=self.agent_name, content=response)
            
            # Parse JSON response
            insights = self._safe_json_loads(response)
            
            # Add to memory
            self.memory.add_general_experience({
                "experience_type": "conference",
                "insights": insights,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            return insights
        
        except Exception as e:
            logger.error(f"Error in attend_conference for {self.agent_name}: {e}")
            return {"error": str(e)}
    
    def respond_to_market_event(
        self,
        event_description: str,
        market_data: MarketData
    ) -> Dict[str, Any]:
        """Responds to significant market events or crises."""
        try:
            # Get relevant reflections and experiences
            relevant_reflections = self.memory.get_relevant_reflections(event_description, 5)
            relevant_experiences = self.memory.get_relevant_experiences(event_description, 5)
            
            # Get current positions
            positions_summary = []
            if self.portfolio:
                positions_summary = [
                    pos for pos in self.portfolio.get_positions_summary() 
                    if pos["asset_class"] == self.asset_class
                ]
            
            # Prepare prompt for LLM
            prompt = f"""An important market event has occurred:
            
            Event Description: {event_description}
            
            Current Market Data:
            {json.dumps(market_data.to_dict(), indent=2)}
            
            Your Current Positions:
            {json.dumps(positions_summary, indent=2)}
            
            Relevant Past Experiences:
            {json.dumps(relevant_experiences, indent=2)}
            
            Relevant Past Reflections:
            {json.dumps(relevant_reflections, indent=2)}
            
            As a {self.asset_class} specialist, provide an emergency response plan that includes:
            1. Immediate actions to take (sell, hedge, hold, buy more)
            2. Risk assessment of the situation
            3. Potential short-term impacts on your asset class
            4. Potential long-term impacts on your asset class
            5. Recommended portfolio adjustments
            
            Format your response as a JSON object with the following structure:
            {{
                "event_assessment": "your assessment of the event",
                "risk_level": "critical/high/medium/low",
                "immediate_actions": [{"action": "action_type", "reason": "explanation"}],
                "short_term_impact": "description of likely short-term effects",
                "long_term_impact": "description of possible long-term effects",
                "portfolio_recommendations": ["recommendation1", "recommendation2", ...],
                "confidence_level": 1-10
            }}
            """
            
            # Get response from LLM
            response = self.run(prompt)
            self.conversation.add(role=self.agent_name, content=response)
            
            # Parse JSON response
            event_response = self._safe_json_loads(response)
            
            # Add to memory
            self.memory.add_general_experience({
                "experience_type": "market_event",
                "event": event_description,
                "response": event_response,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            return event_response
        
        except Exception as e:
            logger.error(f"Error in respond_to_market_event for {self.agent_name}: {e}")
            return {"error": str(e)}
    
    def _safe_json_loads(self, json_str: str) -> Dict[str, Any]:
        """Safely loads JSON from a string, handling common formatting issues."""
        try:
            # Try direct parsing
            return json.loads(json_str)
        except json.JSONDecodeError:
            try:
                # Look for JSON object within text
                match = re.search(r'({.*})', json_str, re.DOTALL)
                if match:
                    return json.loads(match.group(0))
                else:
                    return {"error": "Failed to parse JSON response", "raw_response": json_str}
            except Exception:
                return {"error": "Failed to parse JSON response", "raw_response": json_str}
    
    def save_state(self) -> None:
        """Saves the agent's state to disk."""
        if not self.saved_state_path:
            logger.warning(f"No saved_state_path specified for {self.agent_name}")
            return
        
        try:
            state = {
                "memory": self.memory.to_dict(),
                "conversation": self.conversation.to_dict(),
                "asset_class": self.asset_class,
                "allocation": self.allocation
            }
            
            path = Path(self.saved_state_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
            
            logger.info(f"Saved state for {self.agent_name} to {self.saved_state_path}")
        
        except Exception as e:
            logger.error(f"Error saving state for {self.agent_name}: {e}")
    
    def load_state(self) -> None:
        """Loads the agent's state from disk."""
        if not self.saved_state_path:
            logger.warning(f"No saved_state_path specified for {self.agent_name}")
            return
        
        try:
            path = Path(self.saved_state_path)
            if not path.exists():
                logger.warning(f"No saved state found at {self.saved_state_path}")
                return
            
            with open(path, 'r') as f:
                state = json.load(f)
            
            self.memory = AgentMemory.from_dict(state.get("memory", {}))
            self.conversation = Conversation.from_dict(state.get("conversation", {}))
            self.allocation = state.get("allocation", 0.0)
            
            logger.info(f"Loaded state for {self.agent_name} from {self.saved_state_path}")
        
        except Exception as e:
            logger.error(f"Error loading state for {self.agent_name}: {e}")


class BitcoinAnalyst(AnalystAgent):
    """Bitcoin specialist analyst."""
    def __init__(self, **kwargs):
        super().__init__(
            agent_name="Bitcoin Analyst",
            asset_class=AssetClass.BITCOIN,
            **kwargs
        )


class StocksAnalyst(AnalystAgent):
    """Stocks specialist analyst."""
    def __init__(self, **kwargs):
        super().__init__(
            agent_name="Stocks Analyst",
            asset_class=AssetClass.STOCKS,
            **kwargs
        )


class ForexAnalyst(AnalystAgent):
    """Forex specialist analyst."""
    def __init__(self, **kwargs):
        super().__init__(
            agent_name="Forex Analyst",
            asset_class=AssetClass.FOREX,
            **kwargs
        )


class HedgeFundSwarm:
    """Coordinates multiple analyst agents to manage a hedge fund portfolio."""
    def __init__(
        self,
        initial_capital: float = 1000000.0,
        saved_state_path: Optional[str] = None,
        model_name: str = "gemini/gemini-2.0-flash"
    ):
        self.portfolio = Portfolio(initial_cash=initial_capital)
        self.initial_capital = initial_capital
        self.saved_state_path = saved_state_path
        self.model_name = model_name
        
        # Initialize analysts
        self.analysts = {
            AssetClass.BITCOIN: BitcoinAnalyst(
                model_name=model_name,
                saved_state_path=self._get_analyst_state_path(AssetClass.BITCOIN)
            ),
            AssetClass.STOCKS: StocksAnalyst(
                model_name=model_name,
                saved_state_path=self._get_analyst_state_path(AssetClass.STOCKS)
            ),
            AssetClass.FOREX: ForexAnalyst(
                model_name=model_name,
                saved_state_path=self._get_analyst_state_path(AssetClass.FOREX)
            )
        }
        
        # Set portfolio for all analysts
        for analyst in self.analysts.values():
            analyst.portfolio = self.portfolio
        
        # Default initial allocation (equal split)
        self._allocate_budget_equally()
        
        # Load state if available
        if saved_state_path:
            self.load_state()
    
    def _get_analyst_state_path(self, asset_class: AssetClass) -> str:
        """Generates the state path for an analyst."""
        if not self.saved_state_path:
            return None
        
        base_path = Path(self.saved_state_path)
        return str(base_path.parent / f"{base_path.stem}_{asset_class.value.lower()}{base_path.suffix}")
    
    def _allocate_budget_equally(self) -> None:
        """Allocates budget equally among all analysts."""
        num_analysts = len(self.analysts)
        allocation_per_analyst = self.portfolio.cash / num_analysts
        
        for analyst in self.analysts.values():
            analyst.allocation = allocation_per_analyst
    
    def process_market_data(
        self,
        market_data: Dict[AssetClass, MarketData]
    ) -> Dict[AssetClass, Dict[str, Any]]:
        """Processes market data for all asset classes."""
        results = {}
        
        for asset_class, data in market_data.items():
            if asset_class in self.analysts:
                analyst = self.analysts[asset_class]
                
                # Check stop loss and take profit
                self.portfolio.check_stop_loss_take_profit(data)
                
                # Analyze market
                analysis = analyst.analyze_market(data)
                
                # Make decision
                decision = analyst.make_decision(analysis)
                
                # Execute decision
                execution_result = analyst.execute_decision(decision, data)
                
                results[asset_class] = {
                    "analysis": analysis,
                    "decision": decision,
                    "execution": execution_result
                }
        
        return results
    
    def hold_budget_conference(self) -> Dict[str, Any]:
        """Conducts a budget allocation conference among analysts."""
        # Step 1: Each analyst generates a report
        reports = {}
        for asset_class, analyst in self.analysts.items():
            reports[asset_class] = analyst.generate_report()
        
        # Step 2: Each analyst attends the conference and learns from others
        conference_insights = {}
        for asset_class, analyst in self.analysts.items():
            conference_insights[asset_class] = analyst.attend_conference(list(reports.values()))
        
        # Step 3: Reallocate budget based on reports and portfolio performance
        new_allocations = self._determine_budget_allocation(reports)
        
        # Step 4: Apply new allocations
        for asset_class, allocation in new_allocations.items():
            if asset_class in self.analysts:
                self.analysts[asset_class].allocation = allocation
        
        return {
            "reports": reports,
            "insights": conference_insights,
            "new_allocations": new_allocations
        }
    
    def _determine_budget_allocation(
        self,
        reports: Dict[AssetClass, Dict[str, Any]]
    ) -> Dict[AssetClass, float]:
        """Determines budget allocation based on analyst reports and market performance."""
        # Extract budget requests and confidence levels
        requests = {}
        confidence_levels = {}
        total_requested = 0
        
        for asset_class, report in reports.items():
            if "budget_request" in report and "confidence_level" in report:
                requests[asset_class] = float(report["budget_request"])
                confidence_levels[asset_class] = float(report["confidence_level"])
                total_requested += requests[asset_class]
        
        # Adjust requests if total exceeds available cash
        new_allocations = {}
        available_cash = self.portfolio.cash
        
        if total_requested > available_cash and total_requested > 0:
            # Scale down proportionally
            scaling_factor = available_cash / total_requested
            for asset_class, request in requests.items():
                new_allocations[asset_class] = request * scaling_factor
        else:
            new_allocations = requests
        
        # Ensure all analysts have at least some allocation
        for asset_class in self.analysts:
            if asset_class not in new_allocations or new_allocations[asset_class] < 1000:
                # Minimum allocation of $1000 or 1% of available cash, whichever is higher
                min_allocation = max(1000, available_cash * 0.01)
                new_allocations[asset_class] = min_allocation
        
        return new_allocations
    
    def handle_market_event(
        self,
        event_description: str,
        market_data: Dict[AssetClass, MarketData]
    ) -> Dict[AssetClass, Dict[str, Any]]:
        """Handles significant market events by coordinating responses from all analysts."""
        responses = {}
        
        # Get individual responses from each analyst
        for asset_class, analyst in self.analysts.items():
            if asset_class in market_data:
                response = analyst.respond_to_market_event(
                    event_description,
                    market_data[asset_class]
                )
                responses[asset_class] = response
        
        # Aggregate risk assessments
        risk_levels = [
            response.get("risk_level", "medium") 
            for response in responses.values()
        ]
        
        # Count occurrences of each risk level
        risk_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for level in risk_levels:
            if level in risk_counts:
                risk_counts[level] += 1
        
        # Determine overall risk level
        overall_risk = "medium"
        if risk_counts["critical"] >= 1:
            overall_risk = "critical"
        elif risk_counts["high"] > len(self.analysts) / 2:
            overall_risk = "high"
        elif risk_counts["low"] == len(self.analysts):
            overall_risk = "low"
        
        # Take emergency actions if risk is critical
        emergency_actions = []
        if overall_risk == "critical":
            # Execute immediate actions from highest confidence responses
            for asset_class, response in responses.items():
                confidence = response.get("confidence_level", 5)
                if confidence >= 7 and "immediate_actions" in response:
                    for action in response["immediate_actions"]:
                        # Execute action (simplified)
                        if action.get("action") in ["SELL", "HEDGE"] and asset_class in market_data:
                            result = self.analysts[asset_class].execute_decision(
                                {"action": action.get("action"), "quantity_percent": 100},
                                market_data[asset_class]
                            )
                            emergency_actions.append({
                                "asset_class": asset_class,
                                "action": action,
                                "result": result
                            })
        
        return {
            "responses": responses,
            "overall_risk": overall_risk,
            "emergency_actions": emergency_actions
        }
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Returns a summary of the current portfolio state."""
        total_value = self.portfolio.cash
        for position in self.portfolio.get_positions_summary():
            total_value += position.get("current_value", 0)
        
        return {
            "total_value": total_value,
            "cash": self.portfolio.cash,
            "total_return_pct": ((total_value / self.initial_capital) - 1) * 100,
            "positions": self.portfolio.get_positions_summary(),
            "allocation": {
                asset_class.value: analyst.allocation
                for asset_class, analyst in self.analysts.items()
            }
        }
    
    def save_state(self) -> None:
        """Saves the swarm state to disk."""
        if not self.saved_state_path:
            logger.warning("No saved_state_path specified")
            return
        
        try:
            # Save portfolio state
            state = {
                "portfolio": self.portfolio.to_dict(),
                "initial_capital": self.initial_capital
            }
            
            path = Path(self.saved_state_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)
            
            # Save individual analyst states
            for analyst in self.analysts.values():
                analyst.save_state()
            
            logger.info(f"Saved state to {self.saved_state_path}")
        
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def load_state(self) -> None:
        """Loads the swarm state from disk."""
        if not self.saved_state_path:
            logger.warning("No saved_state_path specified")
            return
        
        try:
            path = Path(self.saved_state_path)
            if not path.exists():
                logger.warning(f"No saved state found at {self.saved_state_path}")
                return
            
            with open(path, 'r') as f:
                state = json.load(f)
            
            # Load portfolio
            if "portfolio" in state:
                self.portfolio = Portfolio.from_dict(state["portfolio"])
                
                # Update portfolio reference in analysts
                for analyst in self.analysts.values():
                    analyst.portfolio = self.portfolio
            
            self.initial_capital = state.get("initial_capital", self.initial_capital)
            
            # Load individual analyst states
            for analyst in self.analysts.values():
                analyst.load_state()
            
            logger.info(f"Loaded state from {self.saved_state_path}")
        
        except Exception as e:
            logger.error(f"Error loading state: {e}")

# Example usage
if __name__ == "__main__":
    import random
    from time import sleep
    
    # Create a hedge fund swarm
    fund = HedgeFundSwarm(
        initial_capital=1000000.0,
        saved_state_path="/d:/Github/swarms/examples/hedge_fund_state.json"
    )
    
    # Simulate market data
    for day in range(10):
        print(f"\n===== Day {day+1} =====")
        
        # Generate simulated market data
        market_data = {
            AssetClass.BITCOIN: MarketData(
                asset_class=AssetClass.BITCOIN,
                price=30000 + random.uniform(-1000, 1000),
                change_percent=random.uniform(-5, 5),
                volume=random.uniform(10000, 50000),
                timestamp=datetime.datetime.now(),
                volatility=random.uniform(1, 5),
                news=[{"title": f"Bitcoin news {i}", "sentiment": random.choice(["positive", "negative", "neutral"])} for i in range(3)],
                technical_indicators={"RSI": random.uniform(30, 70), "MACD": random.uniform(-2, 2)}
            ),
            AssetClass.STOCKS: MarketData(
                asset_class=AssetClass.STOCKS,
                price=4000 + random.uniform(-100, 100),
                change_percent=random.uniform(-3, 3),
                volume=random.uniform(100000, 500000),
                timestamp=datetime.datetime.now(),
                volatility=random.uniform(0.5, 3),
                news=[{"title": f"Stock market news {i}", "sentiment": random.choice(["positive", "negative", "neutral"])} for i in range(3)],
                technical_indicators={"RSI": random.uniform(30, 70), "MACD": random.uniform(-1, 1)}
            ),
            AssetClass.FOREX: MarketData(
                asset_class=AssetClass.FOREX,
                price=1.1 + random.uniform(-0.05, 0.05),
                change_percent=random.uniform(-2, 2),
                volume=random.uniform(1000000, 5000000),
                timestamp=datetime.datetime.now(),
                volatility=random.uniform(0.3, 2),
                news=[{"title": f"Forex news {i}", "sentiment": random.choice(["positive", "negative", "neutral"])} for i in range(3)],
                technical_indicators={"RSI": random.uniform(30, 70), "MACD": random.uniform(-0.5, 0.5)}
            )
        }
        
        # Process market data
        print("Processing market data...")
        results = fund.process_market_data(market_data)
        
        # Print summarized results
        for asset_class, result in results.items():
            print(f"\n{asset_class} Analysis:")
            if "market_summary" in result["analysis"]:
                print(f"Summary: {result['analysis']['market_summary']}")
            print(f"Decision: {result['decision'].get('action')} with {result['decision'].get('confidence')} confidence")
            print(f"Execution: {'Success' if result['execution'].get('success', False) else 'Failed'}")
        
        # Hold budget conference every 5 days
        if (day + 1) % 5 == 0:
            print("\nHolding budget allocation conference...")
            conference_results = fund.hold_budget_conference()
            
            print("\nNew Allocations:")
            for asset_class, allocation in conference_results["new_allocations"].items():
                print(f"{asset_class}: ${allocation:,.2f}")
        
        # Simulate market event on day 7
        if day == 6:
            print("\n!!! MARKET EVENT: Major crypto exchange hacked !!!")
            event_results = fund.handle_market_event(
                "Major cryptocurrency exchange hacked, 100,000 BTC stolen",
                market_data
            )
            
            print(f"Overall Risk Assessment: {event_results['overall_risk']}")
            if event_results["emergency_actions"]:
                print("Emergency Actions Taken:")
                for action in event_results["emergency_actions"]:
                    print(f"- {action['asset_class']}: {action['action'].get('action')}")
        
        # Print portfolio summary
        summary = fund.get_portfolio_summary()
        print(f"\nPortfolio Summary - Day {day+1}:")
        print(f"Total Value: ${summary['total_value']:,.2f}")
        print(f"Cash: ${summary['cash']:,.2f}")
        print(f"Return: {summary['total_return_pct']:.2f}%")
        
        # Save state at the end of each day
        fund.save_state()
        
        # Simulate the passage of time
        sleep(1)
    
    print("\nSimulation complete")
