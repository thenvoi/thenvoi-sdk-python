#!/usr/bin/env python3
"""
Data analysis agent example.

A more advanced example showing a Letta agent designed for data analysis
tasks with specialized tools and memory management.

Usage:
    # Set environment variables
    export THENVOI_AGENT_ID="your-agent-id"
    export THENVOI_API_KEY="your-api-key"
    export LETTA_BASE_URL="http://localhost:8283"

    # Run
    uv run --extra letta python examples/letta/04_data_agent.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import statistics
from datetime import datetime
from typing import Any

from thenvoi import Agent
from thenvoi.adapters.letta import LettaAdapter, LettaConfig, LettaMode
from thenvoi.adapters.letta.tools import CustomToolBuilder
from thenvoi.runtime.types import SessionConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Sample Data Store (simulating a data source)
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_DATA = {
    "sales_q4_2024": {
        "description": "Q4 2024 Sales Data",
        "records": [
            {"month": "October", "revenue": 125000, "units": 450, "region": "North"},
            {"month": "October", "revenue": 98000, "units": 380, "region": "South"},
            {"month": "November", "revenue": 142000, "units": 520, "region": "North"},
            {"month": "November", "revenue": 115000, "units": 445, "region": "South"},
            {"month": "December", "revenue": 185000, "units": 680, "region": "North"},
            {"month": "December", "revenue": 156000, "units": 590, "region": "South"},
        ],
    },
    "customer_feedback": {
        "description": "Customer Satisfaction Survey Results",
        "records": [
            {"category": "Product Quality", "score": 4.5, "responses": 234},
            {"category": "Customer Service", "score": 4.2, "responses": 189},
            {"category": "Pricing", "score": 3.8, "responses": 201},
            {"category": "Delivery Speed", "score": 4.1, "responses": 156},
            {"category": "Website Usability", "score": 3.9, "responses": 178},
        ],
    },
    "inventory": {
        "description": "Current Inventory Levels",
        "records": [
            {
                "product": "Widget A",
                "quantity": 1250,
                "reorder_point": 500,
                "status": "ok",
            },
            {
                "product": "Widget B",
                "quantity": 340,
                "reorder_point": 400,
                "status": "low",
            },
            {
                "product": "Gadget X",
                "quantity": 890,
                "reorder_point": 300,
                "status": "ok",
            },
            {
                "product": "Gadget Y",
                "quantity": 125,
                "reorder_point": 200,
                "status": "critical",
            },
            {
                "product": "Component Z",
                "quantity": 2340,
                "reorder_point": 1000,
                "status": "ok",
            },
        ],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Data Analysis Tools
# ══════════════════════════════════════════════════════════════════════════════

tool_builder = CustomToolBuilder()


@tool_builder.tool
def list_datasets() -> str:
    """
    List all available datasets.

    Returns:
        List of dataset names and descriptions
    """
    result = ["Available datasets:\n"]
    for name, data in SAMPLE_DATA.items():
        result.append(f"- {name}: {data['description']}")
    return "\n".join(result)


@tool_builder.tool
def get_dataset(dataset_name: str) -> str:
    """
    Retrieve a dataset by name.

    Args:
        dataset_name: Name of the dataset to retrieve

    Returns:
        JSON representation of the dataset
    """
    if dataset_name not in SAMPLE_DATA:
        available = ", ".join(SAMPLE_DATA.keys())
        return f"Dataset '{dataset_name}' not found. Available: {available}"

    data = SAMPLE_DATA[dataset_name]
    return json.dumps(data, indent=2)


@tool_builder.tool
def analyze_numeric_field(dataset_name: str, field_name: str) -> str:
    """
    Compute statistics for a numeric field in a dataset.

    Args:
        dataset_name: Name of the dataset
        field_name: Name of the numeric field to analyze

    Returns:
        Statistical summary (mean, median, min, max, std dev)
    """
    if dataset_name not in SAMPLE_DATA:
        return f"Dataset '{dataset_name}' not found"

    records = SAMPLE_DATA[dataset_name]["records"]

    try:
        values = [r[field_name] for r in records if field_name in r]
        if not values:
            return f"Field '{field_name}' not found or has no numeric values"

        # Ensure all values are numeric
        values = [float(v) for v in values]

        return f"""Statistical Analysis of '{field_name}' in '{dataset_name}':
- Count: {len(values)}
- Mean: {statistics.mean(values):.2f}
- Median: {statistics.median(values):.2f}
- Min: {min(values):.2f}
- Max: {max(values):.2f}
- Std Dev: {statistics.stdev(values):.2f if len(values) > 1 else 'N/A'}
- Sum: {sum(values):.2f}"""
    except (ValueError, TypeError) as e:
        return f"Error analyzing field: {e}"


@tool_builder.tool
def filter_dataset(dataset_name: str, field: str, operator: str, value: str) -> str:
    """
    Filter dataset records based on a condition.

    Args:
        dataset_name: Name of the dataset
        field: Field to filter on
        operator: Comparison operator ('eq', 'gt', 'lt', 'gte', 'lte', 'contains')
        value: Value to compare against

    Returns:
        Filtered records as JSON
    """
    if dataset_name not in SAMPLE_DATA:
        return f"Dataset '{dataset_name}' not found"

    records = SAMPLE_DATA[dataset_name]["records"]

    def compare(record_value: Any, compare_value: str, op: str) -> bool:
        try:
            if op == "eq":
                return str(record_value) == compare_value
            elif op == "contains":
                return compare_value.lower() in str(record_value).lower()
            elif op in ("gt", "lt", "gte", "lte"):
                num_val = float(record_value)
                num_compare = float(compare_value)
                if op == "gt":
                    return num_val > num_compare
                elif op == "lt":
                    return num_val < num_compare
                elif op == "gte":
                    return num_val >= num_compare
                elif op == "lte":
                    return num_val <= num_compare
        except (ValueError, TypeError):
            return False
        return False

    filtered = [r for r in records if field in r and compare(r[field], value, operator)]

    return json.dumps(
        {
            "dataset": dataset_name,
            "filter": f"{field} {operator} {value}",
            "count": len(filtered),
            "records": filtered,
        },
        indent=2,
    )


@tool_builder.tool
def aggregate_by_field(dataset_name: str, group_by: str, aggregate_field: str) -> str:
    """
    Aggregate numeric field by grouping field.

    Args:
        dataset_name: Name of the dataset
        group_by: Field to group by
        aggregate_field: Numeric field to sum

    Returns:
        Aggregated results
    """
    if dataset_name not in SAMPLE_DATA:
        return f"Dataset '{dataset_name}' not found"

    records = SAMPLE_DATA[dataset_name]["records"]

    # Group and sum
    groups: dict[str, float] = {}
    for record in records:
        if group_by not in record or aggregate_field not in record:
            continue
        key = str(record[group_by])
        try:
            value = float(record[aggregate_field])
            groups[key] = groups.get(key, 0) + value
        except (ValueError, TypeError):
            continue

    return json.dumps(
        {
            "dataset": dataset_name,
            "grouped_by": group_by,
            "aggregated_field": aggregate_field,
            "results": groups,
            "total": sum(groups.values()),
        },
        indent=2,
    )


@tool_builder.tool
def get_report_timestamp() -> str:
    """Get current timestamp for reports."""
    return f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


async def main():
    # Load configuration from environment
    agent_id = os.environ.get("THENVOI_AGENT_ID")
    api_key = os.environ.get("THENVOI_API_KEY")
    letta_base_url = os.environ.get("LETTA_BASE_URL", "http://localhost:8283")

    if not agent_id or not api_key:
        raise ValueError(
            "Missing required environment variables: "
            "THENVOI_AGENT_ID and THENVOI_API_KEY"
        )

    # Log registered tools
    logger.info("Registered data analysis tools:")
    for name in tool_builder.get_tool_names():
        logger.info(f"  - {name}")

    # Configure Letta adapter for data analysis
    adapter = LettaAdapter(
        config=LettaConfig(
            # api_key is optional for self-hosted Letta
            mode=LettaMode.PER_ROOM,
            base_url=letta_base_url,
            model="openai/gpt-4o-mini",
            embedding_model="openai/text-embedding-3-small",
            persona="""You are a data analysis assistant specialized in business intelligence.

Your capabilities:
1. **Data Discovery**: List and explore available datasets
2. **Statistical Analysis**: Compute statistics on numeric fields
3. **Filtering**: Find records matching specific criteria
4. **Aggregation**: Group and summarize data

Available datasets:
- sales_q4_2024: Quarterly sales data by month and region
- customer_feedback: Customer satisfaction survey results
- inventory: Current inventory levels and status

When users ask about data:
1. First explore the available datasets if needed
2. Retrieve relevant data
3. Apply appropriate analysis (statistics, filtering, aggregation)
4. Present findings clearly with insights

Always explain your analysis methodology and highlight key findings.""",
            custom_tools=tool_builder.get_tool_definitions(),
        ),
        state_storage_path="~/.thenvoi/letta_data_agent_state.json",
    )

    # Create Thenvoi agent
    agent = Agent.create(
        adapter=adapter,
        agent_id=agent_id,
        api_key=api_key,
        session_config=SessionConfig(enable_context_hydration=False),
    )

    logger.info("=" * 60)
    logger.info("Starting Data Analysis Agent")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Sample questions to try:")
    logger.info('  - "What datasets are available?"')
    logger.info('  - "What was the total Q4 revenue?"')
    logger.info('  - "Show me revenue breakdown by region"')
    logger.info('  - "Which products have low inventory?"')
    logger.info('  - "What is the average customer satisfaction score?"')
    logger.info('  - "Compare North vs South region sales"')
    logger.info("")
    logger.info(f"Letta server: {letta_base_url}")
    logger.info("")

    # Run the agent
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
