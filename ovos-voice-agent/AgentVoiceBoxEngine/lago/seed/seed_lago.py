#!/usr/bin/env python3
"""Seed Lago with billable metrics and plans.

This script initializes Lago with the AgentVoiceBox billing configuration:
- Billable metrics (api_requests, audio_minutes, llm_tokens, etc.)
- Subscription plans (Free, Pro, Enterprise, Pro Annual)

Run this after Lago is up and you have an API key:
    python seed_lago.py --api-key YOUR_API_KEY

Requirements: 20.3, 20.4, 23.1
"""
import argparse
import json
import sys
from pathlib import Path

import httpx


def load_json(filename: str) -> dict:
    """Load JSON file from seed directory."""
    path = Path(__file__).parent / filename
    with open(path) as f:
        return json.load(f)


def create_billable_metrics(client: httpx.Client, api_url: str, api_key: str) -> None:
    """Create billable metrics in Lago."""
    print("\n=== Creating Billable Metrics ===")

    data = load_json("billable_metrics.json")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for metric in data["billable_metrics"]:
        print(f"  Creating metric: {metric['code']}...", end=" ")

        payload = {"billable_metric": metric}

        response = client.post(
            f"{api_url}/api/v1/billable_metrics",
            headers=headers,
            json=payload,
        )

        if response.status_code == 200:
            print("✓ Created")
        elif response.status_code == 422:
            # Already exists, try to update
            response = client.put(
                f"{api_url}/api/v1/billable_metrics/{metric['code']}",
                headers=headers,
                json=payload,
            )
            if response.status_code == 200:
                print("✓ Updated")
            else:
                print(f"✗ Failed: {response.text}")
        else:
            print(f"✗ Failed: {response.status_code} - {response.text}")


def create_plans(client: httpx.Client, api_url: str, api_key: str) -> None:
    """Create subscription plans in Lago."""
    print("\n=== Creating Subscription Plans ===")

    data = load_json("plans.json")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for plan in data["plans"]:
        print(f"  Creating plan: {plan['code']}...", end=" ")

        payload = {"plan": plan}

        response = client.post(
            f"{api_url}/api/v1/plans",
            headers=headers,
            json=payload,
        )

        if response.status_code == 200:
            print("✓ Created")
        elif response.status_code == 422:
            # Already exists, try to update
            response = client.put(
                f"{api_url}/api/v1/plans/{plan['code']}",
                headers=headers,
                json=payload,
            )
            if response.status_code == 200:
                print("✓ Updated")
            else:
                print(f"✗ Failed: {response.text}")
        else:
            print(f"✗ Failed: {response.status_code} - {response.text}")


def main():
    parser = argparse.ArgumentParser(
        description="Seed Lago with AgentVoiceBox billing configuration"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:3000",
        help="Lago API URL (default: http://localhost:3000)",
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="Lago API key",
    )

    args = parser.parse_args()

    print(f"Seeding Lago at {args.api_url}")

    with httpx.Client(timeout=30.0) as client:
        # Test connection
        try:
            response = client.get(f"{args.api_url}/health")
            if response.status_code != 200:
                print(f"Error: Lago health check failed: {response.status_code}")
                sys.exit(1)
        except httpx.ConnectError:
            print(f"Error: Cannot connect to Lago at {args.api_url}")
            sys.exit(1)

        print("✓ Connected to Lago")

        # Create billable metrics first (plans depend on them)
        create_billable_metrics(client, args.api_url, args.api_key)

        # Create plans
        create_plans(client, args.api_url, args.api_key)

    print("\n=== Seeding Complete ===")


if __name__ == "__main__":
    main()
