# AI Decision Support Backend — Solar Energy Systems

Backend service for recommending optimal solar-energy configurations using Reinforcement Learning and rule-based decision pipelines.

Built with FastAPI, the system supports multiple real-world scenarios including on-grid, off-grid, hybrid systems, heating, and water pumping.

---

## Overview

This project implements an intelligent decision-support backend that:

- Encodes user inputs into structured system states
- Applies reinforcement learning (TD(0), TD(λ)) for decision optimization
- Combines learned policies with domain-specific constraints
- Returns system recommendations with explanations

The goal is to automate solar system sizing and configuration for practical deployment use cases.

---

## Features

- FastAPI backend for real-time inference
- Multiple recommendation endpoints:
  - On-grid systems
  - Off-grid systems
  - Hybrid systems
  - Solar heating
  - Water pumping
- Reinforcement Learning agents:
  - TD(0)
  - TD(λ) with eligibility traces
- Rule-based constraints integrated with learned policies
- Structured input validation (Pydantic)
- Recommendation explanations (LLM-assisted)

---

## Project Structure
├── main.py # FastAPI entry point
├── model/ # RL agents and decision logic
├── training/ # Training scripts and experiments
├── catalog/ # System components / configurations
├── img/ # Visual assets
├── test.py # Testing scripts
├── pyproject.toml # Dependencies (Poetry)
└── settings.cfg # Configuration

---

## How It Works

1. User input is received via API
2. Input is encoded into a system state
3. RL agent evaluates possible actions
4. Decision is refined with constraints and domain rules
5. Final recommendation is returned with explanation

---

## Example Use Cases

- Solar installation sizing (residential / industrial)
- Off-grid system configuration
- Hybrid energy optimization
- Pumping and heating system recommendations

---

## Tech Stack

- Python
- FastAPI
- NumPy
- Reinforcement Learning (TD methods)
- Pydantic

---

## Notes

This project focuses on **applied AI for real-world decision systems**, combining learning-based approaches with domain constraints.

---

## Author

Yahya Malk
