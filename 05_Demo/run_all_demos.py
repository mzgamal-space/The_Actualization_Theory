"""
run_all_demos.py — Master Runner for All Interactive Demos
===========================================================
Author : Mohamed Gamal Eldin Abdelaziz Noureldin
         Independent Researcher | ORCID: 0009-0006-3991-1153
         Contact: mz.gamal@gmail.com
Pipeline: Actualizer_Engine_FDSA_QCA v1.0.0 (CKT V3_U1)

Runs both interactive demos in sequence:
  1. demo_pipeline.py      — Step-by-step engine breakdown & unified pipeline
  2. demo_qca_parallel.py  — QCA parallel engine & clustered inference

HOW TO RUN
----------
  cd Final_Output/05_Demo
  python run_all_demos.py
"""

import sys
import os
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import demo_pipeline
import demo_qca_parallel

def main():
    print("=" * 74)
    print("  ACTUALIZER ENGINE + FDSA + QCA — MASTER DEMO RUNNER")
    print("  Framework: Computational Knowledge Theory (CKT V3_U1)")
    print("=" * 74)

    print("\n>>> Running Demo 1: Step-by-Step Engine Breakdown & Unified Pipeline...")
    time.sleep(0.5)
    demo_pipeline.run_demo()

    print("\n>>> Running Demo 2: QCA Parallel Engine & Clustered Inference...")
    time.sleep(0.5)
    demo_qca_parallel.run_demo()

    print("=" * 74)
    print("  [COMPLETE] All 05_Demo interactive demonstrations finished successfully!")
    print("=" * 74)

if __name__ == "__main__":
    main()
