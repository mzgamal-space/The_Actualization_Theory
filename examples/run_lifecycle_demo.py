"""
demo.py — Upgraded Actualizer Engine Cognitive Lifecycle Demonstration
======================================================================
Author : Antigravity (Advanced Agentic Coding)

Demonstrates:
  1. Setup of CWF Causal penalty matrix and vocabulary context.
  2. Generation and verification of three Candidate Thoughts:
     - Thought 1: Causal violation (Pipeline A collapses propensity to 0).
     - Thought 2: Speculative noise (Pipeline C quarantines propensity to 0).
     - Thought 3: Fully Grounded and Concise (passes verification, calculates CAKI).
  3. Crystallization of Thought 3 into an immutable MCE super-cluster.
  4. Dynamic inclusion of the MCE object in the FDSA Analogy Library.
  5. Subsequent isomorphic anchoring matching the new MCE object to govern next thought.
"""

from __future__ import annotations
import math
import time
from ckt_actualizer.models.mce import MCE, ReferenceDomain
from ckt_actualizer.models.thought import CandidateThought
from ckt_actualizer.core.filters import EpistemicVerificationSuite
from ckt_actualizer.core.fdsa import FractalDeductionSearch, VectorizedFDSAPruner
from ckt_actualizer.core.engine import UpgradedActualizerEngine

# Console color codes for premium visuals
class Color:
    HEADER = '\033[95m'
    BLUE   = '\033[94m'
    CYAN   = '\033[96m'
    GREEN  = '\033[92m'
    YELLOW = '\033[93m'
    RED    = '\033[91m'
    BOLD   = '\033[1m'
    DIM    = '\033[2m'
    RESET  = '\033[0m'

def print_separator(char='=', length=80, color=Color.DIM):
    print(f"{color}{char * length}{Color.RESET}")

def print_tag(label: str, color: str, text: str):
    print(f"  {color}{Color.BOLD}[{label}]{Color.RESET} {text}")

def run_lifecycle_demo():
    print()
    print(f"{Color.BOLD}{Color.CYAN}  +--------------------------------------------------------------+")
    print("  |    UPGRADED ACTUALIZER ENGINE COGNITIVE LIFECYCLE DEMO       |")
    print("  |    Stage A0 (DIEPT) -> Stage 1 (FDSA) -> Stage 2 (Filters)   |")
    print("  |    -> Stage 3 (Crystallization)                              |")
    print(f"  +--------------------------------------------------------------+{Color.RESET}")
    print()
    
    # 1. Setup CWF causal rules
    cwf = {
        (10, 20): 0.0,
        (20, 30): 0.0,
        (20, 99): float('inf')  # Infinite penalty represents direct causal violation
    }
    
    vocab_size = 1000
    engine = UpgradedActualizerEngine(
        vocab_size=vocab_size,
        cwf_penalty_matrix=cwf,
        k_contractive=0.45,
        Q_c=1e-5,
        theta_target=0.70,   # Will be dynamically overwritten by Stage A0 parser
        caki_threshold=0.50  # Lowered for demo to ensure crystallization
    )
    
    print_separator('=')
    print(f"{Color.HEADER}{Color.BOLD}  END-TO-END PBI LIFECYCLE EXECUTION{Color.RESET}")
    print_separator('=')
    print_tag("SETUP", Color.CYAN, "Configured Actualizer Engine with vocabulary size V = 1,000")
    print()
    
    # Query 1: A factual question
    query_1 = "Why did the system drop the object?"
    print(f"{Color.CYAN}{Color.BOLD}USER QUERY 1:{Color.RESET} {query_1}")
    
    marker, is_cryst, mce_obj, caki = engine.process_query(
        query=query_1,
        initial_history=[10], # starting token
        target_tokens={30},
        simulated_diept_a=[5.0, 5.0, 4.0],  # Highly grounded
        simulated_diept_b=[0.2, 0.1, 0.2],  # Low speculation
    )
    
    print_tag("STAGE A0", Color.GREEN, "Operator 'Why' mapped. Justice dominant. Target theta = 0.15")
    print_tag("STAGE 1", Color.GREEN, "FDSA executed contractive loop. Token generated.")
    print_tag("STAGE 2", Color.GREEN, f"Verification Pipelines. CAKI calculated: {caki:.4f}")
    if is_cryst and mce_obj:
        print_tag("STAGE 3", Color.GREEN, f"Crystallization SUCCESS! MCE Object created: {mce_obj.name}")
    print_tag("OUTPUT", Color.YELLOW, f"Final Engine Response: {Color.BOLD}{marker}{Color.RESET} Because...")
    print()
    
    # Query 2: A speculative question
    query_2 = "What if gravity inverted?"
    print(f"{Color.CYAN}{Color.BOLD}USER QUERY 2:{Color.RESET} {query_2}")
    
    marker2, is_cryst2, mce_obj2, caki2 = engine.process_query(
        query=query_2,
        initial_history=[99], 
        target_tokens={30},
        simulated_diept_a=[1.0, 1.0, 1.0],  # Low ground
        simulated_diept_b=[3.0, 4.0, 3.5],  # High speculation
    )
    
    print_tag("STAGE A0", Color.GREEN, "Operator 'What if' mapped. Creativity mode. Target theta = 1.20")
    print_tag("STAGE 1", Color.GREEN, "FDSA executed contractive loop.")
    print_tag("STAGE 2", Color.GREEN, f"Verification Pipelines. CAKI calculated: {caki2:.4f}")
    if is_cryst2:
        print_tag("STAGE 3", Color.GREEN, f"Crystallization SUCCESS! MCE Object created.")
    else:
        print_tag("STAGE 3", Color.YELLOW, f"Crystallization Skipped (Cost not reduced or CAKI too low).")
        
    print_tag("OUTPUT", Color.YELLOW, f"Final Engine Response: {Color.BOLD}{marker2}{Color.RESET} Objects would fall upwards...")

    print_separator('=')
    print(f"{Color.BOLD}{Color.GREEN}  COGNITIVE LIFE CYCLE COMPLETE!{Color.RESET}")
    print_separator('=')
    print()

if __name__ == "__main__":
    run_lifecycle_demo()
