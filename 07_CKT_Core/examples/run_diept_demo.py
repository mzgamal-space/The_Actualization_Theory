"""
run_diept_demo.py — DIEPT Integration Visualization
======================================================
Author : Mohamed Gamal Eldin Abdelaziz Noureldin
         (Conciseness Framework / CKT)
Code   : Antigravity (Advanced Agentic Coding)

Demonstrates how a Question Operator initializes the architecture,
and how DIEPT states resolve into final linguistic outputs based
on phase angle calculations.
"""

from __future__ import annotations
import math
import sys
import os

# Add parent path to import flat modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diept import DIEPTState, QuestionOperatorParser

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

def draw_bar(value: float, max_val: float = 1.0, width: int = 20, color: str = Color.GREEN) -> str:
    filled = int((value / max_val) * width) if max_val > 0 else 0
    filled = min(max(filled, 0), width)
    bar = "=" * filled + "-" * (width - filled)
    return f"{color}[{bar}]{Color.RESET}"

def run_diept_demo():
    print(f"\n{Color.BOLD}{Color.HEADER}=== DIEPT INTEGRATION DEMONSTRATION ==={Color.RESET}\n")
    
    questions = ["Why did the system crash?", "What if we increased the speed of light?"]
    
    for q in questions:
        print(f"{Color.CYAN}{Color.BOLD}USER QUESTION:{Color.RESET} {q}")
        
        # Stage A0: Parse Question Operator
        op = "why" if "why" in q.lower() else "what if"
        domain, primes, theta_t = QuestionOperatorParser.parse_operator(op)
        
        print(f"  {Color.BOLD}[STAGE A0: PRIME SELECTOR]{Color.RESET}")
        print(f"  Mapped Operator  : {op.upper()}")
        print(f"  Target Domain    : {domain}")
        print(f"  Prime Vector     : {primes}")
        print(f"  Target theta (theta) : {theta_t:.2f} rad\n")
        
        # Stage 3: DIEPT Verification
        # Let's mock a thought's DIEPT state for this query
        if op == "why":
            # A factual query should yield a heavily grounded state
            A = [5.0, 5.0, 4.0]
            B = [0.2, 0.1, 0.2]
        else:
            # A speculative query naturally yields high speculation in the network
            A = [1.0, 1.0, 1.0]
            B = [2.0, 3.0, 2.5]
            
        state = DIEPTState(A, B)
        norm_a = state.norm_A
        norm_b = state.norm_B
        theta = state.theta
        
        print(f"  {Color.BOLD}[STAGE 3: DIEPT SUBSPACES]{Color.RESET}")
        print(f"  Grounded (A) Norm    : {norm_a:.4f} {draw_bar(norm_a, max(norm_a, norm_b))}")
        print(f"  Speculative (B) Norm : {norm_b:.4f} {draw_bar(norm_b, max(norm_a, norm_b), color=Color.YELLOW)}")
        
        # Phase Angle vs Target
        color = Color.GREEN if theta <= theta_t else Color.RED
        print(f"  Phase Angle (theta)      : {color}{theta:.4f} rad{Color.RESET} (Target: {theta_t:.4f} rad)")
        
        # Linguistic Modality Mapping
        marker = state.get_linguistic_marker(theta_t)
        print(f"  Linguistic Marker    : {Color.BOLD}{marker}{Color.RESET}")
        
        # Final Output Mock
        mock_output = "the memory limit was exceeded." if op == "why" else "the universe would shrink."
        print(f"  {Color.BOLD}FINAL CKT OUTPUT{Color.RESET}   : {marker} {mock_output}")
        print("-" * 60 + "\n")

if __name__ == "__main__":
    run_diept_demo()
