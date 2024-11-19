import os
import sys
import numpy as np
from datetime import datetime
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.models.pitcher import Pitcher
from src.models.batter import Batter, BatterAttributes
from src.models.simulator import AtBatSimulator
from src.utils.performance_analyzer import PerformanceAnalyzer

def print_boxed(text, width=70):
    print("╔" + "═" * width + "╗")
    print("║" + text.center(width) + "║")
    print("╚" + "═" * width + "╝")

def run_detailed_simulation():
    # Create players
    pitcher = Pitcher(
        name="Max Scherzer",
        throws="R",
        velocity=95.0,
        control=85.0,
        movement=90.0,
        stamina=100.0,
        pitch_types={
            "4-Seam Fastball": 90,
            "Slider": 85,
            "Changeup": 75,
            "Curveball": 70,
        }
    )
    
    batter = Batter(
        name="Mike Trout",
        bats="R",
        attributes=BatterAttributes(
            contact=90.0,
            power=95.0,
            eye=90.0,
            speed=85.0
        )
    )
    
    # Create simulator and analyzer
    simulator = AtBatSimulator(pitcher, batter)
    analyzer = PerformanceAnalyzer()
    
    print(f"\nSimulating at-bats between {pitcher.name} and {batter.name}")
    print("=" * 70)
    
    # Simulate 50 at-bats
    for i in range(50):
        result = simulator.simulate_at_bat()
        analyzer.record_at_bat(result, simulator.pitcher_learning, simulator.batter_learning)
        print(f"At-bat #{i+1}: {result.outcome}")
    
    # Generate the reports
    report = analyzer.generate_report()
    
    # Display text-based statistics
    print("\nFinal Statistics:")
    print(f"Batting Average: {report['batter_performance']['batting_avg']:.3f}")
    print(f"Pitcher ERA: {report['pitcher_performance']['fip']:.2f}")
    
    # Display visualizations
    print("\nGenerating performance visualizations...")
    analyzer.plot_learning_progress()  # This will show the first set of plots
    
    print("\nGenerating batting trend visualization...")
    analyzer.plot_batting_trend()      # This will show the batting trend plot

if __name__ == "__main__":
    run_detailed_simulation()