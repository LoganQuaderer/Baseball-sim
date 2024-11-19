from typing import Dict, List
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

class PerformanceAnalyzer:
    def __init__(self):
        self.at_bat_history = []
        self.pitcher_stats = {
            'strikeouts': [],
            'walks': [],
            'hits_allowed': [],
            'strike_percentage': [],
            'pitch_type_effectiveness': {},
            'learning_progress': []
        }
        self.batter_stats = {
            'hits': [],
            'at_bats': [],
            'walks': [],
            'learning_progress': []
        }
        self.at_bat_counter = 0
        self.advanced_stats = {
            'exit_velocity': [],
            'launch_angle': [],
            'hard_hit_percentage': [],
            'pitch_types': {},
            'count_performance': {}  # Performance by ball-strike count
        }

    def record_at_bat(self, at_bat_result, pitcher_learning, batter_learning):
        """Record and analyze an at-bat result"""
        self.at_bat_counter += 1
        
        at_bat_data = {
            'at_bat_number': self.at_bat_counter,
            'outcome': at_bat_result.outcome,
            'total_pitches': at_bat_result.total_pitches,
            'balls': at_bat_result.ball_count,
            'strikes': at_bat_result.strike_count,
            'runs_scored': at_bat_result.runs_scored,
            'pitch_sequence': getattr(at_bat_result, 'pitch_sequence', [])
        }
        self.at_bat_history.append(at_bat_data)
        
        self._update_pitcher_stats(at_bat_result)
        self._update_batter_stats(at_bat_result)
        self._update_learning_progress(at_bat_result)

    def _update_pitcher_stats(self, at_bat_result):
        """Update pitcher statistics"""
        if at_bat_result.outcome == 'strikeout':
            self.pitcher_stats['strikeouts'].append(1)
        else:
            self.pitcher_stats['strikeouts'].append(0)
            
        if at_bat_result.outcome == 'walk':
            self.pitcher_stats['walks'].append(1)
        else:
            self.pitcher_stats['walks'].append(0)
            
        if at_bat_result.outcome in ['single', 'double', 'triple', 'home_run']:
            self.pitcher_stats['hits_allowed'].append(1)
        else:
            self.pitcher_stats['hits_allowed'].append(0)

    def _update_batter_stats(self, at_bat_result):
        """Update batter statistics"""
        is_hit = at_bat_result.outcome in ['single', 'double', 'triple', 'home_run']
        is_walk = at_bat_result.outcome == 'walk'
        
        if is_walk:
            self.batter_stats['walks'].append(1)
            self.batter_stats['at_bats'].append(0)
            self.batter_stats['hits'].append(0)
        else:
            self.batter_stats['walks'].append(0)
            self.batter_stats['at_bats'].append(1)
            self.batter_stats['hits'].append(1 if is_hit else 0)

    def _calculate_learning_progress(self, player_type: str, window_size: int = 10) -> float:
        """Calculate learning progress"""
        return 0.1  # Placeholder for now

    def _update_learning_progress(self, at_bat_result):
        """Update learning progress for both players"""
        self.pitcher_stats['learning_progress'].append(0.5 + len(self.at_bat_history) * 0.01)
        self.batter_stats['learning_progress'].append(0.4 + len(self.at_bat_history) * 0.015)

    def plot_learning_progress(self):
        """Plot learning progress with key performance indicators"""
        plt.style.use('default')  # Changed from 'seaborn' to 'default'
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # 1. Learning Progress Plot
        at_bats = range(len(self.at_bat_history))
        ax1.plot(at_bats, self.pitcher_stats['learning_progress'], 
                label='Pitcher', color='blue', linewidth=2)
        ax1.plot(at_bats, self.batter_stats['learning_progress'], 
                label='Batter', color='red', linewidth=2)
        ax1.set_title('Learning Progress')
        ax1.set_xlabel('At Bats')
        ax1.set_ylabel('Success Rate')
        ax1.grid(True)
        ax1.legend()

        # 2. Outcome Distribution
        outcomes = [ab['outcome'] for ab in self.at_bat_history]
        unique_outcomes = sorted(set(outcomes))
        outcome_counts = [outcomes.count(outcome) for outcome in unique_outcomes]
        colors = ['green' if outcome in ['single', 'double', 'triple', 'home_run']
                 else 'red' if outcome == 'strikeout'
                 else 'blue' if outcome == 'walk'
                 else 'gray' for outcome in unique_outcomes]
        
        bars = ax2.bar(unique_outcomes, outcome_counts, color=colors)
        ax2.set_title('At-Bat Outcomes')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add percentage labels
        total = sum(outcome_counts)
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height/total*100:.1f}%',
                    ha='center', va='bottom')

        # 3. Performance by Count
        counts = ['0-0', '1-0', '0-1', '1-1', '2-1', '1-2', '2-2', '3-2']
        pitcher_success = []
        batter_success = []
        
        for count in counts:
            balls, strikes = map(int, count.split('-'))
            relevant_abs = [ab for ab in self.at_bat_history 
                          if ab.get('balls') == balls and ab.get('strikes') == strikes]
            
            if relevant_abs:
                # Pitcher success: strikeouts and outs
                p_success = len([ab for ab in relevant_abs 
                               if ab['outcome'] in ['strikeout', 'out']]) / len(relevant_abs)
                # Batter success: hits and walks
                b_success = len([ab for ab in relevant_abs 
                               if ab['outcome'] in ['single', 'double', 'triple', 'home_run', 'walk']]) / len(relevant_abs)
                
                pitcher_success.append(p_success)
                batter_success.append(b_success)
            else:
                pitcher_success.append(0)
                batter_success.append(0)

        x = np.arange(len(counts))
        width = 0.35
        ax3.bar(x - width/2, pitcher_success, width, label='Pitcher', color='blue', alpha=0.6)
        ax3.bar(x + width/2, batter_success, width, label='Batter', color='red', alpha=0.6)
        ax3.set_title('Success Rate by Count')
        ax3.set_xticks(x)
        ax3.set_xticklabels(counts)
        ax3.legend()
        
        plt.tight_layout()
        plt.show()

    def plot_batting_trend(self):
        """Plot batting performance trends"""
        plt.figure(figsize=(12, 6))
        
        # Calculate rolling averages
        window = 5  # 5 at-bat rolling average
        hits = self.batter_stats['hits']
        at_bats = self.batter_stats['at_bats']
        
        rolling_avg = []
        for i in range(len(hits)):
            start_idx = max(0, i - window + 1)
            period_hits = sum(hits[start_idx:i+1])
            period_abs = sum(at_bats[start_idx:i+1])
            avg = period_hits / period_abs if period_abs > 0 else 0
            rolling_avg.append(avg)

        # Plot batting average trend
        plt.plot(rolling_avg, color='blue', linewidth=2, 
                label=f'{window}-AB Rolling Average')
        
        # Add outcome markers
        for i, ab in enumerate(self.at_bat_history):
            if ab['outcome'] in ['single', 'double', 'triple', 'home_run']:
                plt.scatter(i, rolling_avg[i], color='green', s=100, alpha=0.6)
            elif ab['outcome'] == 'strikeout':
                plt.scatter(i, rolling_avg[i], color='red', s=100, alpha=0.6)
            elif ab['outcome'] == 'walk':
                plt.scatter(i, rolling_avg[i], color='blue', s=100, alpha=0.6)

        plt.title('Batting Performance Trend')
        plt.xlabel('At Bats')
        plt.ylabel('Batting Average')
        plt.grid(True, alpha=0.3)
        
        # Custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', label=f'{window}-AB Rolling Average'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                   label='Hit', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   label='Strikeout', markersize=10),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                   label='Walk', markersize=10)
        ]
        plt.legend(handles=legend_elements)
        plt.show()

    def plot_detailed_statistics(self):
        """Placeholder for detailed statistics plotting"""
        pass

    def _calculate_actual_stats(self) -> Dict[str, Dict]:
        """Calculate actual statistics for pitchers and batters"""
        # Pitcher stats calculation
        total_pitches = sum(len(ab.get('pitch_sequence', [])) for ab in self.at_bat_history)
        pitch_stats = {}
        
        # Process each at-bat for detailed stats
        for ab in self.at_bat_history:
            for pitch in ab.get('pitch_sequence', []):
                pitch_type = pitch.get('pitch_type')
                if pitch_type not in pitch_stats:
                    pitch_stats[pitch_type] = {
                        'total': 0,
                        'whiffs': 0,
                        'strikes': 0,
                        'velocity_sum': 0,
                        'balls_in_play': 0
                    }
                
                stats = pitch_stats[pitch_type]
                stats['total'] += 1
                stats['velocity_sum'] += pitch.get('velocity', 0)
                
                if pitch.get('result') == 'swinging_strike':
                    stats['whiffs'] += 1
                if 'strike' in pitch.get('result', ''):
                    stats['strikes'] += 1
                if pitch.get('result') in ['single', 'double', 'triple', 'home_run', 'out']:
                    stats['balls_in_play'] += 1

        # Calculate actual pitch stats
        calculated_pitch_stats = {}
        for pitch_type, stats in pitch_stats.items():
            calculated_pitch_stats[pitch_type] = {
                'usage_rate': stats['total'] / total_pitches if total_pitches > 0 else 0,
                'whiff_rate': stats['whiffs'] / stats['total'] if stats['total'] > 0 else 0,
                'strike_rate': stats['strikes'] / stats['total'] if stats['total'] > 0 else 0,
                'avg_velocity': stats['velocity_sum'] / stats['total'] if stats['total'] > 0 else 0
            }

        # Calculate batter vs pitch type stats
        batter_pitch_stats = {}
        for pitch_type in pitch_stats.keys():
            abs_with_pitch = [ab for ab in self.at_bat_history 
                            if any(p.get('pitch_type') == pitch_type 
                                for p in ab.get('pitch_sequence', []))]
            
            hits = len([ab for ab in abs_with_pitch 
                       if ab['outcome'] in ['single', 'double', 'triple', 'home_run']])
            total_abs = len(abs_with_pitch)
            
            batter_pitch_stats[pitch_type] = {
                'avg': hits / total_abs if total_abs > 0 else 0,
                'slg': self._calculate_slugging_vs_pitch(abs_with_pitch),
                'whiff_rate': pitch_stats[pitch_type]['whiffs'] / pitch_stats[pitch_type]['total']
                             if pitch_stats[pitch_type]['total'] > 0 else 0
            }

        return {
            'pitch_stats': calculated_pitch_stats,
            'batter_vs_pitch': batter_pitch_stats
        }

    def _calculate_slugging_vs_pitch(self, at_bats: List[Dict]) -> float:
        """Calculate slugging percentage against specific pitch"""
        if not at_bats:
            return 0.0
        total_bases = sum(
            4 if ab['outcome'] == 'home_run' else
            3 if ab['outcome'] == 'triple' else
            2 if ab['outcome'] == 'double' else
            1 if ab['outcome'] == 'single' else 0
            for ab in at_bats
        )
        return total_bases / len(at_bats)

    def _calculate_quality_of_abs(self) -> float:
        """Calculate real quality of at-bats metric"""
        if not self.at_bat_history:
            return 0.0
        quality_scores = []
        for ab in self.at_bat_history:
            # Base score for number of pitches (rewards working the count)
            pitch_count = len(ab.get('pitch_sequence', []))
            base_score = min(1.0, pitch_count / 6.0)
            
            # Bonus for positive outcomes
            outcome_bonus = {
                'home_run': 1.0,
                'triple': 0.8,
                'double': 0.6,
                'single': 0.4,
                'walk': 0.3,
                'strikeout': -0.2,
                'out': 0.0
            }.get(ab['outcome'], 0.0)
            
            # Bonus for battling with two strikes
            two_strike_bonus = 0.2 if ab.get('strikes', 0) == 2 else 0.0
            
            # Calculate final quality score
            quality_score = base_score + outcome_bonus + two_strike_bonus
            quality_scores.append(max(0.0, min(1.0, quality_score)))
            
        return sum(quality_scores) / len(quality_scores)

    def _calculate_situational_stats(self, situation: str) -> float:
        """Calculate actual situational statistics"""
        if not self.at_bat_history:
            return 0.0
        
        relevant_abs = []
        for ab in self.at_bat_history:
            balls = ab.get('balls', 0)
            strikes = ab.get('strikes', 0)
            
            if situation == 'ahead' and balls < strikes:
                relevant_abs.append(ab)
            elif situation == 'behind' and balls > strikes:
                relevant_abs.append(ab)
            elif situation == 'two_strikes' and strikes == 2:
                relevant_abs.append(ab)
        
        if not relevant_abs:
            return 0.0
        
        success_abs = [ab for ab in relevant_abs if ab['outcome'] in 
                      (['strikeout', 'out'] if situation == 'pitcher_success' else
                      ['single', 'double', 'triple', 'home_run', 'walk'])]
        
        return len(success_abs) / len(relevant_abs)

    def _calculate_hard_hit_rate(self):
        """Calculate rate of hard-hit balls"""
        if not self.at_bat_history:
            return 0.0
            
        contact_abs = [ab for ab in self.at_bat_history if ab['outcome'] in 
                      ['single', 'double', 'triple', 'home_run', 'out']]
        if not contact_abs:
            return 0.0
        
        hard_hits = sum(1 for ab in contact_abs if ab['outcome'] in 
                       ['double', 'triple', 'home_run'])
        return hard_hits / len(contact_abs)

    def _calculate_contact_rate(self):
        """Calculate overall contact rate"""
        if not self.at_bat_history:
            return 0.0
        
        total_swings = 0
        contact = 0
        for ab in self.at_bat_history:
            for pitch in ab.get('pitch_sequence', []):
                if pitch.get('swing', False):
                    total_swings += 1
                    if pitch['result'] not in ['swinging_strike']:
                        contact += 1
        
        return contact / total_swings if total_swings > 0 else 0.0

    def _calculate_chase_rate(self):
        """Calculate rate of swinging at pitches outside zone"""
        if not self.at_bat_history:
            return 0.0
        
        outside_pitches = 0
        chases = 0
        for ab in self.at_bat_history:
            for pitch in ab.get('pitch_sequence', []):
                # Check if location exists before accessing it
                if pitch.get('location', 'strike') == 'ball':
                    outside_pitches += 1
                    if pitch.get('swing', False):
                        chases += 1
        
        return chases / outside_pitches if outside_pitches > 0 else 0.0

    def _calculate_zone_contact(self):
        """Calculate contact rate on pitches in zone"""
        if not self.at_bat_history:
            return 0.0
        
        zone_swings = 0
        zone_contact = 0
        for ab in self.at_bat_history:
            for pitch in ab.get('pitch_sequence', []):
                if pitch.get('location', 'strike') == 'strike' and pitch.get('swing', False):
                    zone_swings += 1
                    if pitch.get('result') not in ['swinging_strike']:
                        zone_contact += 1
        
        return zone_contact / zone_swings if zone_swings > 0 else 0.0

    def _calculate_batter_advantage(self):
        """Calculate batter's overall advantage in matchup"""
        if not self.at_bat_history:
            return 0.5
        
        success = sum(1 for ab in self.at_bat_history if ab['outcome'] in 
                     ['single', 'double', 'triple', 'home_run', 'walk'])
        return success / len(self.at_bat_history)

    def _calculate_pitcher_advantage(self):
        """Calculate pitcher's overall advantage in matchup"""
        if not self.at_bat_history:
            return 0.5
        
        success = sum(1 for ab in self.at_bat_history if ab['outcome'] in 
                     ['strikeout', 'out'])
        return success / len(self.at_bat_history)

    def _calculate_recognition_improvement(self):
        """Calculate improvement in pitch recognition"""
        if len(self.at_bat_history) < 10:
            return 0.0
        
        early_abs = self.at_bat_history[:5]
        recent_abs = self.at_bat_history[-5:]
        
        def calculate_recognition_rate(abs_list):
            good_decisions = 0
            total_decisions = 0
            for ab in abs_list:
                for pitch in ab.get('pitch_sequence', []):
                    # Skip if location data is missing
                    if 'location' not in pitch:
                        continue
                    
                    total_decisions += 1
                    if (pitch['location'] == 'ball' and not pitch.get('swing', False)) or \
                       (pitch['location'] == 'strike' and pitch.get('swing', False)):
                        good_decisions += 1
            return good_decisions / total_decisions if total_decisions > 0 else 0
        
        early_rate = calculate_recognition_rate(early_abs)
        recent_rate = calculate_recognition_rate(recent_abs)
        
        return recent_rate - early_rate

    def _calculate_learning_trend(self, values: List[float]) -> float:
        """Calculate learning trend using linear regression"""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        y = np.array(values)
        coeffs = np.polyfit(x, y, 1)
        return coeffs[0]  # Return slope of trend line

    def generate_report(self):
        """Generate a comprehensive performance report"""
        print("Starting generate_report...")
        try:
            print("Calculating stats...")
            stats = self._calculate_actual_stats()
            print("Stats calculated successfully")
            
            hits = sum(self.batter_stats['hits'])
            at_bats = sum(self.batter_stats['at_bats'])
            print(f"Debug - Hits: {hits}, At bats: {at_bats}")
            
            report = {
                'total_at_bats': len(self.at_bat_history),
                'batter_performance': {
                    'batting_avg': hits / max(1, at_bats),
                    'walks': sum(self.batter_stats['walks']),
                    'quality_of_abs': self._calculate_quality_of_abs(),
                    'hard_hit_rate': self._calculate_hard_hit_rate(),
                    'contact_rate': self._calculate_contact_rate(),
                    'chase_rate': self._calculate_chase_rate(),
                    'zone_contact': self._calculate_zone_contact()
                },
                'pitcher_performance': {
                    'strikeouts': sum(self.pitcher_stats['strikeouts']),
                    'walks': sum(self.pitcher_stats['walks']),
                    'hits_allowed': sum(self.pitcher_stats['hits_allowed']),
                    'pitch_stats': stats['pitch_stats'],
                    'fip': self._calculate_fip()
                },
                'matchup_analysis': {
                    'batter_advantage': self._calculate_batter_advantage(),
                    'pitcher_advantage': self._calculate_pitcher_advantage(),
                    'recognition_improvement': self._calculate_recognition_improvement(),
                    'batter_vs_pitch': stats['batter_vs_pitch']
                }
            }
            print("Report generated successfully")
            print(f"Report keys: {report.keys()}")
            return report
            
        except Exception as e:
            print(f"Error in generate_report: {str(e)}")
            print("Full traceback:")
            import traceback
            traceback.print_exc()
            
            # Create a basic report with just the essential stats
            basic_report = {
                'total_at_bats': len(self.at_bat_history),
                'batter_performance': {
                    'batting_avg': sum(self.batter_stats['hits']) / max(1, sum(self.batter_stats['at_bats'])),
                    'walks': sum(self.batter_stats['walks'])
                },
                'pitcher_performance': {
                    'fip': self._calculate_fip()
                }
            }
            print(f"Returning basic report with keys: {basic_report.keys()}")
            return basic_report

    def _calculate_fip(self):
        """Calculate FIP (Fielding Independent Pitching)"""
        innings = len(self.at_bat_history) / 3  # Approximate innings pitched
        if innings == 0:
            return 0.0
            
        hr = sum(1 for ab in self.at_bat_history if ab['outcome'] == 'home_run')
        bb = sum(self.pitcher_stats['walks'])
        k = sum(self.pitcher_stats['strikeouts'])
        
        # FIP constant is typically around 3.10
        fip_constant = 3.10
        
        # FIP formula: ((13*HR + 3*BB - 2*K) / IP) + FIP_constant
        fip = ((13 * hr + 3 * bb - 2 * k) / innings) + fip_constant
        return fip

    def enhanced_visualization(self):
        # Zone charts
        self.plot_heat_map()
        
        # Performance trends
        self.plot_performance_over_time()
        
        # Pitch sequence analysis
        self.plot_pitch_sequences()

print(plt.style.available)