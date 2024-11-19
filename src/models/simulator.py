from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
from sklearn.neural_network import MLPClassifier
from .pitcher import Pitcher
from .batter import Batter, BatterAttributes

@dataclass
class GameState:
    inning: int = 1
    outs: int = 0
    balls: int = 0
    strikes: int = 0
    score: int = 0
    runners: List[str] = None
    pitcher_pitch_count: int = 0
    previous_pitch_types: List[str] = None

    def __post_init__(self):
        if self.runners is None:
            self.runners = [None, None, None]
        if self.previous_pitch_types is None:
            self.previous_pitch_types = []

    def add_hit(self, batter_name: str, hit_type: str) -> int:
        runs_scored = 0
        if hit_type == 'home_run':
            runs_scored = 1 + sum(1 for runner in self.runners if runner is not None)
            self.runners = [None, None, None]
        elif hit_type == 'triple':
            runs_scored = sum(1 for runner in self.runners if runner is not None)
            self.runners = [None, None, batter_name]
        elif hit_type == 'double':
            runs_scored = sum(1 for runner in self.runners[1:] if runner is not None)
            self.runners = [None, batter_name, self.runners[0] if self.runners[0] else None]
        elif hit_type == 'single':
            if self.runners[2]:
                runs_scored += 1
            if self.runners[1]:
                self.runners[2] = self.runners[1]
            if self.runners[0]:
                self.runners[1] = self.runners[0]
            self.runners[0] = batter_name
        
        self.score += runs_scored
        return runs_scored

@dataclass
class AtBatResult:
    outcome: str
    pitch_sequence: List[Dict]
    total_pitches: int
    ball_count: int
    strike_count: int
    runs_scored: int = 0
    runners_before: List[str] = None
    runners_after: List[str] = None

@dataclass
class PitcherLearning:
    pitch_outcomes: List[Dict] = None
    batter_weaknesses: Dict[str, float] = None
    successful_sequences: List[List[str]] = None
    model: MLPClassifier = None

    def __post_init__(self):
        if self.pitch_outcomes is None:
            self.pitch_outcomes = []
        if self.batter_weaknesses is None:
            self.batter_weaknesses = {}
        if self.successful_sequences is None:
            self.successful_sequences = []
        if self.model is None:
            self.model = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000)

@dataclass
class BatterLearning:
    pitcher_tendencies: Dict[str, Dict] = None
    pitch_recognition: Dict[str, float] = None
    swing_outcomes: List[Dict] = None
    model: MLPClassifier = None

    def __post_init__(self):
        if self.pitcher_tendencies is None:
            self.pitcher_tendencies = {}
        if self.pitch_recognition is None:
            self.pitch_recognition = {}
        if self.swing_outcomes is None:
            self.swing_outcomes = []
        if self.model is None:
            self.model = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000)

class AtBatSimulator:
    def __init__(self, pitcher: Pitcher, batter: Batter):
        self.pitcher = pitcher
        self.batter = batter
        self.game_state = GameState()
        self.pitcher_learning = PitcherLearning()
        self.batter_learning = BatterLearning()
        self.pitch_sequence = []

    def simulate_at_bat(self) -> AtBatResult:
        self.game_state.balls = 0
        self.game_state.strikes = 0
        self.pitch_sequence = []
        runners_before = self.game_state.runners.copy()
        runs_scored = 0

        while True:
            situation = self._get_game_situation()
            pitch_type = self._select_pitch_type()
            pitch_info = self._generate_pitch()
            swing = self._decide_swing(pitch_info, situation)

            pitch_info['swing'] = swing
            result = self._process_pitch_result(pitch_info, swing)

            pitch_data = {
                'pitch_type': pitch_type,
                'result': result,
                'velocity': pitch_info['velocity'],
                'swing': swing
            }
            self.pitch_sequence.append(pitch_data)

            if result == 'ball':
                self.game_state.balls += 1
            elif 'strike' in result:
                self.game_state.strikes += 1

            if self._is_at_bat_over(result):
                if result in ['single', 'double', 'triple', 'home_run']:
                    runs_scored = self.game_state.add_hit(self.batter.name, result)
                break

            self.game_state.pitcher_pitch_count += 1
            self.pitcher.stamina = max(0, self.pitcher.stamina - 0.1)

        final_outcome = self._determine_outcome(self.pitch_sequence[-1]['result'])
        
        result = AtBatResult(
            outcome=final_outcome,
            pitch_sequence=self.pitch_sequence,
            total_pitches=len(self.pitch_sequence),
            ball_count=self.game_state.balls,
            strike_count=self.game_state.strikes,
            runs_scored=runs_scored,
            runners_before=runners_before,
            runners_after=self.game_state.runners.copy()
        )

        self._update_pitcher_learning(result)
        self._update_batter_learning(result)

        return result

    def _get_game_situation(self) -> Dict:
        return {
            'count': (self.game_state.balls, self.game_state.strikes),
            'stamina': self.pitcher.stamina,
            'runners': self.game_state.runners,
            'outs': self.game_state.outs,
            'inning': self.game_state.inning
        }

    def _select_pitch_type(self) -> str:
        if len(self.pitcher_learning.pitch_outcomes) < 10:
            pitch_probabilities = []
            pitch_types = []
            
            total_confidence = sum(self.pitcher.pitch_types.values())
            for pitch, confidence in self.pitcher.pitch_types.items():
                pitch_types.append(pitch)
                pitch_probabilities.append(confidence / total_confidence)
                
            return np.random.choice(pitch_types, p=pitch_probabilities)
        
        features = [
            self.game_state.balls,
            self.game_state.strikes,
            len(self.pitcher_learning.pitch_outcomes),
            self.batter.attributes.contact,
            self.batter.attributes.power,
            self.batter.attributes.eye
        ]
        
        try:
            best_pitch = None
            best_prob = -1
            for pitch_type in self.pitcher.pitch_types:
                prob = self.pitcher_learning.model.predict_proba([features])[0][1]
                if prob > best_prob:
                    best_prob = prob
                    best_pitch = pitch_type
            return best_pitch if best_pitch else list(self.pitcher.pitch_types.keys())[0]
        except:
            return list(self.pitcher.pitch_types.keys())[0]

    def _generate_pitch(self) -> Dict:
        current_velocity = self.pitcher.velocity * (self.pitcher.stamina / 100)
        current_velocity += np.random.normal(0, 1.5)
        
        control_factor = self.pitcher.control / 100 * (self.pitcher.stamina / 100)
        is_strike = np.random.random() < control_factor
        
        return {
            'velocity': current_velocity,
            'location': 'strike' if is_strike else 'ball'
        }

    def _decide_swing(self, pitch_info: Dict, situation: Dict) -> bool:
        if len(self.batter_learning.swing_outcomes) < 10:
            swing_probability = 0.5
            
            if situation['count'][1] == 2:  # Two strikes
                swing_probability += 0.2
            elif situation['count'][0] == 3:  # Three balls
                swing_probability -= 0.1
                
            swing_probability *= (self.batter.attributes.eye / 100)
            
            if pitch_info['location'] == 'strike':
                swing_probability += 0.2
                
            return np.random.random() < swing_probability
        
        features = [
            situation['count'][0],
            situation['count'][1],
            self.pitcher.velocity,
            self.pitcher.control,
            list(self.pitcher.pitch_types.values()).index(
                max(self.pitcher.pitch_types.values())
            )
        ]
        
        try:
            swing_probability = self.batter_learning.model.predict_proba([features])[0][1]
            pitch_type = pitch_info.get('pitch_type', '')
            if pitch_type in self.batter_learning.pitch_recognition:
                swing_probability *= self.batter_learning.pitch_recognition[pitch_type]
            return np.random.random() < swing_probability
        except:
            return np.random.random() < 0.5

    def _process_pitch_result(self, pitch_info: Dict, swing: bool) -> str:
        if not swing:
            return 'ball' if pitch_info['location'] == 'ball' else 'called_strike'
        
        contact_probability = self.batter.attributes.contact / 100
        if pitch_info['location'] == 'ball':
            contact_probability *= 0.6
            
        if np.random.random() > contact_probability:
            return 'swinging_strike'
            
        return self._generate_contact_result()

    def _generate_contact_result(self) -> str:
        hit_probability = (self.batter.attributes.contact + self.batter.attributes.power) / 200
        result_roll = np.random.random()
        
        power_factor = self.batter.attributes.power / 100
        if result_roll < 0.1 * power_factor:
            return 'home_run'
        elif result_roll < 0.2 * power_factor:
            return 'double'
        elif result_roll < hit_probability:
            return 'single'
        return 'out'

    def _is_at_bat_over(self, result: str) -> bool:
        if result in ['single', 'double', 'triple', 'home_run', 'out']:
            return True
        if self.game_state.strikes >= 3:
            return True
        if self.game_state.balls >= 4:
            return True
        return False

    def _determine_outcome(self, last_result: str) -> str:
        if self.game_state.strikes >= 3:
            return 'strikeout'
        elif self.game_state.balls >= 4:
            return 'walk'
        else:
            return last_result

    def _update_pitcher_learning(self, at_bat_result: AtBatResult):
        X = []
        y = []
        
        for pitch in at_bat_result.pitch_sequence:
            features = [
                self.game_state.balls,
                self.game_state.strikes,
                len(self.pitcher_learning.pitch_outcomes),
                self.batter.attributes.contact,
                self.batter.attributes.power,
                self.batter.attributes.eye
            ]
            X.append(features)
            y.append(1 if pitch['result'] in ['swinging_strike', 'called_strike', 'out'] else 0)
            
            self.pitcher_learning.pitch_outcomes.append({
                'pitch_type': pitch['pitch_type'],
                'result': pitch['result'],
                'count': (self.game_state.balls, self.game_state.strikes)
            })
        
        if len(X) > 10:
            try:
                self.pitcher_learning.model.partial_fit(X, y, classes=[0, 1])
            except ValueError:
                self.pitcher_learning.model.fit(X, y)

    def _update_batter_learning(self, at_bat_result: AtBatResult):
        X = []
        y = []
        
        for pitch in at_bat_result.pitch_sequence:
            features = [
                self.game_state.balls,
                self.game_state.strikes,
                self.pitcher.velocity,
                self.pitcher.control,
                list(self.pitcher.pitch_types.values()).index(
                    max(self.pitcher.pitch_types.values())
                )
            ]
            X.append(features)
            y.append(1 if pitch['result'] in ['single', 'double', 'triple', 'home_run'] else 0)
            
            pitch_type = pitch['pitch_type']
            if pitch_type not in self.batter_learning.pitch_recognition:
                self.batter_learning.pitch_recognition[pitch_type] = 0.5
            
            if (not pitch['swing'] and pitch['result'] == 'ball') or \
               (pitch['swing'] and 'hit' in pitch['result']):
                self.batter_learning.pitch_recognition[pitch_type] += 0.01
            
        if len(X) > 10:
            try:
                self.batter_learning.model.partial_fit(X, y, classes=[0, 1])
            except ValueError:
                self.batter_learning.model.fit(X, y)