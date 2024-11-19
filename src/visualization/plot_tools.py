import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

class BaseballVisualizer:
    def __init__(self):
        self.strike_zone_height = (1.5, 3.5)  # feet
        self.strike_zone_width = (-0.83, 0.83)  # feet
        
    def plot_pitch_location(self, pitches_df: pd.DataFrame):
        """Create strike zone plot with pitch locations"""
        fig = go.Figure()
        
        # Add strike zone
        fig.add_shape(
            type="rect",
            x0=self.strike_zone_width[0],
            y0=self.strike_zone_height[0],
            x1=self.strike_zone_width[1],
            y1=self.strike_zone_height[1],
            line=dict(color="black"),
            fillcolor="rgba(0,0,0,0)"
        )
        
        # Add pitches
        fig.add_scatter(
            x=pitches_df['location_x'],
            y=pitches_df['location_z'],
            mode='markers',
            marker=dict(
                size=10,
                color=pitches_df['velocity'],
                colorscale='Viridis',
                showscale=True
            ),
            text=pitches_df['pitch_type'],
            hovertemplate=(
                "Pitch Type: %{text}<br>" +
                "Velocity: %{marker.color:.1f} mph<br>" +
                "Location: (%{x:.2f}, %{y:.2f})<br>"
            )
        )
        
        fig.update_layout(
            title="Pitch Location Plot",
            xaxis_title="Horizontal Location (ft)",
            yaxis_title="Height (ft)",
            showlegend=False
        )
        
        return fig
    
    def plot_outcome_distribution(self, results_df: pd.DataFrame):
        """Plot distribution of at-bat outcomes"""
        fig = px.pie(
            results_df,
            names='result',
            title='At-Bat Outcome Distribution'
        )
        return fig
    
    def plot_pitch_sequence(self, pitcher_df: pd.DataFrame):
        """Plot pitch sequence patterns"""
        fig = px.line(
            pitcher_df,
            x='pitch_number',
            y='velocity',
            color='pitch_type',
            title='Pitch Sequence Analysis'
        )
        return fig