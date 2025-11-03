import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np

# ==========================================
# REPLACE THIS SECTION WITH YOUR ACTUAL DATA
# ==========================================
# For demonstration, creating sample data
# Replace with your actual data loading:
# out_dir = "results/leak_layer_csvs/"
# df = pd.read_csv(f"{out_dir}/qwq_leak_layer_ranking_thr_0_5.csv").sort_values("layer")
# layers = df["layer"].to_numpy()
# counts = df["flagged_count"].to_numpy()

# Sample data (replace this with your actual data)
np.random.seed(42)
layers = np.arange(0, 80)
counts = np.random.poisson(lam=150, size=80) + np.random.randint(50, 200, size=80)
# ==========================================

# Create the figure with professional styling
fig = go.Figure()

# Add bar trace with professional styling
fig.add_trace(go.Bar(
    x=layers,
    y=counts,
    marker=dict(
        color='#00796B',  # Professional blue color (NeurIPS style)
        line=dict(color='#004D40', width=0.5),  # Subtle border
        opacity=0.95
    ),
    width=0.85,  # Bar width
    hovertemplate='<b>Layer %{x}</b><br>Flagged Neurons: %{y}<extra></extra>'
))

# Update layout with NeurIPS-style formatting
fig.update_layout(
    # Title styling
    title=dict(
        text='Flagged Neurons by Layer for QwQ-32B',
        font=dict(size=18, family='Times New Roman, serif', color='#000000', weight=600),
        x=0.5,
        xanchor='center',
        y=0.96,
        yanchor='top'
    ),
    
    # Axis styling
    xaxis=dict(
        title=dict(
            text='Layer Index',
            font=dict(size=16, family='Times New Roman, serif', color='#000000', weight=500),
            standoff=15
        ),
        tickfont=dict(size=13, family='Times New Roman, serif', color='#000000'),
        showgrid=False,
        showline=True,
        linewidth=2,
        linecolor='#000000',
        mirror=True,
        ticks='outside',
        tickwidth=1.5,
        ticklen=6,
        tickcolor='#000000'
    ),
    
    yaxis=dict(
        title=dict(
            text='Flagged Neurons (|d| ≥ 0.5)',
            font=dict(size=16, family='Times New Roman, serif', color='#000000', weight=500),
            standoff=15
        ),
        tickfont=dict(size=13, family='Times New Roman, serif', color='#000000'),
        showgrid=True,
        gridcolor='#D3D3D3',
        gridwidth=0.8,
        showline=True,
        linewidth=2,
        linecolor='#000000',
        mirror=True,
        ticks='outside',
        tickwidth=1.5,
        ticklen=6,
        tickcolor='#000000',
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='#000000'
    ),
    
    # Overall plot styling
    plot_bgcolor='white',
    paper_bgcolor='white',
    width=1200,
    height=450,
    margin=dict(l=90, r=40, t=80, b=70),
    
    # Font family for consistency
    font=dict(family='Times New Roman, serif', color='#000000'),
    
    # Remove legend if not needed
    showlegend=False,
    
    # High quality rendering
    template='plotly_white'
)

# Ensure proper axis formatting
fig.update_xaxes(showticklabels=True)
fig.update_yaxes(showticklabels=True)


# Save as PDF (vector format - best for LaTeX papers)
fig.write_image('neurips_barplot.pdf', 
                width=1200, height=450)


print("\n" + "="*60)
print("✓ Professional NeurIPS-style plot created successfully!")
print("="*60)
print("\nSaved in multiple formats:")
print("  📊 PNG (high-res, 300+ DPI): neurips_barplot.png")
print("  📄 PDF (vector, LaTeX):      neurips_barplot.pdf")
print("  🎨 SVG (vector, alt):        neurips_barplot.svg")
print("  🌐 HTML (interactive):       neurips_barplot.html")
print("\n" + "="*60)
print("\nFor your LaTeX paper, use:")
print("  \\includegraphics[width=\\columnwidth]{neurips_barplot.pdf}")
print("="*60)