import plotly.graph_objects as go
from pathlib import Path


def plot_cpl_vanilla_vs_salt() -> None:
    """
    Display a publication-quality bar chart comparing CPL for Vanilla vs SALT.
    
    - X-axis: Each model
    - Two bars per model: Vanilla and SALT
    - Y axis: 0–100% (percentage scale)
    - NeurIPS-style formatting with patterns
    """

    model_names = [
        "Llama-3.1-8B",
        "QwQ-32B",
        "DeepSeek-R1-Distill-<br>Qwen-1.5B",  # Line break for better spacing
    ]

    # CPL values (lower is better)
    vanilla_cpl = [0.385, 0.727, 0.077]
    salt_cpl = [0.316, 0.595, 0.053]
    std_error_vanilla = [0.007, 0.008, 0.004]
    std_error_salt = [0.008, 0.008, 0.003]

    # Convert to percentages for display
    vanilla_cpl_pct = [v * 100 for v in vanilla_cpl]
    salt_cpl_pct = [s * 100 for s in salt_cpl]
    std_error_vanilla_pct = [e * 100 for e in std_error_vanilla]
    std_error_salt_pct = [e * 100 for e in std_error_salt]

    # NeurIPS-style colors (professional and colorblind-friendly)
    vanilla_color = "rgb(189, 189, 189)"   # Light grey
    salt_color = "rgb(0, 121, 107)"         # Teal #00796B

    fig = go.Figure()

    # Add Vanilla bars with dot pattern
    fig.add_trace(go.Bar(
        name='Vanilla',
        x=model_names,
        y=vanilla_cpl_pct,
        error_y=dict(
            type='data',
            array=std_error_vanilla_pct,
            visible=True,
            color='black',
            thickness=4,
            width=10,
        ),
        marker=dict(
            color=vanilla_color,
            pattern=dict(
                shape='.',  # Dot pattern
                size=14,
                solidity=0.5,
            ),
            line=dict(color='black', width=2.5),
        ),
        text=[f'{v:.1f}%' for v in vanilla_cpl_pct],
        textposition='outside',
        textfont=dict(size=28, color='black', family='Times New Roman', weight='bold'),
        width=0.35,
    ))

    # Add SALT bars with plus pattern
    fig.add_trace(go.Bar(
        name='SALT',
        x=model_names,
        y=salt_cpl_pct,
        error_y=dict(
            type='data',
            array=std_error_salt_pct,
            visible=True,
            color='black',
            thickness=4,
            width=10,
        ),
        marker=dict(
            color=salt_color,
            pattern=dict(
                shape='+',  # Plus/cross pattern
                size=14,
                solidity=0.5,
            ),
            line=dict(color='black', width=2.5),
        ),
        text=[f'{s:.1f}%' for s in salt_cpl_pct],
        textposition='outside',
        textfont=dict(size=28, color='black', family='Times New Roman', weight='bold'),
        width=0.35,
    ))

    # Update layout with NeurIPS-style formatting
    fig.update_layout(
        # Figure size - much larger
        width=1400,
        height=800,
        
        # Title
        title=dict(
            text='Contextual Privacy Leakage (CPL) Before and After SALT Defense',
            font=dict(size=36, color='black', family='Times New Roman', weight='bold'),
            x=0.5,
            xanchor='center',
            y=0.97,
            yanchor='top',
        ),
        
        # Axes
        xaxis=dict(
            title=dict(
                text='Models',
                font=dict(size=32, color='black', family='Times New Roman', weight='bold'),
            ),
            tickfont=dict(size=30, color='black', family='Times New Roman'),
            showgrid=False,
            showline=True,
            linewidth=3,
            linecolor='black',
            mirror=False,
            ticks='',  # Remove tick marks
            tickwidth=3,
            ticklen=0,
        ),
        yaxis=dict(
            title=dict(
                text='Contextual Privacy Leakage (CPL)',
                font=dict(size=32, color='black', family='Times New Roman', weight='bold'),
            ),
            tickfont=dict(size=30, color='black', family='Times New Roman'),
            ticksuffix='%',
            showgrid=True,
            gridwidth=1.5,
            gridcolor='lightgray',
            showline=True,
            linewidth=3,
            linecolor='black',
            mirror=False,
            ticks='outside',
            tickwidth=3,
            ticklen=8,
            range=[0, max(vanilla_cpl_pct + salt_cpl_pct) * 1.25],  # Add headroom for labels
        ),
        
        # Legend
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(size=28, color='black', family='Times New Roman'),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=2,
        ),
        
        # Bar mode
        barmode='group',
        bargap=0.3,
        bargroupgap=0.0,
        
        # Background and margins
        plot_bgcolor='#FFFEF9',
        paper_bgcolor='white',
        margin=dict(l=140, r=70, t=150, b=120),
        
        # Font (consistent throughout)
        font=dict(family='Times New Roman', size=28, color='black'),
    )

    # Update x and y axes to have black borders
    fig.update_xaxes(showline=True, linewidth=3, linecolor='black', mirror=False)
    fig.update_yaxes(showline=True, linewidth=3, linecolor='black', mirror=False)

    # Save as high-quality PDF (vector format for LaTeX)
    out_path_pdf = Path(__file__).parent / "cpl_vanilla_vs_salt.pdf"
    fig.write_image(out_path_pdf, format="pdf", width=1400, height=800, scale=2)
    print(f"Saved PDF: {out_path_pdf}")
    
    # Also save as high-DPI PNG as backup
    out_path_png = Path(__file__).parent / "cpl_vanilla_vs_salt.png"
    fig.write_image(out_path_png, format="png", width=1400, height=800, scale=3)
    print(f"Saved PNG: {out_path_png}")
    
    # Save as HTML for interactive viewing
    out_path_html = Path(__file__).parent / "cpl_vanilla_vs_salt.html"
    fig.write_html(out_path_html)
    print(f"Saved HTML: {out_path_html}")

    fig.show()


if __name__ == "__main__":
    plot_cpl_vanilla_vs_salt()