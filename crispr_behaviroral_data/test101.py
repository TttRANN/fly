import plotly.graph_objs as go
import numpy as np

rng = np.random.default_rng(19680801)
data = np.array([20, 20, 20, 20])
x = ['A', 'B', 'C', 'D']
frames = []

# Generate frames
for _ in range(20):
    data += rng.integers(low=0, high=10, size=data.shape)
    frame = go.Frame(data=[go.Bar(x=data, y=x, orientation='h', marker=dict(color=['blue', 'red', 'green', 'purple']))])
    frames.append(frame)

# Create initial figure
fig = go.Figure(
    data=[go.Bar(x=[20, 20, 20, 20], y=x, orientation='h', marker=dict(color=['blue', 'red', 'green', 'purple']))],
    layout=go.Layout(
        title="Animated Bar Chart",
        updatemenus=[dict(type='buttons', showactive=False,
                          buttons=[dict(label='Play',
                                        method='animate',
                                        args=[None, dict(frame=dict(duration=400, redraw=True), fromcurrent=True)])])]),
    frames=frames
)

fig.show()
