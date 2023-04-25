import torch
import torch.nn as nn
import streamlit as st

class CustomCNN(nn.Module):
    def __init__(self, num_conv_layers, num_filters, filter_sizes, dropout_rates):
        super(CustomCNN, self).__init__()

        conv_layers = []
        for i in range(num_conv_layers):
            in_channels = 3 if i == 0 else num_filters[i-1]
            conv_layers.append(nn.Conv2d(in_channels, num_filters[i], filter_sizes[i], padding=(filter_sizes[i] // 2)))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(2))
            if dropout_rates[i] > 0:
                conv_layers.append(nn.Dropout(dropout_rates[i]))
        self.conv = nn.Sequential(*conv_layers)

        num_flat_features = num_filters[-1] * ((32 // (2**num_conv_layers))**2)

        self.fc1 = nn.Linear(num_flat_features, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return nn.functional.softmax(x, dim=1)


import plotly.graph_objects as go
import numpy as np

# def create_cnn_visualization(num_conv_layers, num_filters, fc_layers):
#     shapes = []
#     annotations = []
#
#     layer_spacing = 2
#     conv_max_width = 20
#     conv_max_height = 20
#     fc_max_width = 20
#     fc_max_height = 10
#     total_layers = num_conv_layers + len(fc_layers)
#     x_positions = [i * layer_spacing for i in range(total_layers)]
#
#     max_filters = max(num_filters)
#     max_fc_nodes = max(fc_layers)
#
#     for i in range(total_layers):
#         x0 = x_positions[i] - 0.5
#         x1 = x_positions[i] + 0.5
#         layer_label_y = -5
#
#         # Add lines connecting the rectangles
#         if i > 0:
#             shapes.append(
#                 dict(
#                     type="line",
#                     xref="x",
#                     yref="y",
#                     x0=x_positions[i - 1] + 0.5,
#                     x1=x_positions[i] - 0.5,
#                     y0=y1,
#                     y1=y1,
#                     line=dict(color="black"),
#                 )
#             )
#             shapes.append(
#                 dict(
#                     type="line",
#                     xref="x",
#                     yref="y",
#                     x0=x_positions[i - 1] + 0.5,
#                     x1=x_positions[i] - 0.5,
#                     y0=y0,
#                     y1=y0,
#                     line=dict(color="black"),
#                 )
#             )
#
#         if i < num_conv_layers:
#             layer_width = np.interp(num_filters[i], [16, max_filters], [1, conv_max_width])
#             layer_height = conv_max_height
#             y0 = -layer_height / 2
#             y1 = layer_height / 2
#             layer_label = f"Conv {i + 1} ({num_filters[i]}x{num_filters[i]})"
#         else:
#             layer_width = fc_max_width
#             layer_height = np.interp(fc_layers[i - num_conv_layers], [10, max_fc_nodes], [1, fc_max_height])
#             y0 = -layer_height / 2
#             y1 = layer_height / 2
#             layer_label = f"FC {i - num_conv_layers + 1} ({fc_layers[i - num_conv_layers]})"
#
#         shapes.append(
#             dict(
#                 type="rect",
#                 xref="x",
#                 yref="y",
#                 x0=x0,
#                 x1=x1,
#                 y0=y0,
#                 y1=y1,
#                 fillcolor="#1f77b4",
#                 line=dict(color="#1f77b4"),
#             )
#         )
#
#         annotations.append(
#             dict(
#                 x=x_positions[i],
#                 y=layer_label_y,
#                 xref="x",
#                 yref="y",
#                 text=layer_label,
#                 showarrow=False,
#                 font=dict(size=14, color="black"),
#             )
#         )
#
#     layout = go.Layout(
#         showlegend=False,
#         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
#         plot_bgcolor="white",
#         shapes=shapes,
#         annotations=annotations,
#         width=800,  # Adjust the width of the plot
#         height=800,  # Adjust the height of the plot
#     )
#
#     return go.Figure(layout=layout)

import hiddenlayer as hl

def visualize_pytorch_model(model):
    transforms = [
        hl.transforms.Prune("Constant"),
        hl.transforms.Fold("Conv > ReLU > MaxPool", "Conv > ReLU > MaxPool"),
        hl.transforms.Fold("Linear > ReLU", "Linear > ReLU"),
    ]

    graph = hl.build_graph(model, torch.zeros([1, 3, 32, 32]), transforms=transforms)
    return graph


st.title('Interactive CNN Builder for CIFAR-10 (PyTorch)')

num_conv_layers = st.slider('Number of Convolutional Layers', min_value=1, max_value=5, value=1)
num_filters = []
filter_sizes = []
dropout_rates = []
fc_layers = [64, 10]  # Number of nodes in fully connected layers

for i in range(num_conv_layers):
    st.subheader(f'Convolutional Layer {i+1}')
    num_filters.append(st.slider(f'Number of Filters in Layer {i+1}', min_value=16, max_value=128, value=32, step=16))
    filter_sizes.append(st.slider(f'Filter Size in Layer {i+1}', min_value=1, max_value=5, value=3))
    dropout_rates.append(st.slider(f'Dropout Rate in Layer {i+1}', min_value=0.0, max_value=0.5, value=0.0, step=0.1))

model = CustomCNN(num_conv_layers, num_filters, filter_sizes, dropout_rates)

st.subheader('Model Summary')
st.text(str(model))

st.subheader('Model Visualization')
graph = visualize_pytorch_model(model)
st.graphviz_chart(str(graph))


