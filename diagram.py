from graphviz import Digraph

def create_architecture_diagram():
    # Initialize the graph
    dot = Digraph(comment='Bitcoin Price Prediction Model', format='png')
    dot.attr(rankdir='TB')  # Top-to-Bottom layout
    dot.attr('node', shape='box', style='filled', fillcolor='aliceblue', fontname='Helvetica')
    dot.attr('edge', fontname='Helvetica', fontsize='10')

    # --- Hyperparameters Note ---
    dot.node('Config', 
             'TRAINING SETUP\n'
             'Optimizer: Adam (lr=2e-4)\n'
             'Loss: Focal Loss\n'
             'Batch Size: 64\n'
             'Scaler: RobustScaler', 
             shape='note', fillcolor='lightyellow')

    # --- Input Layer ---
    dot.node('Input', 'Input Layer\nShape: (14, 13)\n(14 Days, 13 Features)', fillcolor='lightgrey')

    # --- Bi-LSTM Block 1 ---
    dot.node('BiLSTM1', 'Bi-Directional LSTM (1)\n128 Units (256 Total)\nL2 Reg: 0.0005\nReturn Sequences: True')
    dot.node('LN1', 'Layer Normalization + Dropout (0.3)')

    # --- Bi-LSTM Block 2 ---
    dot.node('BiLSTM2', 'Bi-Directional LSTM (2)\n64 Units (128 Total)\nL2 Reg: 0.0005\nReturn Sequences: True')
    dot.node('LN2', 'Layer Normalization + Dropout (0.3)')

    # --- Attention Block (Cluster) ---
    with dot.subgraph(name='cluster_attention') as c:
        c.attr(label='Residual Attention Block', style='dashed', color='grey')
        c.node('MHA', 'Multi-Head Attention\nHeads: 4\nKey Dim: 32', fillcolor='gold')
        c.node('Add', 'Add\n(Residual Connection)', shape='circle', width='0.8', fillcolor='white')
        c.node('LN3', 'Layer Normalization')

    # --- Feature Compression ---
    dot.node('LSTM3', 'LSTM (3)\n32 Units\nL2 Reg: 0.0005\nReturn Sequences: False')
    dot.node('Drop3', 'Dropout (0.3)')

    # --- Classification Head ---
    with dot.subgraph(name='cluster_head') as c:
        c.attr(label='Classification Head', color='grey')
        c.node('Dense1', 'Dense Layer\n64 Units (ReLU)\nBatch Normalization')
        c.node('Drop4', 'Dropout (0.21)')
        
        c.node('Dense2', 'Dense Layer\n32 Units (ReLU)')
        c.node('Drop5', 'Dropout (0.15)')

    # --- Output ---
    dot.node('Output', 'Output Layer\n1 Unit (Sigmoid)\n(0=No Continuation, 1=Uptrend)', 
             shape='ellipse', fillcolor='salmon')

    # --- Define Edges ---
    # Main Flow
    dot.edge('Input', 'BiLSTM1')
    dot.edge('BiLSTM1', 'LN1')
    dot.edge('LN1', 'BiLSTM2')
    dot.edge('BiLSTM2', 'LN2')

    # Residual Connection Split
    dot.edge('LN2', 'MHA', label='Query/Key/Value')
    dot.edge('LN2', 'Add', label='Skip Connection')
    dot.edge('MHA', 'Add')
    
    # Continuation
    dot.edge('Add', 'LN3')
    dot.edge('LN3', 'LSTM3')
    dot.edge('LSTM3', 'Drop3')
    
    dot.edge('Drop3', 'Dense1')
    dot.edge('Dense1', 'Drop4')
    dot.edge('Drop4', 'Dense2')
    dot.edge('Dense2', 'Drop5')
    dot.edge('Drop5', 'Output')

    # Render the graph to a file
    output_filename = 'bitcoin_model_architecture'
    dot.render(output_filename, view=True)
    print(f"Diagram saved as {output_filename}.png")

if __name__ == '__main__':
    create_architecture_diagram()