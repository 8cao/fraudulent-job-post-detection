import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import pickle

# Load data and models
df_results = pd.read_csv('results_dash.csv')
df_docs= pd.read_pickle('df_docs.pkl')
model_w2v = Word2Vec.load("model_w2v.model")
df_tsne = pd.read_csv('df_tsne.csv')
topic_keywords = pickle.load(open('topic_keywords.pkl', 'rb'))

# Set up Dash app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define the layout with tabs
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Word2Vec Visualization', children=[
            html.Div([
                html.H1("Word2Vec t-SNE Visualization", style={'textAlign': 'center'}),
                dcc.Graph(id='word-plot', figure=px.scatter(
                    df_tsne, x='x', y='y', text='word', hover_data=['word']
                ).update_traces(textposition='top center').update_layout(height=600)),
                html.Div(id='similar-words', style={'padding': '20px', 'font-size': '16px'})
            ])
        ]),
        dcc.Tab(label='Document Topic Distribution Viewer', children=[
            html.Div([
                html.H1("Document Topic Distribution Viewer", style={'textAlign': 'center'}),
                dcc.RadioItems(
                    id='fraud-filter',
                    options=[
                        {'label': 'All Documents', 'value': 'all'},
                        {'label': 'Fraudulent', 'value': 'fraud'},
                        {'label': 'Not Fraudulent', 'value': 'not_fraud'}
                    ],
                    value='all',
                    labelStyle={'display': 'inline-block'},
                    style={'textAlign': 'center', 'margin': '10px'}
                ),
                dcc.Dropdown(
                    id='document-selector',
                    multi=True,
                    style={'width': '60%', 'marginLeft': 'auto', 'marginRight': 'auto'}
                ),
                dcc.Graph(id='topic-distribution-chart'),
                html.Div(id='document-texts', style={'white-space': 'pre-wrap', 'margin': '20px'})
            ])
        ]),
        dcc.Tab(label='Model Performance Dashboard', children=[
            html.Div([
                html.H1("Model Performance"),
                dcc.Dropdown(
                    id='vectorization-dropdown',
                    options=[{'label': i, 'value': i} for i in df_results['Vectorization'].unique()],
                    value=['Bag of Words'],  # Default value
                    multi=True,  # Allow multiple selections
                    clearable=False
                ),
                dcc.Dropdown(
                    id='model-dropdown',
                    options=[{'label': i, 'value': i} for i in df_results['Model'].unique()],
                    value=[],  # Default to empty
                    clearable=False,
                    multi=True  # Allow multiple selections
                ),
                dcc.Graph(id='performance-graph')
            ])
        ])
    ])
])

# Callbacks for Word2Vec
@app.callback(
    Output('similar-words', 'children'),
    [Input('word-plot', 'clickData')]
)
def display_similar_words(clickData):
    if clickData:
        word = clickData['points'][0]['text']
        similar_words = model_w2v.wv.most_similar(positive=[word], topn=5)
        similar_words_html = html.Ul([html.Li(f"{sim_word}") for sim_word, _ in similar_words])
        return [
            html.P(f"5 most similar words to '{word}':", style={'font-weight': 'bold', 'font-size': '20px'}),
            similar_words_html
        ]
    return html.P("Click on a word to see similar words.", style={'font-size': '16px'})

# Callbacks for Model Performance Dashboard
@app.callback(
    Output('performance-graph', 'figure'),
    [Input('vectorization-dropdown', 'value'),
     Input('model-dropdown', 'value')]
)
def update_graph(selected_vectorizations, selected_models):
    if not selected_models or not selected_vectorizations:
        return px.bar()  # Return an empty plot if no model or vectorization is selected

    filtered_df = df_results[(df_results['Vectorization'].isin(selected_vectorizations)) & (df_results['Model'].isin(selected_models))]
    if filtered_df.empty:
        return px.bar(title="No data available for the selected filters")
    
    melted_df = filtered_df.melt(id_vars=['Model', 'Vectorization'], value_vars=['C0_recall', 'C1_recall', 'Macro_F1_Score'],
                                 var_name='Metric', value_name='Score')
    fig = px.bar(melted_df, x='Model', y='Score', color='Metric',
                 facet_col='Vectorization', barmode='group',
                 title="Comparison of Model Performance Metrics Across Vectorization Techniques",
                 labels={"Score": "Metric Score", "Metric": "Performance Metric"})
    fig.update_layout(xaxis={'categoryorder':'total descending'})
    return fig

# Callbacks for Document Topic Distribution Viewer
@app.callback(
    Output('document-selector', 'options'),
    [Input('fraud-filter', 'value')]
)
def update_document_options(fraud_filter):
    if fraud_filter == 'fraud':
        filtered_docs = df_docs[df_docs['fraudulent'] == 1]
    elif fraud_filter == 'not_fraud':
        filtered_docs = df_docs[df_docs['fraudulent'] == 0]
    else:
        filtered_docs = df_docs

    return [{'label': f'Document {doc_id}', 'value': doc_id} for doc_id in filtered_docs['Document_ID']]

@app.callback(
    [
        Output('topic-distribution-chart', 'figure'),
        Output('document-texts', 'children')
    ],
    [Input('document-selector', 'value'),
     Input('fraud-filter', 'value')]
)
def update_output(selected_docs, fraud_filter):
    if not selected_docs:
        return [px.pie(values=[], names=[]), "Select documents to see their text and topic distribution."]

    # Filter based on fraud status
    if fraud_filter == 'fraud':
        filtered_docs = df_docs[df_docs['fraudulent'] == 1]
    elif fraud_filter == 'not_fraud':
        filtered_docs = df_docs[df_docs['fraudulent'] == 0]
    else:
        filtered_docs = df_docs

    selected_filtered_docs = filtered_docs[filtered_docs['Document_ID'].isin(selected_docs)]

    if selected_filtered_docs.empty:
        return [px.pie(values=[], names=[]), "No documents match the selected criteria."]

    # Aggregate topic distributions for selected documents
    selected_probs = selected_filtered_docs['Topic_Prob_Dist']
    avg_probs = np.mean(selected_probs.tolist(), axis=0)

    # Generate pie chart for top 5 topics
    top_indices = np.argsort(avg_probs)[-5:][::-1]
    top_values = [avg_probs[i] for i in top_indices]
    top_labels = [topic_keywords[i] for i in top_indices]
    pie_chart = px.pie(values=top_values, names=top_labels, title="Top 5 Predominant Topics Distribution")
    pie_chart.update_traces(textinfo='percent', hoverinfo='label+percent', marker=dict(line=dict(color='#000000', width=1)))
    pie_chart.update_layout(showlegend=True, legend_title_text='Topic Descriptions')

    # Display selected documents texts
    selected_texts = selected_filtered_docs['text']
    documents_text = format_document_text(zip(selected_filtered_docs['Document_ID'], selected_texts))

    return [pie_chart, documents_text]

def format_document_text(texts, word_limit=100):
    formatted_texts = []
    for doc_id, text in texts:
        joined_text = " ".join(text[:word_limit])
        formatted_texts.append(f"Document {doc_id}: {joined_text}...")
    return '\n\n'.join(formatted_texts)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
