# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import permutation_test_score
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.utils import shuffle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import cross_validation
import plotting
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')

from pyChemometrics import ChemometricsScaler

import os


df = pd.read_csv('/Users/aeiwz/Library/CloudStorage/OneDrive-KhonKaenUniversity/KKUPC/Project/Alpha/KKUPC6602014/Dataset/KKUPC6602014_dataset_preprocessed.csv')
#Drop QC samples
df = df.drop(df[df['Row'] == 'QC'].index)



#Make directory
# path folder
path_ = '/Users/aeiwz/Library/CloudStorage/OneDrive-KhonKaenUniversity/KKUPC/Project/Alpha/KKUPC6602014/Project report/OPLS_result_path/element'

# Create directories if they don't exist
os.makedirs('/Users/aeiwz/Library/CloudStorage/OneDrive-KhonKaenUniversity/KKUPC/Project/Alpha/KKUPC6602014/Project report/OPLS_result_path/element', exist_ok=True)
os.makedirs('/Users/aeiwz/Library/CloudStorage/OneDrive-KhonKaenUniversity/KKUPC/Project/Alpha/KKUPC6602014/Project report/OPLS_result_path/element/hist_plot', exist_ok=True)
os.makedirs('/Users/aeiwz/Library/CloudStorage/OneDrive-KhonKaenUniversity/KKUPC/Project/Alpha/KKUPC6602014/Project report/OPLS_result_path/element/loading_plot', exist_ok=True)
os.makedirs('/Users/aeiwz/Library/CloudStorage/OneDrive-KhonKaenUniversity/KKUPC/Project/Alpha/KKUPC6602014/Project report/OPLS_result_path/element/score_plot', exist_ok=True)
os.makedirs('/Users/aeiwz/Library/CloudStorage/OneDrive-KhonKaenUniversity/KKUPC/Project/Alpha/KKUPC6602014/Project report/OPLS_result_path/element/s_plot', exist_ok=True)
os.makedirs('/Users/aeiwz/Library/CloudStorage/OneDrive-KhonKaenUniversity/KKUPC/Project/Alpha/KKUPC6602014/Project report/OPLS_result_path/main', exist_ok=True)
os.makedirs('/Users/aeiwz/Library/CloudStorage/OneDrive-KhonKaenUniversity/KKUPC/Project/Alpha/KKUPC6602014/Project report/OPLS_result_path/element/Lingress', exist_ok=True)

# Import the datasets from the /data directory
# X for the NMR spectra and Y for the 2 outcome variables

#test group

unique_gr = df['Intervention'].unique()

gr1 = df[df['Intervention'] == unique_gr[0]]
gr2 = df[df['Intervention'] == unique_gr[1]]
gr3 = df[df['Intervention'] == unique_gr[2]]
gr4 = df[df['Intervention'] == unique_gr[3]]
gr5 = df[df['Intervention'] == unique_gr[4]]
gr6 = df[df['Intervention'] == unique_gr[5]]
gr7 = df[df['Intervention'] == unique_gr[6]]


#Combind group

c_1 = pd.concat([gr1,gr2])
c_2 = pd.concat([gr1,gr3])
c_3 = pd.concat([gr1,gr4])
c_4 = pd.concat([gr1,gr5])
c_5 = pd.concat([gr1,gr6])
c_6 = pd.concat([gr1,gr7])

c_8 = pd.concat([gr2,gr3])
c_9 = pd.concat([gr2,gr4])
c_10 = pd.concat([gr2,gr5])
c_11 = pd.concat([gr2,gr6])
c_12 = pd.concat([gr2,gr7])

c_14 = pd.concat([gr3,gr4])
c_15 = pd.concat([gr3,gr5])
c_16 = pd.concat([gr3,gr6])
c_17 = pd.concat([gr3,gr7])

c_19 = pd.concat([gr4,gr5])
c_20 = pd.concat([gr4,gr6])
c_21 = pd.concat([gr4,gr7])

c_23 = pd.concat([gr5,gr6])
c_24 = pd.concat([gr5,gr7])

c_26 = pd.concat([gr6,gr7])


c_list = [c_1,c_2,c_3,c_4,c_5,c_6,c_8,c_9,c_10,
          c_11,c_12,c_14,c_15,c_16,c_17,c_19,c_20,
          c_21,c_23,c_24,c_26]


# File name
name = [ "Gr1 vs Gr2", "Gr1 vs Gr3", "Gr1 vs Gr4", "Gr1 vs Gr5", "Gr1 vs Gr6", "Gr1 vs Gr7",
        "Gr2 vs Gr3", "Gr2 vs Gr4", "Gr2 vs Gr5", "Gr2 vs Gr6", "Gr2 vs Gr7",
        "Gr3 vs Gr4", "Gr3 vs Gr5", "Gr3 vs Gr6", "Gr3 vs Gr7",
        "Gr4 vs Gr5", "Gr4 vs Gr6", "Gr4 vs Gr7",
        "Gr5 vs Gr6", "Gr5 vs Gr7",
        "Gr6 vs Gr7"]



for i in range(len(c_list)):
    

    test_select = c_list[i]

    spectra = test_select.iloc[:, 43:]
    ppm = list(spectra.columns.astype(float))
    X = spectra.values
    y = test_select['Intervention'].values
    
    # Create a pipeline with data preprocessing and OPLS-DA model
    pipeline = Pipeline([
                            ('scale', ChemometricsScaler(scale_power=0.5)),
                            ('oplsda', PLSRegression(n_components=2)),
                            ('opls', cross_validation.CrossValidation(kfold=3, estimator='opls', scaler='pareto'))
                         ])

    oplsda = pipeline.named_steps['oplsda']
    cv = pipeline.named_steps['opls']
    cv.fit(X, y)
    cv.reset_optimal_num_component(2)
    oplsda.fit(X, pd.Categorical(y).codes)
    n_permutate = 1000

    # Permutation test to assess the significance of the model
    acc_score, permutation_scores, p_value = permutation_test_score(
    pipeline.named_steps['oplsda'], X, pd.Categorical(y).codes, cv=3, n_permutations=n_permutate, n_jobs=-1, random_state=57, verbose=10)


    s_scores_df = pd.DataFrame({'correlation': cv.correlation,'covariance': cv.covariance}, index=ppm)
    df_opls_scores = pd.DataFrame({'t_scores': cv.scores, 't_ortho': cv.orthogonal_score, 't_pred': cv.predictive_score, 'label': y})

        
    #Visualise
    from pca_ellipse import confidence_ellipse
    fig = px.scatter(df_opls_scores, x='t_scores', y='t_ortho', 
                    color='label', 
                    color_discrete_map={
                                        "Corn oil": "#E91E63",        
                                        "D-galactose": "#FF9800",
                                        "AB extract 500 mg/kg": "#FFEB3B",       
                                        "AP extract 250 mg/kg": "#9C27B0",
                                        "Vitamin E": "#03A9F4",
                                        "AP extract 500 mg/kg": "#4CAF50",        
                                        "AB extract 250 mg/kg": "#B30000",
                                        "0.5% SCMC": "#3F51B5"
                                        }, 
                    title='<b>OPLS-DA Scores Plot<b>', 
                    height=900, width=1300,
                    labels={
                        't_pred': 't<sub>predict</sub>',
                        't_ortho': 't<sub>orthogonal</sub>',
                        't_scores': 't<sub>scores</sub>',
                        'label': 'Intervention'}
                    )

    #fig.add_annotation(yref = 'paper', y = -1.06, xref = 'paper', x=1.06 , text='Q2' +' = {}'.format(np.round(df_explained_variance_.iloc[2,2], decimals=2)))
    #fig.update_annotations(font = {
    #    'size': 20}, showarrow=False)

    #set data point fill alpha with boarder in each color
    fig.update_traces(marker=dict(size=35, opacity=0.7, line=dict(width=2, color='DarkSlateGrey')))

    fig.add_annotation(dict(font=dict(color="black",size=20),
                            #x=x_loc,
                            x=0,
                            y=1.04+0.05,
                            showarrow=False,
                            text='<b>R<sup>2</sup>X: {}%<b>'.format(np.round(cv.R2Xcorr*100, decimals=2)),
                            textangle=0,
                            xref="paper",
                            yref="paper"),
                            # set alignment of text to left side of entry
                            align="left")

    fig.add_annotation(dict(font=dict(color="black",size=20),
                            #x=x_loc,
                            x=0,
                            y=1.0+0.05,
                            showarrow=False,
                            text='<b>R<sup>2</sup>Y: {}%<b>'.format(np.round(cv.R2y*100, decimals=2)),
                            textangle=0,
                            xref="paper",
                            yref="paper"),
                            # set alignment of text to left side of entry
                            align="left")
    fig.add_annotation(dict(font=dict(color="black",size=20),
                            #x=x_loc,
                            x=0,
                            y=1.08+0.05,
                            showarrow=False,
                            text='<b>Q<sup>2</sup>: {}%<b>'.format(np.round(cv.q2*100, decimals=2)),
                            textangle=0,
                            xref="paper",
                            yref="paper"),
                            # set alignment of text to left side of entry
                            align="left")

    fig.add_shape(type='path',
            path=confidence_ellipse(df_opls_scores['t_scores'], df_opls_scores['t_ortho']))


    fig.update_traces(marker=dict(size=35))
    #fig.update_traces(textposition='top center') #Text label position
    #change M to 10^6
    fig.update_yaxes(tickformat=",.0")
    fig.update_xaxes(tickformat=",.0")

    #fig.update_traces(marker=dict(size=12, color=Y1_color, marker=Y2_marker))

    fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
    fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_layout(
        title={
            'y':1,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        font=dict(size=20))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')

    #fig.show()
    fig.write_image("{}/score_plot/score_plot_{}.png".format(path_, name[i]))
    fig.write_html("{}/score_plot/score_plot{}.html".format(path_, name[i]))




    #Histrogram
    #Plot histogram of permutation scores
    fig = px.histogram(permutation_scores, nbins=50, height=500, width=1000, 
                    title='<b>Permutation scores<b>',
                    labels={'value': 'Accuracy score', 
                            'count': 'Frequency'})
    #add dashed line to indicate the accuracy score of the model line y location is maximum count of histogram
    fig.add_shape(type='line', yref='paper', y0=0, y1=1, xref='x', x0=acc_score, x1=acc_score, line=dict(dash='dash', color='red', width=3))


    fig.add_annotation(dict(font=dict(color="black",size=14),
                            #x=x_loc,
                            x=0,
                            y=1.25,
                            #y=1.18,
                            showarrow=False,
                            text='Number of permutation: {}'.format(n_permutate),
                            textangle=0,
                            xref="paper",
                            yref="paper"),
                            # set alignment of text to left side of entry
                            align="left")

    fig.add_annotation(dict(font=dict(color="black",size=14),
                            #x=x_loc,
                            x=0,
                            y=1.18,
                            showarrow=False,
                            text='Accuracy score: {}'.format(np.round(acc_score, decimals=3)),
                            textangle=0,
                            xref="paper",
                            yref="paper"),
                            # set alignment of text to left side of entry
                            align="left")
    fig.add_annotation(dict(font=dict(color="black",size=14),
                            #x=x_loc,
                            x=0,
                            y=1.11,
                            showarrow=False,
                            text='<i>p-value</i>: {}'.format(np.round(p_value, decimals=6)),
                            textangle=0,
                            xref="paper",
                            yref="paper"),
                            # set alignment of text to left side of entry
                            align="left")

    fig.update_layout(showlegend=False)

    fig.update_layout(title_x=0.5)

    #fig.show()
    fig.write_image("{}/hist_plot/Permutation_scores_{}.png".format(path_, name[i]))
    fig.write_html("{}/hist_plot/Permutation_scores_{}.html".format(path_, name[i]))
    
    
    
    #S plot
    # sub-plot covariance for x and correlation for y S-plot using plotly, color by covariance with jet colormap
    #setup figure size


    fig = px.scatter(s_scores_df, x='covariance', y='correlation', color='covariance', range_color=[-1,1],
                     color_continuous_scale='jet', text=s_scores_df.index, height=900, width=2000)
    fig.update_layout(title='<b>S-plot</b>', xaxis_title='Covariance', yaxis_title='Correlation')

    #add line of axis and set color to black and line width to 2 pixel
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    #Add tick width to 2 pixel
    fig.update_xaxes(tickwidth=2)
    fig.update_yaxes(tickwidth=2)
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    fig.update_yaxes(tickformat=",.0")
    #fig.update_xaxes(tickformat=",.0")
    fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
    fig.update_yaxes(zeroline=True, zerolinewidth=2, zerolinecolor='Black')
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_layout(
        title={
            'y':1,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        font=dict(size=20))
    #Set font size to 20
    #Set marker size to 5 pixel
    fig.update_traces(marker=dict(size=14))
    #fig.show()
    fig.write_image("{}/s_plot/S_plot_{}.png".format(path_, name[i]))
    fig.write_html("{}/s_plot/S_plot_{}.html".format(path_, name[i]))
    

    #Loadings plot

    fig = px.scatter(s_scores_df, x=ppm, y=np.median(X, axis=0), color='covariance', color_continuous_scale='jet', text=s_scores_df.index, height=500, width=2000)

    fig.update_traces(marker=dict(size=3))
    fig.update_xaxes(autorange="reversed")
    fig.update_layout(title='<b>Median spectra</b>', xaxis_title='ppm', yaxis_title='Intensity (AU)')
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    #Add tick width to 2 pixel
    fig.update_xaxes(tickwidth=2)
    fig.update_yaxes(tickwidth=2)

    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)')
    fig.update_yaxes(tickformat=",.0")
    #fig.update_xaxes(tickformat=",.0")
    fig.update_layout(
        title={
            'y':1,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        font=dict(size=20))
    #Set marker size to 5 pixel
    fig.update_traces(marker=dict(size=3))
    #fig.show()
    fig.write_image("{}/loading_plot/loadings_plot_{}.png".format(path_, name[i]))
    fig.write_html("{}/loading_plot/loadings_plot_{}.html".format(path_, name[i]))

