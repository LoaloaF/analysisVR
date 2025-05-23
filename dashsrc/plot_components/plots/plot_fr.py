import plotly.graph_objects as go

import pandas as pd
import numpy as np
from dash import dcc, html
from plotly.subplots import make_subplots

def render_plot_old(fr_avg, tracK_data, s_id):
    # create 2 row subplot setup
    print("Rendering FR plot")
    
    fr_avg = fr_avg.reset_index(drop=True)
    print(fr_avg)
    fig = make_subplots(rows=2, cols=1, row_heights=[.65, .35],
                        shared_xaxes=True,
                        vertical_spacing=0.1, subplot_titles=["HP", "mPFC"])
    
    fr_avg.columns = fr_avg.columns.astype(int)
    fr_avg = fr_avg.T
    fr_hz_averages = fr_avg.reindex(np.arange(fr_avg.index.max() + 1)).fillna(0)
    

    fr_hz_averages_HP = fr_hz_averages.iloc[:20]
    fr_hz_averages_mPFC = fr_hz_averages.iloc[20:]

    # Create a heatmap using Plotly
    fig.add_trace(
        go.Heatmap(
            z=fr_hz_averages_HP.values,  # Heatmap values (transposed)
            x=fr_hz_averages_HP.columns,  # time bins
            y=np.arange(len(fr_hz_averages_HP.index)),  # neurons
            colorscale="Picnic",  # Color scale
            zmax=1.4,
            zmin=-1.4,
            colorbar=dict(title="Firing Rate (Z)"),  # Colorbar title
        ), row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(
            z=fr_hz_averages_mPFC.values,  # Heatmap values (transposed)
            x=fr_hz_averages_mPFC.columns,  # time bins
            y=np.arange(len(fr_hz_averages_mPFC.index)),  # neurons
            colorscale="Picnic",  # Color scale
            zmax=1.4,
            zmin=-1.4,
            # ytype="category",
            colorbar=dict(title="Firing Rate (Z)"),  # Colorbar title
        ), row=2, col=1,
    )
    
    # Update layout for better visualization
    fig.update_layout(
        title=f"Firing Rate Heatmap, session_id={s_id}",
        xaxis_title="Track Bins",
        yaxis_title="Neurons",
        template="plotly_white",
        height=800,  # Adjust height to fit all rows
        width=1000,  # Adjust width for better visualization
        yaxis_type="category",
    )
    print(fr_hz_averages.index.astype(str).values )
    fig.update_yaxes(
            # title='your_title',
            # autorange='reversed',
            # showgrid=False,   
            # autotick=False,  
            # dtick=1,
            
            tickmode='linear',  # Ensure ticks are evenly spaced
            tickvals=np.arange(len(fr_hz_averages.index)),  # Match the number of rows in the heatmap
            ticktext=fr_hz_averages.index.astype(str).values,  # Use the neuron indices as labels
            scaleanchor=None,  # Prevent distortion of the heatmap
        )
    

    return fig

import numpy as np

def render_pca_plot(fr):
    fr.columns = fr.columns.astype(int)
    fr = fr.reindex(columns=sorted(fr.columns))
    
    def pca_reduce(X, variance_threshold=0.95):
        # Step 1: Mean-center the data along neurons
        X_centered = X - np.mean(X, axis=0, keepdims=True)  # Mean-center each timebin

        # Step 2: Compute covariance matrix over neurons
        cov_matrix = np.cov(X_centered, rowvar=True)  # Neuron-to-neuron covariance
        
        # Step 3: Eigen-decomposition
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)  # eigh for symmetric matrices
        
        # Step 4: Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigvals)[::-1]
        eigvals_sorted = eigvals[sorted_indices]
        
        # Step 5: Calculate cumulative variance ratio
        explained_variance_ratio = eigvals_sorted / np.sum(eigvals_sorted)
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Step 6: Find number of components to retain desired variance
        n_components = np.searchsorted(cumulative_variance, variance_threshold) + 1
        
        return n_components, eigvals_sorted

    for s_id in fr.index.unique('session_id'):
        X = fr.loc[pd.IndexSlice[:,:,s_id]].fillna(0).values.T.astype(float)
        n_components, eigvals = pca_reduce(X, variance_threshold=0.8)
        print(f"Number of components needed for Session {s_id}: {n_components}")


def render_unit_activity_plot(fr):
    fr = fr.set_index(['trial_id', 'from_z_position_bin', 'cue'], append=True, )
    fr.drop(columns=['trial_outcome','bin_length'], inplace=True)
    fr.columns = fr.columns.astype(int)
    fr = fr.reindex(columns=sorted(fr.columns))
    
    fr = fr.iloc[:, 20:]
    title = 'mPFC'
    # title = 'HP'
    height = 20 *15
    # height = 20 *20
    
    trial_avg = fr.groupby(['trial_id', 'session_id']).mean().sort_index(level=['session_id', 'trial_id']).T.fillna(0).astype(float)
    
    
    # # Perform hierarchical clustering
    # from scipy.cluster.hierarchy import linkage, leaves_list
    # linkage_matrix = linkage(trial_avg, method='ward')  # Using Ward's method
    # ordered_indices = leaves_list(linkage_matrix)  # Get the order of rows
    # # Reorder the rows of spks based on clustering
    # trial_avg_reord = trial_avg.iloc[ordered_indices, :]

    from rastermap import Rastermap
    model = Rastermap(n_clusters=None, # None turns off clustering and sorts single neurons 
                    n_PCs=20, # use fewer PCs than neurons
                    locality=0.6, # some locality in sorting (this is a value from 0-1)
                    time_lag_window=50, # use future timepoints to compute correlation
                    grid_upsample=0, # 0 turns off upsampling since we're using single neurons
                    ).fit(trial_avg.values) # transpose to get neurons x timepoints
    y = model.embedding # neurons x 1
    isort = model.isort
    trial_avg_reord = trial_avg.iloc[isort, :]
    
    # normalize 
    # max_fr = np.percentile(trial_avg_reord.values, 95, axis=1)
    # trial_avg_reord /= max_fr[:, None]
    
    # other norm
    trial_avg_reord = (trial_avg_reord != 0).astype(int)
    print(trial_avg_reord)
    
    # draw a heatmap
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Heatmap(
            z=trial_avg_reord.values,  # Heatmap values (transposed)
            y=trial_avg_reord.index,  # time bins
            x=np.arange(len(trial_avg_reord.columns)),  # neurons
            # colorscale="Greys",  # Color scale
            colorscale="BuGn",  # Color scale
            zmax=2,
            zmin=0,
            colorbar=dict(title="Spike detected"),  # Colorbar title
            # colorbar=dict(title="rel. firing rate"),  # Colorbar title
        ), row=1, col=1,
    )
    
    # add session_id lines
    session_gaps = np.argwhere(trial_avg_reord.columns.get_level_values('trial_id') == 1.0)
    for gap in session_gaps:
        fig.add_shape(
            type="line",
            x0=gap[0],
            y0=0,
            x1=gap[0],
            y1=len(trial_avg_reord.index),
            line=dict(color="red", width=.6),
        )
    
    fig.update_layout(
        title=f"Firing Rate over 2 months, {title}",
        xaxis_title="Trials",
        yaxis_title="Neurons",
        template="plotly_white",
        height=height,  # Adjust height to fit all rows
        width=1300,  # Adjust width for better visualization
        yaxis_type="category",
    )
    
    
    return fig


# from sklearn.decomposition import PCA
# import numpy as np
# from sklearn.svm import SVC
# from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches

def decode_cue(fr, metadata):
    fr = fr.set_index(['trial_id', 'from_z_position_bin', 'cue'], append=True, )
    outcome = fr.loc[:, 'trial_outcome']
    fr.drop(columns=['trial_outcome','bin_length'], inplace=True)
    fr.columns = fr.columns.astype(int)
    fr = fr.reindex(columns=sorted(fr.columns))
    print(fr)
    # print(metadata.columns)
    # print(metadata.loc[:, "DR"])
    
    for i in range(2):
        
        if i == 0:
            cue_fr = fr.iloc[:, :20]
            title = 'HP'
        else:   
            cue_fr = fr.iloc[:, 20:]
            title = 'mPFC'

        cue_fr = cue_fr.loc[pd.IndexSlice[:,:,:,:,:, -80:50]]
        print(outcome)
        cue_outcome = outcome.loc[pd.IndexSlice[:,:,:,:,:, -80:50]]
        cue_fr.index = cue_fr.index.droplevel((0,1,3))
        cue_outcome.index = cue_outcome.index.droplevel((0,1,3))
        print(cue_fr)
        print(cue_outcome)
        
        m_vals, std_vals, s_ids = [], [], []
        outcome_res = []
        for s_id in cue_fr.index.unique('session_id'):
            # print(cue_fr)
            X = cue_fr.loc[s_id].unstack(level='from_z_position_bin').fillna(0).astype(float).copy()
            print(X.shape)
            if X.shape[0] < 10:
                print(f"Session {s_id} has too few trials, skipping")
                continue
            # if s_id <8:
            #     continue
            
            Y = X.index.get_level_values('cue').values.astype(int)
            Y[Y == 0] = 1 # very rare
            # print(Y)
            # print(Y.shape)
            
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            pca = PCA(n_components=20)
            X_pca = pca.fit_transform(X_scaled)
            

            
            cue_outcomes_reshaped = cue_outcome.loc[s_id].unstack(level='from_z_position_bin').iloc[:,0].values
            cue_outcomes_reshaped[cue_outcomes_reshaped!=0] = 1
            Y2 = cue_outcomes_reshaped
            
            # Define classifier
            clf = SVC(kernel='rbf', C=1, gamma='scale')
            scores = cross_val_score(clf, X_pca, Y, cv=10)
            
            # clf.fit(X_pca, Y2)
            # correct_predictions = clf.predict(X_pca) == Y2
            # print(res.shape)
            # cue_predicted_outcomes = np.mean(cue_outcomes_reshaped[correct_predictions])
            # cue_notpredicted_outcomes = np.mean(cue_outcomes_reshaped[~correct_predictions])
            # outcome_res.append((cue_predicted_outcomes, cue_notpredicted_outcomes))
            
            # print(f"Accuracy per fold: {scores}")
            print(f"S{s_id:02d} - Mean accuracy: {scores.mean():.3f} ± {scores.std():.3f}, ")
            
            m_vals.append(scores.mean())
            std_vals.append(scores.std())
            s_ids.append(s_id)
            
        fig = plt.figure(figsize=(10, 6))
        plt.errorbar(np.arange(len(m_vals)), m_vals, yerr=std_vals, fmt='o')
        plt.title(f"Decoding Cue from {cue_fr.shape[1]} units in {title}")
        plt.xlabel("Session")
        plt.ylabel("Accuracy")
        plt.xticks(np.arange(len(m_vals)), s_ids)
        
        plt.hlines([0.5], xmin=-1, xmax=19, color='red', linestyle='--', linewidth=2, label='Chance level')
        plt.legend()
        # plt.ylim(0, 1)        

        plt.show()
        
        # outcome_res = np.array(outcome_res)
        # plt.figure(figsize=(10, 6))
        # plt.plot(outcome_res[:,0], label='Predicted outcomes', color='blue')
        # plt.plot(outcome_res[:,1], label='Not predicted outcomes', color='orange')
        # plt.title(f"Decoding Cue from {cue_fr.shape[1]} units in {title}")
        # plt.xlabel("Session")
        # plt.ylabel("Outcome")
        # plt.show()