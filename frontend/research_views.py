"""Research figures rendered from offline aggregate evaluation artifacts."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def curve(x, y, name, xlabel, ylabel):
    fig = go.Figure(go.Scatter(x=x, y=y, mode='lines+markers', name=name,
                             line=dict(color='#087f8c', width=2), marker=dict(size=4)))
    fig.update_layout(template='plotly_white', height=360, margin=dict(l=30, r=20, t=25, b=45),
                      xaxis_title=xlabel, yaxis_title=ylabel, showlegend=True,
                      legend=dict(orientation='h', y=1.12), font=dict(size=12))
    return fig


def render_research(data):
    st.subheader('Research Results')
    st.caption('Random Forest + SMOTE | original held-out patients | record-level evaluation')
    if 'threshold_analysis' not in data:
        st.info('Research plots are not generated yet. Run python training/refresh_analytics.py offline, then restart the backend.')
        return
    st.caption(f"{data['evaluation_records']:,} test observations, {data['evaluation_positive']:,} positive labels. No external or prospective validation; no confidence intervals estimated.")
    tabs = st.tabs(['Evaluation curves', 'SHAP summary', 'Tables & heatmap'])
    with tabs[0]:
        left, right = st.columns(2)
        with left:
            st.markdown('**A. Receiver operating characteristic**')
            roc = data['roc_curve']
            fig = curve(roc['fpr'], roc['tpr'], f"RF (AUC {data['roc_auc']:.3f})", '1 - Specificity', 'Sensitivity')
            fig.add_scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance', line=dict(dash='dot', color='#999999'))
            fig.update_xaxes(range=[0, 1]); fig.update_yaxes(range=[0, 1])
            st.plotly_chart(fig, width='stretch')
        with right:
            st.markdown('**B. Precision-recall curve**')
            pr = data['precision_recall_curve']
            fig = curve(pr['recall'], pr['precision'], f"RF (AP {data['average_precision']:.3f})", 'Recall (sensitivity)', 'Precision (PPV)')
            fig.add_scatter(x=[0, 1], y=[data['test_prevalence']]*2, mode='lines', name='Prevalence', line=dict(dash='dot', color='#999999'))
            fig.update_xaxes(range=[0, 1]); fig.update_yaxes(range=[0, 1])
            st.plotly_chart(fig, width='stretch')
        left, right = st.columns(2)
        with left:
            st.markdown('**C. Calibration plot**')
            cal = data['calibration']
            fig = curve(cal['predicted'], cal['observed'], 'RF (10 uniform bins)', 'Mean predicted probability', 'Observed positive fraction')
            fig.add_scatter(x=[0, 1], y=[0, 1], mode='lines', name='Ideal', line=dict(dash='dot', color='#999999'))
            fig.update_xaxes(range=[0, 1]); fig.update_yaxes(range=[0, 1])
            st.plotly_chart(fig, width='stretch')
            st.caption(f"Brier score: {data['brier_score']:.4f}. This plot evaluates calibration; it does not recalibrate the model.")
        with right:
            st.markdown('**D. Exploratory decision curve**')
            table = pd.DataFrame(data['threshold_analysis'])
            fig = curve(table.threshold, table.net_benefit, 'Random Forest', 'Threshold probability', 'Net benefit (per observation)')
            fig.add_scatter(x=table.threshold, y=table.treat_all, mode='lines', name='Treat all (reference)', line=dict(dash='dash', color='#648e54'))
            fig.add_scatter(x=table.threshold, y=table.treat_none, mode='lines', name='Treat none (reference)', line=dict(dash='dot', color='#999999'))
            fig.update_yaxes(range=[-.05, max(.05, data['test_prevalence'] * 1.2)])
            st.plotly_chart(fig, width='stretch')
            st.caption('Display clips very negative net benefit; complete values are in the table. NB = TP/N - FP/N * threshold/(1-threshold).')
        st.warning('Decision curves are exploratory, observation-weighted summaries with uncalibrated scores. Repeated observations are not independent patients. These curves do not establish treatment benefit or select a clinical threshold.')
    with tabs[1]:
        summary = data.get('shap_summary')
        if not summary:
            st.info('SHAP is not available in these artifacts. Install requirements and run python training/refresh_analytics.py, then restart the backend.')
        else:
            importance = pd.Series(summary['mean_absolute']).sort_values()
            points = pd.DataFrame(summary['points'])
            # Fixed jitter separates binned densities; never changes the SHAP x coordinate.
            positions = {f: i for i, f in enumerate(importance.index)}
            points['Position'] = points.Feature.map(positions) + np.random.default_rng(42).uniform(-.22, .22, len(points))
            fig = px.scatter(points, x='SHAP', y='Position', color='Value', size='Count', size_max=14,
                             color_continuous_scale=['#f6c744', '#de6981', '#7139b5'], range_color=[0, 1],
                             hover_data={'Feature': True, 'Count': True, 'Position': False},
                             labels={'SHAP': 'SHAP value (impact on positive-class model score)', 'Value': 'Relative feature value'})
            fig.update_layout(template='plotly_white', height=740, yaxis=dict(tickvals=list(positions.values()), ticktext=list(positions), title=''),
                              coloraxis_colorbar=dict(tickvals=[0, 1], ticktext=['Low', 'High']))
            fig.add_vline(x=0, line_color='#888888', line_width=1)
            st.plotly_chart(fig, width='stretch')
            st.caption(f"Exact TreeSHAP on a fixed random sample of {summary['sample_size']} held-out observations, after median imputation. Points are binned; size indicates count. Colors are normalized within each feature. Only the project's 21 features are used. Attribution is not clinical causation.")
    with tabs[2]:
        report = pd.DataFrame({key: value for key, value in data['classification_report'].items()
                               if isinstance(value, dict)}).T
        st.markdown('**Classification report**')
        st.dataframe(report, width='stretch')
        st.download_button('Download classification report', report.to_csv(), 'classification_report.csv', 'text/csv')
        scores = pd.DataFrame([[data[k] for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']]],
                              index=['Held-out patients'], columns=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'])
        st.markdown('**Held-out performance heatmap**')
        fig = px.imshow(scores, text_auto='.3f', zmin=0, zmax=1, color_continuous_scale='Teal')
        fig.update_layout(height=250)
        st.plotly_chart(fig, width='stretch')
        st.caption('One evaluated model and one dataset. Multi-hospital comparisons and early-warning time estimates are not available.')
        st.markdown('**Threshold and decision-curve table**')
        st.caption('Exploratory only. Browsing thresholds here does not change live predictions; threshold selection requires a separate validation set.')
        st.dataframe(table, hide_index=True, width='stretch')
        st.download_button('Download threshold analysis', table.to_csv(index=False), 'threshold_analysis.csv', 'text/csv')
