from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, average_precision_score
from training.research_analysis import research_metrics
from config import DECISION_THRESHOLD

def evaluate(model, X, y, threshold=DECISION_THRESHOLD):
    probability = model.predict_proba(X)[:, list(model.classes_).index(1)]
    prediction = (probability >= threshold).astype(int)
    fpr,tpr,thresholds=roc_curve(y,probability)
    return {**research_metrics(y, probability),
            'decision_threshold': float(threshold),
            'accuracy':accuracy_score(y,prediction),
            'precision':precision_score(y,prediction,zero_division=0),
            'recall':recall_score(y,prediction,zero_division=0),
            'f1':f1_score(y,prediction,zero_division=0),
            'roc_auc':roc_auc_score(y,probability),
            'average_precision':average_precision_score(y,probability),
            'confusion_matrix':confusion_matrix(y,prediction,labels=[0,1]).tolist(),
            'classification_report':classification_report(y,prediction,labels=[0,1],target_names=['No Sepsis','Sepsis'],output_dict=True,zero_division=0),
            'roc_curve':{'fpr':fpr.tolist(),'tpr':tpr.tolist()},
            'test_prevalence':float(y.mean()),'majority_baseline_accuracy':float(max(y.mean(),1-y.mean()))}
