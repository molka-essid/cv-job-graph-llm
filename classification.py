"""
Node Classification Module
Classifies CV and Job nodes using multiple ML approaches.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.semi_supervised import LabelPropagation
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings("ignore")


def classify_cv_nodes(cvs, features_dict, target="level"):
    """
    Classify CV nodes.
    Targets: 'level' (junior/intermediate/senior) or 'domain' or 'polyvalent'.
    """
    results = {}
    
    node_ids = [cv["id"] for cv in cvs if cv["id"] in features_dict]
    X = np.array([features_dict[nid] for nid in node_ids])
    
    cv_lookup = {cv["id"]: cv for cv in cvs}
    
    if target == "level":
        y_raw = [cv_lookup[nid]["level"] for nid in node_ids]
    elif target == "domain":
        y_raw = [cv_lookup[nid]["domain"] for nid in node_ids]
    elif target == "polyvalent":
        y_raw = ["polyvalent" if cv_lookup[nid]["polyvalent"] else "spécialisé" for nid in node_ids]
    else:
        y_raw = [cv_lookup[nid]["level"] for nid in node_ids]
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=50, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
    }
    
    cv_splits = min(5, len(np.unique(y)))  # Avoid too many splits for small classes
    skf = StratifiedKFold(n_splits=max(2, cv_splits), shuffle=True, random_state=42)
    
    model_results = {}
    best_model = None
    best_score = -1
    
    for name, clf in models.items():
        try:
            scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring="accuracy")
            clf.fit(X_scaled, y)
            y_pred = clf.predict(X_scaled)
            report = classification_report(y, y_pred, target_names=le.classes_, output_dict=True)
            
            model_results[name] = {
                "cv_accuracy_mean": float(scores.mean()),
                "cv_accuracy_std": float(scores.std()),
                "train_accuracy": float(accuracy_score(y, y_pred)),
                "report": report,
                "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
                "classes": list(le.classes_),
            }
            
            if scores.mean() > best_score:
                best_score = scores.mean()
                best_model = name
        except Exception as e:
            model_results[name] = {"error": str(e)}
    
    # Semi-supervised
    try:
        n_labeled = max(int(len(y) * 0.5), 3)
        y_semi = y.copy().astype(float)
        unlabeled_idx = np.random.choice(len(y), size=len(y) - n_labeled, replace=False)
        y_semi[unlabeled_idx] = -1
        
        lp = LabelPropagation(kernel="rbf", gamma=0.1, max_iter=1000)
        lp.fit(X_scaled, y_semi.astype(int))
        y_pred_semi = lp.predict(X_scaled)
        acc_semi = accuracy_score(y, y_pred_semi)
        
        model_results["Label Propagation (Semi-Supervisé)"] = {
            "cv_accuracy_mean": acc_semi,
            "cv_accuracy_std": 0.0,
            "train_accuracy": acc_semi,
            "classes": list(le.classes_),
            "note": f"50% labeled, 50% unlabeled"
        }
    except Exception as e:
        model_results["Label Propagation (Semi-Supervisé)"] = {"error": str(e)}
    
    results = {
        "target": target,
        "n_samples": len(node_ids),
        "classes": list(le.classes_),
        "models": model_results,
        "best_model": best_model,
        "best_score": best_score,
        "node_ids": node_ids,
        "y_true": y_raw,
        "le": le,
    }
    return results


def classify_job_nodes(jobs, features_dict, target="level"):
    """Classify Job nodes by level or domain."""
    node_ids = [job["id"] for job in jobs if job["id"] in features_dict]
    X = np.array([features_dict[nid] for nid in node_ids])
    
    job_lookup = {job["id"]: job for job in jobs}
    
    if target == "level":
        y_raw = [job_lookup[nid]["level"] for nid in node_ids]
    else:
        y_raw = [job_lookup[nid]["domain"] for nid in node_ids]
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    skf = StratifiedKFold(n_splits=min(3, len(np.unique(y))), shuffle=True, random_state=42)
    
    try:
        scores = cross_val_score(clf, X_scaled, y, cv=skf, scoring="accuracy")
        clf.fit(X_scaled, y)
        y_pred = clf.predict(X_scaled)
        
        importances = None
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_.tolist()
        
        return {
            "target": target,
            "n_samples": len(node_ids),
            "classes": list(le.classes_),
            "cv_accuracy_mean": float(scores.mean()),
            "cv_accuracy_std": float(scores.std()),
            "train_accuracy": float(accuracy_score(y, y_pred)),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "node_ids": node_ids,
            "y_true": y_raw,
            "y_pred": [le.classes_[p] for p in y_pred],
            "feature_importances": importances,
        }
    except Exception as e:
        return {"error": str(e)}