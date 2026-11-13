import hashlib
import io
import json
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import clone
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, IsolationForest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, TimeSeriesSplit, RandomizedSearchCV, ParameterGrid, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score, silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.inspection import permutation_importance

SEED = 42
JOBS = 2
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    HAS_SM = True
except ImportError:
    HAS_SM = False

st.set_page_config(page_title="Machine Learning Intuition Lab", layout="wide")

def prep(X, scale=True):
    nums = X.select_dtypes(include=np.number).columns.tolist()
    cats = [c for c in X if c not in nums]
    num_steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(num_steps), nums),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats),
    ])

def model_X(df, columns):
    X = df[columns].copy()
    for c in X:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].where(X[c].notna(), "Missing").astype(str)
    return X.replace([np.inf, -np.inf], np.nan)

def candidates(clf):
    if clf:
        return {
            "Logistic Regression": (LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED), {"model__C": [.1, 1, 10]}),
            "Decision Tree": (DecisionTreeClassifier(random_state=SEED, class_weight="balanced"), {"model__max_depth": [3, 7, None], "model__min_samples_leaf": [1, 5, 20]}),
            "Random Forest": (RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=1), {"model__max_depth": [None, 10], "model__min_samples_leaf": [1, 5]}),
            "Extra Trees": (ExtraTreesClassifier(n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=1), {"model__max_depth": [None, 10], "model__min_samples_leaf": [1, 5]}),
            "HistGradient Boosting": (HistGradientBoostingClassifier(random_state=SEED), {"model__learning_rate": [.05, .1], "model__max_leaf_nodes": [15, 31]}),
            "SVM": (SVC(probability=False, class_weight="balanced", random_state=SEED), {"model__C": [.1, 1, 5], "model__kernel": ["linear", "rbf"]}),
            "KNN": (KNeighborsClassifier(), {"model__n_neighbors": [3, 5, 11]}),
            "Naive Bayes": (GaussianNB(), {"model__var_smoothing": [1e-9, 1e-7]}),
        }
    return {
        "Ridge": (Ridge(), {"model__alpha": [.1, 1, 10, 100]}),
        "Decision Tree": (DecisionTreeRegressor(random_state=SEED), {"model__max_depth": [3, 7, None], "model__min_samples_leaf": [1, 5, 20]}),
        "Random Forest": (RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=1), {"model__max_depth": [None, 10], "model__min_samples_leaf": [1, 5]}),
        "Extra Trees": (ExtraTreesRegressor(n_estimators=200, random_state=SEED, n_jobs=1), {"model__max_depth": [None, 10], "model__min_samples_leaf": [1, 5]}),
        "HistGradient Boosting": (HistGradientBoostingRegressor(random_state=SEED), {"model__learning_rate": [.05, .1], "model__max_leaf_nodes": [15, 31]}),
        "SVM": (TransformedTargetRegressor(regressor=SVR(), transformer=StandardScaler()), {"model__regressor__C": [.1, 1, 5]}),
        "KNN": (KNeighborsRegressor(), {"model__n_neighbors": [3, 5, 11]}),
    }

def cv_for(y, clf, folds):
    if clf:
        smallest = int(pd.Series(y).value_counts().min())
        if smallest < folds:
            raise ValueError(f"Each training class needs at least {folds} rows for the requested inner folds; smallest has {smallest}.")
        return StratifiedKFold(folds, shuffle=True, random_state=SEED)
    if len(y) < 2 * folds:
        raise ValueError("Not enough training rows for the requested folds.")
    return KFold(folds, shuffle=True, random_state=SEED)

def choose_model(X, y, clf, names, folds, budget, tune):
    scoring = "balanced_accuracy" if clf else "neg_root_mean_squared_error"
    cv = cv_for(y, clf, folds)
    rows, fitted = [], {}
    for name in names:
        start = time.perf_counter()
        est, grid = candidates(clf)[name]
        pipe = Pipeline([("prep", prep(X)), ("model", est)])
        try:
            if tune:
                search = RandomizedSearchCV(pipe, grid, n_iter=min(budget, len(ParameterGrid(grid))), cv=cv, scoring=scoring, n_jobs=JOBS, random_state=SEED, refit=True, error_score=np.nan)
                search.fit(X, y)
                score, std = float(search.best_score_), float(search.cv_results_["std_test_score"][search.best_index_])
                fit = search.best_estimator_
                params = search.best_params_
            else:
                scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=JOBS, error_score="raise")
                score, std = float(scores.mean()), float(scores.std())
                fit, params = pipe.fit(X, y), {}
            if not np.isfinite(score):
                raise ValueError("All scores were non-finite.")
            fitted[name] = fit
            rows.append({"Algorithm": name, "Selection score": score, "Fold std": std, "Seconds": time.perf_counter()-start, "Settings": str(params), "Error": ""})
        except Exception as e:
            rows.append({"Algorithm": name, "Selection score": np.nan, "Fold std": np.nan, "Seconds": time.perf_counter()-start, "Settings": "", "Error": str(e)[:250]})
    board = pd.DataFrame(rows).sort_values("Selection score", ascending=False, na_position="last")
    if not fitted:
        raise ValueError("All algorithms failed: " + "; ".join(board["Error"].tolist()))
    winner = board.iloc[0]["Algorithm"]
    return fitted[winner], winner, board

def scores(y, p, clf):
    if clf:
        return {"Accuracy": accuracy_score(y,p), "Balanced accuracy": balanced_accuracy_score(y,p), "Macro F1": f1_score(y,p,average="macro",zero_division=0)}
    return {"RMSE": float(np.sqrt(mean_squared_error(y,p))), "MAE": mean_absolute_error(y,p), "R2": r2_score(y,p)}

def show_metrics(m):
    cols = st.columns(len(m))
    for c, (k,v) in zip(cols,m.items()):
        c.metric(k, f"{v:.4f}")

def export_panel(frame, report, key):
    if st.button("Prepare downloads", key=key+"_prepare"):
        st.session_state[key] = (frame.to_csv(index=False).encode("utf-8-sig"), report)
    if key in st.session_state:
        data, text = st.session_state[key]
        st.download_button("Download results CSV", data, key+".csv", "text/csv", key=key+"_csv")
        st.download_button("Download methodology", text, key+".md", "text/markdown", key=key+"_md")

def supervised(df):
    st.subheader("Supervised learning")
    st.caption("Predict a known target. Features are explicitly selected; no target-based ranking is performed before validation.")
    target = st.selectbox("Target", df.columns)
    clf = st.radio("Task", ["Classification", "Regression"], horizontal=True)=="Classification"
    features = st.multiselect("Features: exclude IDs, target copies, and information unavailable at prediction time", [c for c in df if c != target], default=[c for c in df if c != target][:5])
    names = st.multiselect("Algorithms", list(candidates(clf)), default=list(candidates(clf))[:3])
    strategy = st.selectbox("Evaluation", ["80/20 holdout", "70/30 holdout", "Nested cross-validation"])
    folds = st.slider("Inner tuning folds", 2, 10, 5)
    outer = st.slider("Outer evaluation folds", 2, 10, 5) if strategy.startswith("Nested") else 5
    budget = st.slider("Maximum parameter candidates per algorithm", 1, 20, 6)
    tune = st.checkbox("Tune parameters", True)
    st.info("Nested CV selects both algorithm and settings inside every outer training fold. It can be expensive. No hidden row caps or fold reductions are applied.")
    if st.button("Train supervised model", type="primary"):
        st.session_state.pop("supervised_result",None)
        st.session_state.pop("sup_export",None)
        if not features or not names:
            st.error("Select features and algorithms.")
        else:
            try:
                work = df.dropna(subset=[target]).copy()
                encoder = None
                if clf:
                    encoder = LabelEncoder()
                    y = encoder.fit_transform(work[target].astype(str))
                    if len(encoder.classes_) < 2:
                        raise ValueError("At least two classes are required.")
                else:
                    valid = pd.to_numeric(work[target],errors="coerce").replace([np.inf,-np.inf],np.nan)
                    work = work.loc[valid.notna()]
                    y = valid.loc[valid.notna()].to_numpy(dtype=float)
                X = model_X(work,features)
                if len(X)<30:
                    raise ValueError("At least 30 usable rows are required.")
                with st.spinner("Training and evaluating..."):
                    base = DummyClassifier(strategy="most_frequent") if clf else DummyRegressor(strategy="mean")
                    if strategy.startswith("Nested"):
                        cv = cv_for(y,clf,outer)
                        pred = np.empty(len(y), dtype=int if clf else float)
                        bp = np.empty_like(pred)
                        fold_rows=[]
                        for f,(tr,te) in enumerate(cv.split(X,y),1):
                            model, winner, _ = choose_model(X.iloc[tr],y[tr],clf,names,folds,budget,tune)
                            pred[te]=model.predict(X.iloc[te])
                            bp[te]=clone(base).fit(X.iloc[tr],y[tr]).predict(X.iloc[te])
                            fold_rows.append({"Fold":f,"Selected algorithm":winner,**scores(y[te],pred[te],clf)})
                        evaluation="Out-of-fold evaluation of the complete algorithm-selection procedure"
                        Xe, ye = X,y
                        Xt,yt=X,y
                    else:
                        tr,te=train_test_split(np.arange(len(y)),test_size=.2 if strategy.startswith("80") else .3,random_state=SEED,stratify=y if clf else None)
                        Xt,yt=X.iloc[tr],y[tr]
                        Xe,ye=X.iloc[te],y[te]
                        model,winner,board=choose_model(Xt,yt,clf,names,folds,budget,tune)
                        pred=model.predict(Xe)
                        bp=base.fit(Xt,yt).predict(Xe)
                        fold_rows=[]
                        evaluation="Untouched holdout evaluation; do not repeatedly optimize against this set"
                    if strategy.startswith("Nested"):
                        model,winner,board=choose_model(X,y,clf,names,folds,budget,tune)
                    out=work.loc[Xe.index].copy()
                    out.insert(0,"Source row",Xe.index)
                    out["Actual"]=encoder.inverse_transform(ye.astype(int)) if clf else ye
                    out["Predicted"]=encoder.inverse_transform(pred.astype(int)) if clf else pred
                    st.session_state.supervised_result=dict(model=model,winner=winner,board=board,metrics=scores(ye,pred,clf),baseline=scores(ye,bp,clf),evaluation=evaluation,out=out,Xe=Xe,ye=ye,Xt=Xt,yt=yt,clf=clf,encoder=encoder,pred=pred,fold_rows=fold_rows,folds=folds,diagnostics={})
            except Exception as e:
                st.error(str(e))
    r=st.session_state.get("supervised_result")
    if r:
        st.caption("Stored run; changing controls does not retrain. Press Train to replace it.")
        st.success("Final selected algorithm: "+r["winner"])
        st.write(r["evaluation"])
        show_metrics(r["metrics"])
        st.write("Baseline on the same evaluation rows",r["baseline"])
        st.dataframe(r["board"],hide_index=True)
        st.caption("Selection score is balanced accuracy for classification and negative RMSE for regression. Higher is better. These are selection scores, not another final test.")
        if r["fold_rows"]:
            st.dataframe(pd.DataFrame(r["fold_rows"]),hide_index=True)
        if r["clf"]:
            st.dataframe(pd.DataFrame(classification_report(r["ye"],r["pred"],zero_division=0,output_dict=True)).T)
        else:
            st.plotly_chart(px.scatter(r["out"].sample(min(3000,len(r["out"])),random_state=SEED),x="Actual",y="Predicted"),use_container_width=True)
        st.subheader("Optional diagnostics")
        st.caption("Importance uses the final model: in nested-CV mode it is in-sample and descriptive, not out-of-fold importance. Do not use holdout diagnostics to repeatedly select features without a fresh final evaluation.")
        if st.button("Compute permutation importance"):
            sample=r["Xe"].sample(min(1000,len(r["Xe"])),random_state=SEED)
            pos=r["Xe"].index.get_indexer(sample.index)
            pi=permutation_importance(r["model"],sample,r["ye"][pos],scoring="balanced_accuracy" if r["clf"] else "neg_root_mean_squared_error",n_repeats=3,random_state=SEED,n_jobs=1)
            r["diagnostics"]["importance"]=pd.DataFrame({"Feature":sample.columns,"Score drop":pi.importances_mean})
        if "importance" in r["diagnostics"]:
            st.dataframe(r["diagnostics"]["importance"].sort_values("Score drop",ascending=False))
        if st.button("Compute learning curve"):
            try:
                sizes,tr,va=learning_curve(clone(r["model"]),r["Xt"],r["yt"],cv=cv_for(r["yt"],r["clf"],r["folds"]),scoring="balanced_accuracy" if r["clf"] else "neg_root_mean_squared_error",train_sizes=[.5,.75,1.],n_jobs=JOBS,shuffle=True,random_state=SEED,error_score="raise")
                r["diagnostics"]["curve"]=pd.DataFrame({"Rows":sizes,"Training":tr.mean(axis=1),"Validation":va.mean(axis=1)})
            except Exception as e:
                st.error(str(e))
        if "curve" in r["diagnostics"]:
            st.line_chart(r["diagnostics"]["curve"].set_index("Rows"))
        report="# Supervised methodology\n"+r["evaluation"]+"\nFinal model: "+r["winner"]+"\nEvaluation: "+str(r["metrics"])+"\nBaseline: "+str(r["baseline"])+"\nPreprocessing is fitted within training folds. Class weights are enabled only for supported candidates. Numeric targets retain floating-point precision.\n"
        export_panel(r["out"],report,"sup_export")

def umodel(name,k,eps,minimum):
    return {"K-Means":lambda:KMeans(n_clusters=k,n_init=10,random_state=SEED),"Mini-Batch K-Means":lambda:MiniBatchKMeans(n_clusters=k,n_init=10,random_state=SEED),"Gaussian Mixture":lambda:GaussianMixture(n_components=k,random_state=SEED),"Birch":lambda:Birch(n_clusters=k),"Hierarchical":lambda:AgglomerativeClustering(n_clusters=k),"DBSCAN":lambda:DBSCAN(eps=eps,min_samples=minimum),"Isolation Forest":lambda:IsolationForest(random_state=SEED)}[name]()

def quality(X,labels):
    mask=labels!=-1
    a,b=X[mask],labels[mask]
    if not 2<=len(np.unique(b))<len(b):
        return {"Silhouette":np.nan,"Davies-Bouldin":np.nan}
    return {"Silhouette":silhouette_score(a,b,sample_size=min(1500,len(a)),random_state=SEED),"Davies-Bouldin":davies_bouldin_score(a,b)}

def unsupervised(df):
    st.subheader("Unsupervised discovery")
    columns=df.select_dtypes(include=np.number).columns.tolist()
    features=st.multiselect("Numeric discovery features",columns,default=columns[:3])
    name=st.selectbox("Discovery algorithm",["K-Means","Mini-Batch K-Means","Gaussian Mixture","Birch","Hierarchical","DBSCAN","Isolation Forest"])
    k=st.slider("Clusters",2,10,3)
    eps=st.number_input("DBSCAN eps",.01,100.,.5)
    minimum=st.slider("DBSCAN minimum neighbors",2,30,5)
    scale=st.checkbox("Standardize discovery features",True)
    mode=st.selectbox("Discovery validation",["80/20 holdout","Cross-validation"])
    folds=st.slider("Discovery folds",2,10,5)
    limit=st.number_input("Maximum rows analyzed (explicit random sample)",100,100000,5000,100)
    st.caption("Hierarchical/DBSCAN validation uses a KNN label-transfer surrogate, not native clustering prediction. Similar separation scores do not prove cluster stability or anomaly accuracy.")
    if st.button("Run discovery",type="primary"):
        st.session_state.pop("unsup_result",None)
        st.session_state.pop("unsup_export",None)
        try:
            if len(features)<2:
                raise ValueError("Select at least two numeric features.")
            data=df[features].replace([np.inf,-np.inf],np.nan)
            if data.isna().all().any():
                raise ValueError("Remove entirely missing features.")
            data=data.sample(min(int(limit),len(data)),random_state=SEED).sort_index()
            if len(data)<20:
                raise ValueError("At least 20 rows required.")
            transform=lambda:Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler() if scale else "passthrough")])
            splits=KFold(folds,shuffle=True,random_state=SEED).split(data) if mode=="Cross-validation" else [train_test_split(np.arange(len(data)),test_size=.2,random_state=SEED)]
            checks=[]
            with st.spinner("Fitting discovery and validation..."):
                for fold,(tr,te) in enumerate(splits,1):
                    pp=transform()
                    a=pp.fit_transform(data.iloc[tr]); b=pp.transform(data.iloc[te])
                    m=umodel(name,k,eps,minimum)
                    la=m.fit_predict(a)
                    lb=m.predict(b) if hasattr(m,"predict") else KNeighborsClassifier(n_neighbors=min(5,len(a))).fit(a,la).predict(b)
                    if name=="Isolation Forest":
                        checks.append({"Fold":fold,"Training anomaly rate":(la==-1).mean(),"Held-out anomaly rate":(lb==-1).mean()})
                    else:
                        checks.append({"Fold":fold,**{"Training "+key:v for key,v in quality(a,la).items()},**{"Held-out "+key:v for key,v in quality(b,lb).items()}})
                pp=transform(); X=pp.fit_transform(data)
                model=umodel(name,k,eps,minimum); labels=model.fit_predict(X)
                from sklearn.decomposition import PCA
                pc=PCA(n_components=2).fit_transform(X)
                out=data.copy(); out.insert(0,"Source row",data.index)
                out["Group"]=labels.astype(str);out["PC1"]=pc[:,0];out["PC2"]=pc[:,1]
                st.session_state.unsup_result=dict(out=out,X=X,labels=labels,checks=pd.DataFrame(checks),name=name,features=features,elbow=None,quality=quality(X,labels) if name!="Isolation Forest" else {"Anomaly rate":float((labels==-1).mean())})
        except Exception as e:
            st.error(str(e))
    r=st.session_state.get("unsup_result")
    if r:
        st.write("Stored discovery:",r["name"],"; analyzed rows:",len(r["out"]))
        st.write(r["quality"]);st.dataframe(r["checks"])
        st.caption("Noise (-1) is excluded from cluster-quality scores. Anomaly rate is descriptive, not accuracy. PCA is only a projection.")
        plot=r["out"].sample(min(3000,len(r["out"])),random_state=SEED)
        st.plotly_chart(px.scatter(plot,x="PC1",y="PC2",color="Group"),use_container_width=True)
        st.dataframe(r["out"].groupby("Group")[r["features"]].mean())
        if r["name"] in ["K-Means","Mini-Batch K-Means"]:
            if st.button("Compute optional cluster-count comparison"):
                X=r["X"]
                if len(X)>5000:
                    X=X[np.random.RandomState(SEED).choice(len(X),5000,replace=False)]
                rows=[]
                for kk in range(2,min(10,len(X)-1,len(np.unique(X,axis=0)))+1):
                    m=KMeans(n_clusters=kk,n_init=10,random_state=SEED).fit(X)
                    rows.append({"Clusters":kk,"Inertia":m.inertia_,**quality(X,m.labels_)})
                r["elbow"]=pd.DataFrame(rows)
            if r["elbow"] is not None:
                st.dataframe(r["elbow"])
        export_panel(r["out"],"# Discovery methodology\n"+r["name"]+"\nImputation/scaling fitted separately in each validation split. Hierarchical/DBSCAN use KNN surrogate assignment. Cluster quality excludes noise.\n"+r["checks"].to_csv(index=False),"unsup_export")

FREQ={"Daily":"D","Hourly":"h","Weekly":"W","Monthly":"MS","Quarterly":"QS","Yearly":"YS"}

def lag_matrix(values,lags):
    return np.array([values[i-lags:i][::-1] for i in range(lags,len(values))]),values[lags:]

def forecast(name,train,steps,season,lags):
    v=train.to_numpy(dtype=float)
    if name=="Naive":return np.repeat(v[-1],steps)
    if name=="Seasonal Naive":
        if len(v)<season:raise ValueError("Insufficient seasonal history.")
        return np.resize(v[-season:],steps)
    if name=="Moving Average":return np.repeat(v[-min(season,len(v)):].mean(),steps)
    if name=="Holt-Winters":
        return np.asarray(ExponentialSmoothing(train,trend="add",seasonal="add" if season>1 else None,seasonal_periods=season if season>1 else None).fit().forecast(steps))
    if name=="ARIMA":return np.asarray(ARIMA(train,order=(2,1,2)).fit().forecast(steps))
    if len(v)<lags+10:raise ValueError("Insufficient history for selected lags.")
    est={"Ridge lags":Pipeline([("scale",StandardScaler()),("model",Ridge())]),"Random Forest lags":RandomForestRegressor(n_estimators=200,random_state=SEED,n_jobs=1),"HistGradient lags":HistGradientBoostingRegressor(random_state=SEED)}[name]
    X,y=lag_matrix(v,lags);est.fit(X,y)
    history=list(v);out=[]
    for _ in range(steps):
        p=float(est.predict(np.array(history[-lags:][::-1]).reshape(1,-1))[0]);out.append(p);history.append(p)
    return np.array(out)

def timeseries(df):
    st.subheader("Time-series forecasting")
    date=st.selectbox("Date column",df.columns)
    nums=df.select_dtypes(include=np.number).columns.tolist()
    if not nums:
        st.warning("A numeric value column is required.");return
    value=st.selectbox("Value column",nums)
    freq=FREQ[st.selectbox("Frequency",list(FREQ))]
    aggregation=st.selectbox("Period aggregation",["Mean","Sum"])
    dayfirst=st.checkbox("Day-first dates",False)
    season=st.number_input("Season length",1,400,7)
    lags=st.number_input("Lag count",2,100,7)
    horizon=st.number_input("Forecast and backtest horizon",1,365,7)
    folds=st.slider("Chronological selection folds",2,10,3)
    options=["Naive","Seasonal Naive","Moving Average","Ridge lags","Random Forest lags","HistGradient lags"]+(["Holt-Winters","ARIMA"] if HAS_SM else [])
    names=st.multiselect("Forecasters",options,default=["Naive","Seasonal Naive","Ridge lags"])
    st.info("The latest horizon is reserved for final testing. Earlier rolling-origin folds select the model. Gaps in training are forward-filled using past observations only; missing actual values are excluded from scoring. No 95% interval is claimed.")
    if st.button("Train and forecast",type="primary"):
        st.session_state.pop("ts_result",None);st.session_state.pop("ts_export",None)
        try:
            if not names:raise ValueError("Select forecasters.")
            d=pd.DataFrame({"Date":pd.to_datetime(df[date],errors="coerce",dayfirst=dayfirst),"Value":pd.to_numeric(df[value],errors="coerce").replace([np.inf,-np.inf],np.nan)}).dropna(subset=["Date"]).sort_values("Date")
            if d.empty:raise ValueError("No valid dates.")
            # Guard before resampling to avoid creating an enormous time grid.
            span=(d.Date.max()-d.Date.min()).total_seconds()
            unit={"h":3600,"D":86400,"W":604800,"MS":2419200,"QS":7257600,"YS":31536000}[freq]
            if span/unit>50000:raise ValueError("Selected frequency would create over 50,000 periods; use a coarser frequency or shorter history.")
            raw=d.set_index("Date").Value.resample(freq)
            s=raw.mean() if aggregation=="Mean" else raw.sum(min_count=1)
            s=s.loc[s.first_valid_index():] if s.notna().any() else s
            h=int(horizon); inner=s.iloc[:-h]; final=s.iloc[-h:]
            if len(inner)<(folds+1)*h+int(lags)+10:raise ValueError("More history is needed for these folds, horizon, and lags.")
            splits=list(TimeSeriesSplit(n_splits=folds,test_size=h).split(inner))
            rows=[]
            with st.spinner("Running chronological model selection..."):
                for name in names:
                    try:
                        errors=[];ae=[]
                        for tr,te in splits:
                            train=inner.iloc[tr].ffill().dropna()
                            p=forecast(name,train,len(te),int(season),int(lags))
                            actual=inner.iloc[te].to_numpy();mask=np.isfinite(actual)
                            if mask.sum()<1:raise ValueError("A validation block has no observed targets.")
                            errors.extend((actual[mask]-p[mask])**2);ae.extend(np.abs(actual[mask]-p[mask]))
                        rows.append({"Algorithm":name,"Validation RMSE":float(np.sqrt(np.mean(errors))),"Validation MAE":float(np.mean(ae)),"Error":""})
                    except Exception as e:
                        rows.append({"Algorithm":name,"Validation RMSE":np.nan,"Validation MAE":np.nan,"Error":str(e)[:200]})
                board=pd.DataFrame(rows).sort_values("Validation RMSE",na_position="last")
                if not board["Validation RMSE"].notna().any():raise ValueError("All forecasters failed: "+str(board.Error.tolist()))
                winner=board.iloc[0].Algorithm
                p=forecast(winner,inner.ffill().dropna(),h,int(season),int(lags))
                baseline=forecast("Naive",inner.ffill().dropna(),h,int(season),int(lags))
                mask=final.notna().to_numpy()
                if not mask.any():raise ValueError("Final holdout has no observed values.")
                metrics=scores(final.to_numpy()[mask],p[mask],False)
                base=scores(final.to_numpy()[mask],baseline[mask],False)
                future=forecast(winner,s.ffill().dropna(),h,int(season),int(lags))
                idx=pd.date_range(s.index[-1],periods=h+1,freq=freq)[1:]
                out=pd.DataFrame({"Date":s.index,"Actual":s.values,"Observed":s.notna().values,"Partition":["Selection history"]*len(inner)+["Final holdout"]*h,"Held-out prediction":np.r_[np.full(len(inner),np.nan),p]})
                out=pd.concat([out,pd.DataFrame({"Date":idx,"Forecast":future,"Partition":"Future","Observed":False})],ignore_index=True)
                st.session_state.ts_result=dict(board=board,winner=winner,metrics=metrics,baseline=base,out=out,value=value,freq=freq,horizon=h,folds=folds)
        except Exception as e:
            st.error(str(e))
    r=st.session_state.get("ts_result")
    if r:
        st.success("Selected forecast model: "+r["winner"])
        show_metrics(r["metrics"]);st.write("Naive baseline on the same observed holdout:",r["baseline"])
        st.dataframe(r["board"],hide_index=True)
        st.plotly_chart(px.line(r["out"].tail(1000),x="Date",y=["Actual","Held-out prediction","Forecast"],title=r["value"]),use_container_width=True)
        export_panel(r["out"],"# Forecast methodology\nValue: "+r["value"]+"\nWinner: "+r["winner"]+"\nChronological rolling-origin selection with horizon-matched blocks; final holdout not used for selection. Training gaps forward-filled; missing actuals excluded from metrics. Future model refitted on all available history. No calibrated prediction intervals provided.\nFinal scores: "+str(r["metrics"])+"\nBaseline: "+str(r["baseline"]),"ts_export")

@st.cache_data(max_entries=3)
def load_csv(content,rows):
    return pd.read_csv(io.BytesIO(content),nrows=rows,low_memory=False)

st.title("Machine Learning Intuition Lab")
st.caption("Upload, inspect, configure, validate, interpret, and export. This is a streamlined replacement, not a line-for-line patch of the previous app.")
file=st.file_uploader("Upload CSV",type="csv")
row_limit=st.sidebar.number_input("Maximum CSV rows loaded",100,1000000,100000,100)
st.sidebar.caption("No automatic package installation. Keep deployment dependencies in requirements.txt.")
if file:
    content=file.getvalue(); signature=hashlib.sha256(content+str(row_limit).encode()).hexdigest()
    if st.session_state.get("dataset_signature")!=signature:
        for key in ["supervised_result","unsup_result","ts_result","sup_export","unsup_export","ts_export"]:
            st.session_state.pop(key,None)
        st.session_state.dataset_signature=signature
    try:
        df=load_csv(content,int(row_limit))
        if df.empty:raise ValueError("The CSV has no data rows.")
        if not df.columns.is_unique:raise ValueError("Column names must be unique.")
        a,b,c=st.columns(3)
        a.metric("Loaded rows",len(df));b.metric("Columns",len(df.columns));c.metric("Missing cells",int(df.isna().sum().sum()))
        st.caption("Row limit loads the first rows, not a random sample. Original numeric precision is retained. Duplicate records and repeated subjects can invalidate random-row evaluation; use an appropriate group/time split outside this app for those datasets.")
        with st.expander("Dataset inspection"):
            st.dataframe(df.head(100))
            st.write("Duplicate rows:",int(df.duplicated().sum()))
            st.dataframe(pd.DataFrame({"Type":df.dtypes.astype(str),"Missing":df.isna().sum(),"Unique":df.nunique()}))
        mode=st.radio("Analysis",["Supervised","Unsupervised","Time series"],horizontal=True)
        if mode=="Supervised":supervised(df)
        elif mode=="Unsupervised":unsupervised(df)
        else:timeseries(df)
    except Exception as e:
        st.error(str(e))
