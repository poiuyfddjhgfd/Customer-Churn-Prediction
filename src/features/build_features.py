import pandas as pd

def _map_binary_series(s: pd.Series) -> pd.Series:
    """
    Apply deterministic mapping to binary encoding to 2-category features.
    This Function implement the core binary encoding logic that converts 
    categorical features with two unique values into 0/1 integers. The mapping are 
    deterministics and must be consistent between training and serving
    """

    # Get unique values and remove NaN
    vals=list(pd.Series(s.dropna().unique()).astype(str))
    valset=set(vals)

    # === Deterministic Binary Mapping===
    # Critical : These exact mappings are hardcoded in serving pipeline
    # 
    # Yes/No mapping (most common pattern in telecom data) 
    if valset=={"Yes","No"}:
        return s.map({"Yes":1,"No":0}).astype("Int64")
    
    # Gender mapping (demographic feature)
    if valset=={"Female","Male"}:
        return s.map({"Female":0,"Male":1}).astype("Int64")
    
    # === GENERIC BINARY MAPPING===
    # For any other 2-category feature, use stable alphabetical ordering
    if len(vals)==2:
        # Sort Values to ensure consistent mapping across runs
        sorted_vals=sorted(vals)
        mapping={sorted_vals[0]:0,sorted_vals[1]:1}
        return s.astype(str).map(mapping).astype("Int64")
    
    # === NON BINARY FEATURES
    # RETURN UNCHANGED -WILL BE HANDLED BY ONE -HOT ENCODING
    return s

def build_features(df: pd.DataFrame,target_col: str="Churn")-> pd.DataFrame:
    """
    Apply complete features engineering pipelines for training data.
    This is the main feature engineering function that transform raw customer data 
    into ML-ready features. The Transformations must be exactly replicated in the serving pipeline to ensure prediction accuracy.
    """

    df=df.copy()
    print(f"Starting feature engineering on {df.shape[1]} columns...")

    # === STEP 1: Identify Feature Types===
    # Find categorical columns(object dtype) excluding the target variable
    obj_cols=[c for c in df.select_dtypes(include=["object"]).columns if c!=target_col]
    numerical_cols=df.select_dtypes(include=["int64","float64"]).columns.tolist()

    print(f"Found {len(obj_cols)} categorical and {len(numerical_cols)} numerical features.")

    # == Step2: Split Categorical by Cardinality==
    # Binary Features (exactly 2 unique values) get binary encoding
    # Multi category features(more than 2 unique values ) get one - hot encoding
    binary_cols=[c for c in obj_cols if df[c].dropna().nunique()==2]
    multi_cols=[c for c in obj_cols if df[c].dropna().nunique()>2]

    print(f"Binary Feature :{len(binary_cols)}| Multi-category features:{len(multi_cols)}")
    if binary_cols:
        print(f"Binary:{binary_cols}")
    if multi_cols:
        print(f"Multi-category:{multi_cols}")

    #=== STEP 3# : Apply Binary Encoding=== 
    #  Convert binary category featuresto 0/1 using deterministics mapping 
    for c in binary_cols:
        original_dtype=df[c].dtype
        df[c]=_map_binary_series(df[c].astype(str))
        print(f"{c}:{original_dtype}-> binary(0/1)")      

    # === STEP 4: Convert Boolean Columns===
    # XGBoost require integer input not boolean
    bool_cols=df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols]=df[bool_cols].astype(int)
        print(f"Converted {len(bool_cols)} boolean columns to int:{bool_cols}")

    # == Step5: One Hot Encoding with drop_first=True (same as serving)
    df=pd.get_dummies(df,columns=multi_cols,drop_first=True)
    new_features=df.shape[1]-original_shape[1]+len(multi_cols)
    print(f"Created {new_features} new features from {len(multi_cols)} catgorical columns")

    # === Step 6: DATA TYPE CLEANUP===
    # Convert nullable integer to standard integer for XGBOOST 
    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            # Fill any Nan Values with 0 and convert to int
            df[c]=df[c].fillna(0).astype(int)
            
    print(f"Feature engineering  complete:{df.shape[1]} final features")
    return df        