import pandas as pd
import numpy as np
from evaluator import WRMSSEEvaluator

valid_preds = pd.read_csv("../sub/submission_v2_2.622.csv")
valid_preds = valid_preds[valid_preds.id.str.contains("validation")]
evaluator = WRMSSEEvaluator()
print(evaluator.score(valid_preds.iloc[:,1:].to_numpy()))
