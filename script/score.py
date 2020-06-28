import pandas as pd
import numpy as np
from evaluator import WRMSSEEvaluator

valid_preds = pd.read_csv("../sub/stage2/sub_v1_dept_id_2.853.csv")
# valid_preds = valid_preds[valid_preds.id.str.contains("validation")]
valid_preds = valid_preds[valid_preds.id.str.contains("evaluation")]

evaluator = WRMSSEEvaluator()
print(evaluator.score(valid_preds.iloc[:,1:].to_numpy()))
