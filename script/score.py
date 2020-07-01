import pandas as pd
import numpy as np
from evaluator import WRMSSEEvaluator

preds = pd.read_csv("../sub/sub_v8_store_id_3.159.csv")
preds = preds[preds.id.str.contains("validation")].iloc[:,1:].to_numpy()
# preds = preds[preds.id.str.contains("evaluation")]

# preds1 = pd.read_csv("../sub/sub_v2_dept_id_0.488.csv")
# preds2 = pd.read_csv("../sub/sub_v2_store_id_0.474.csv")
# preds3 = pd.read_csv("../sub/sub_v2_dept_store_id_0.489.csv")

# preds4 = pd.read_csv("../sub/sub_v6_dept_id_0.670.csv")
# preds5 = pd.read_csv("../sub/sub_v6_store_id_0.622.csv")
# preds6 = pd.read_csv("../sub/sub_v6_dept_store_id_0.616.csv")

# preds1 = preds1[preds1.id.str.contains("validation")]
# preds2 = preds2[preds2.id.str.contains("validation")]
# preds3 = preds3[preds3.id.str.contains("validation")]
# preds4 = preds4[preds4.id.str.contains("validation")]
# preds5 = preds5[preds5.id.str.contains("validation")]
# preds6 = preds6[preds6.id.str.contains("validation")]

# preds = (preds1.iloc[:,1:].to_numpy()*0.0 +
#          preds2.iloc[:,1:].to_numpy()*0.5 + 
#          preds3.iloc[:,1:].to_numpy()*0.0 + 
#          preds4.iloc[:,1:].to_numpy()*0.1 + 
#          preds5.iloc[:,1:].to_numpy()*0.3 +
#          preds6.iloc[:,1:].to_numpy()*0.1)


evaluator = WRMSSEEvaluator()
print(evaluator.score(preds))
