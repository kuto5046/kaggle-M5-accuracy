import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from evaluator import WRMSSEEvaluator


# 与えられた重みでscoreを計算する関数
def score_func(weights):
    ''' scipy minimize will pass the weights as a numpy array '''
    final_pred = 0
    for weight, pred in zip(weights, preds):
            final_pred += weight*pred.iloc[:,1:]

    final_pred.insert(0, "id", valid_preds1["id"])
    final_pred = final_pred[final_pred.id.str.contains("validation")]

    return evaluator.score(final_pred.iloc[:, 1:].to_numpy())


def submission(pred, score):

    sub = pd.read_csv('../input/m5-forecasting-accuracy/sample_submission.csv')[['id']]
    sub = sub.merge(pred, on=['id'], how='left').fillna(0)
    sub.to_csv('../sub/sub_ensemble_' + str(round(score, 3)) + "_" + str(END_TRAIN) + '.csv', index=False)


END_TRAIN = 1941
valid_preds1 = pd.read_csv("../sub/stage2/sub_v1_dept_id_0.802.csv")
valid_preds2 = pd.read_csv("../sub/stage2/sub_v1_store_id_0.835.csv")
valid_preds3 = pd.read_csv("../sub/stage2/sub_v1_dept_store_id_0.772.csv")
# valid_preds1 = pd.read_csv("../sub/sub_v2_dept_id_0.488.csv")
# valid_preds2 = pd.read_csv("../sub/sub_v2_store_id_0.474.csv")
# valid_preds3 = pd.read_csv("../sub/sub_v2_dept_store_id_0.489.csv")
preds = [valid_preds1, valid_preds2, valid_preds3]

if END_TRAIN == 1913:

    lls = []
    weights = []
    evaluator = WRMSSEEvaluator()

    # Optimization runs 100 times.
    for _ in tqdm(range(100)):
        starting_values = np.random.uniform(size=len(preds))
        # cons are given as constraints.
        cons = ({'type':'eq','fun':lambda w: 1-sum(w)})
        bounds = [(0,1)]*len(preds)
        
        res = minimize(score_func, starting_values, constraints=cons,
                    bounds = bounds, method='SLSQP')

        lls.append(res['fun'])
        weights.append(res['x'])
    bestSC = np.min(lls)
    bestWeight = weights[np.argmin(lls)]
    print('\n Ensemble Score: {best_score:.7f}'.format(best_score=bestSC))
    print('\n Best Weights: {weights:}'.format(weights=bestWeight))

else:
    bestSC = 0
    bestWeight = [0.1, 0.9, 0.0]


ensemble_pred = 0
for weight, pred in zip(bestWeight, preds):
    ensemble_pred += weight*pred.iloc[:, 1:]

ensemble_pred.insert(0, "id", valid_preds1.id)
submission(ensemble_pred, bestSC)














