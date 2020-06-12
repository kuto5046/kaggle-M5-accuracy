import json
import requests
import numpy as np
import random

def send_slack_notification(message):
    webhook_url = 'https://hooks.slack.com/services/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'  # 終わったら無効化する
    data = json.dumps({'text': message})
    headers = {'content-type': 'application/json'}
    requests.post(webhook_url, data=data, headers=headers)


def send_slack_error_notification(message):
    webhook_url = 'https://hooks.slack.com/services/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'  # 終わったら無効化する
    data = json.dumps({"text":":no_entry_sign:" + message})
    headers = {'content-type': 'application/json'}
    requests.post(webhook_url, data=data, headers=headers)


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
