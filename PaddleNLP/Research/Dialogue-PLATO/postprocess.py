import json
for data in ['DSTC7_Ubuntu','DailyDialog']:
  infer = json.load(open(f'outputs/{data}.infer/infer_0.result.json','r'))
  for e in infer:
    del e["scores"]
    e["gt"] = e["tgt"]
    e["PLATO"] = e["preds"] 
    del e["preds"]
    del e["tgt"]
  json.dump(infer,open(f'outputs/{data}.infer/postprocessed_result.json','w'))
