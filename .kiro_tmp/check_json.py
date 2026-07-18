import json
d = json.load(open('proper_evaluation_results.json'))
print(f"Keys: {list(d.keys())}")
print(f"Methods: {list(d['methods'].keys())}")
for m, data in d['methods'].items():
    ro = data['relation_only']['aggregate']
    e2e = data['end_to_end']['aggregate']
    print(f"\n{m}:")
    print(f"  Rel-Only: Micro F1={ro['micro_f1']}, CI={ro['ci_95']}")
    print(f"  End2End:  Micro F1={e2e['micro_f1']}, CI={e2e['ci_95']}")
