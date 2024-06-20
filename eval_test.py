from evaluate_remote_nb import calculate_accuracy_from_json
import os
#print("Current Working Directory:", os.getcwd())
p = os.path.join("logs", "lewis_n_repetitions_3.json")
calculate_accuracy_from_json(p)