import torch
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib

class LeakSenseEnsemble:
    def __init__(self, leak_net, loc_net, gnn_net, rf_model, gb_model):
        self.leak_net = leak_net
        self.loc_net = loc_net
        self.gnn_net = gnn_net
        self.rf = rf_model
        self.gb = gb_model
        
    def predict(self, features_tensor, edge_index=None, features_numpy=None):
        self.leak_net.eval()
        self.loc_net.eval()
        self.gnn_net.eval()
        
        with torch.no_grad():
            p_nn_binary = self.leak_net(features_tensor).item()
            p_loc_probs = self.loc_net(features_tensor).numpy()[0]
            
            # For ensemble we need a common ground. 
            # We'll use the binary detection from LeakSenseNet and combine with others
            # LocalizationNet also gives a 'No Leak' class (0)
            p_nn_loc = 1.0 - p_loc_probs[0] # Probability of leak from LocalizationNet
            
            p_rf = self.rf.predict_proba(features_numpy)[0][1] if self.rf else 0.5
            p_gb = self.gb.predict_proba(features_numpy)[0][1] if self.gb else 0.5
            
            # Weighted ensemble for binary detection
            final_prob = (0.40 * p_nn_binary + 0.30 * p_nn_loc + 
                          0.15 * p_rf + 0.15 * p_gb)
            
            # Localization
            suspected_zone_idx = np.argmax(p_loc_probs[1:]) + 1
            
        return {
            'leak_detected': final_prob > 0.5,
            'confidence': float(final_prob),
            'zone_probabilities': p_loc_probs.tolist(),
            'suspected_zone_idx': int(suspected_zone_idx)
        }

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
