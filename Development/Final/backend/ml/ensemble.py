"""
LeakSense Twin — Ensemble Predictor
Combines: LeakSenseNet + LeakLocalizationNet + RandomForest + GradientBoosting
Weighted vote: NN:0.35, Localization:0.25, RF:0.20, GB:0.20
"""

import numpy as np
import torch
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


class LeakSenseEnsemble:
    """
    Production ensemble combining neural nets with traditional ML models.
    Uses weighted voting for final leak detection and localization.
    """

    def __init__(self):
        self.leak_net = None        # LeakSenseNet (binary)
        self.loc_net = None         # LeakLocalizationNet (multi-class)
        self.rf_detector = None     # RandomForest binary detector
        self.gb_detector = None     # GradientBoosting binary detector
        self.rf_localizer = None    # RandomForest zone localizer
        self.gb_localizer = None    # GradientBoosting zone localizer
        self.scaler = None          # StandardScaler for features
        self.is_fitted = False

        # Ensemble weights
        self.weights = {
            'leak_net': 0.35,
            'loc_net': 0.25,
            'rf': 0.20,
            'gb': 0.20,
        }

    def fit_sklearn_models(self, X_train: np.ndarray, y_binary: np.ndarray,
                           y_zone: np.ndarray):
        """
        Fit the RandomForest and GradientBoosting models.
        Neural nets are trained separately via PyTorch.
        """
        # Fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        # Binary detection models
        self.rf_detector = RandomForestClassifier(
            n_estimators=100, max_depth=12, random_state=42,
            class_weight='balanced', n_jobs=-1
        )
        self.rf_detector.fit(X_scaled, y_binary)

        self.gb_detector = GradientBoostingClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42
        )
        self.gb_detector.fit(X_scaled, y_binary)

        # Zone localization models
        self.rf_localizer = RandomForestClassifier(
            n_estimators=100, max_depth=12, random_state=42,
            class_weight='balanced', n_jobs=-1
        )
        self.rf_localizer.fit(X_scaled, y_zone)

        self.gb_localizer = GradientBoostingClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42
        )
        self.gb_localizer.fit(X_scaled, y_zone)

        self.is_fitted = True

    def predict(self, features_tensor: torch.Tensor = None,
                features_numpy: np.ndarray = None) -> dict:
        """
        Ensemble prediction combining all models.

        Args:
            features_tensor: PyTorch tensor for neural nets
            features_numpy: NumPy array for sklearn models (pre-scaled)

        Returns:
            dict with confidence, zone prediction, and component scores
        """
        scores = {}

        # Neural net predictions
        if self.leak_net is not None and features_tensor is not None:
            with torch.no_grad():
                nn_conf = float(self.leak_net(features_tensor).item())
                scores['leak_net'] = nn_conf

        if self.loc_net is not None and features_tensor is not None:
            with torch.no_grad():
                loc_probs = self.loc_net(features_tensor).numpy()[0]
                scores['loc_probs'] = loc_probs

        # Sklearn predictions
        if features_numpy is not None and self.is_fitted:
            # RF detection
            rf_proba = self.rf_detector.predict_proba(features_numpy)
            scores['rf_conf'] = float(rf_proba[0][1]) if rf_proba.shape[1] > 1 else 0.5

            # GB detection
            gb_proba = self.gb_detector.predict_proba(features_numpy)
            scores['gb_conf'] = float(gb_proba[0][1]) if gb_proba.shape[1] > 1 else 0.5

            # RF/GB zone localization
            rf_zone = self.rf_localizer.predict_proba(features_numpy)[0]
            gb_zone = self.gb_localizer.predict_proba(features_numpy)[0]
            scores['rf_zone_probs'] = rf_zone
            scores['gb_zone_probs'] = gb_zone

        # Weighted ensemble for binary detection
        w = self.weights
        conf_components = []
        weight_sum = 0.0

        if 'leak_net' in scores:
            conf_components.append(w['leak_net'] * scores['leak_net'])
            weight_sum += w['leak_net']
        if 'rf_conf' in scores:
            conf_components.append(w['rf'] * scores['rf_conf'])
            weight_sum += w['rf']
        if 'gb_conf' in scores:
            conf_components.append(w['gb'] * scores['gb_conf'])
            weight_sum += w['gb']

        if weight_sum > 0:
            final_confidence = sum(conf_components) / weight_sum
        else:
            final_confidence = 0.5

        # Weighted ensemble for zone localization
        n_classes = 6
        zone_votes = np.zeros(n_classes)

        if 'loc_probs' in scores:
            zone_votes += w['loc_net'] * scores['loc_probs']
        if 'rf_zone_probs' in scores:
            zone_votes += w['rf'] * scores['rf_zone_probs']
        if 'gb_zone_probs' in scores:
            zone_votes += w['gb'] * scores['gb_zone_probs']

        zone_votes_sum = zone_votes.sum()
        if zone_votes_sum > 0:
            zone_votes /= zone_votes_sum

        suspected_zone_idx = int(np.argmax(zone_votes))

        return {
            'confidence': float(final_confidence),
            'suspected_zone_idx': suspected_zone_idx,
            'zone_probabilities': zone_votes.tolist(),
            'component_scores': {
                k: float(v) if isinstance(v, (int, float, np.floating)) else v
                for k, v in scores.items()
                if not isinstance(v, np.ndarray)
            },
        }

    def save(self, path: str):
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> 'LeakSenseEnsemble':
        return joblib.load(path)
