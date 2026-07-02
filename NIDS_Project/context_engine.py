import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime

class ContextEngine:
    """
    Context-Aware Component for NIDS.
    Tracks recent predictions in a sliding time window and assigns
    severity levels based on attack frequency and confidence scores.
    """

    def __init__(self, window_size=50):
        """
        window_size: number of recent packets to consider for context
        """
        self.window_size    = window_size
        self.history        = deque(maxlen=window_size)
        self.alert_log      = []
        self.packet_count   = 0
        self.attack_count   = 0

    # ── Severity Logic ────────────────────────────────────────────────────────
    def get_severity(self, attack_prob, attack_rate_in_window):
        """
        Combines model confidence + recent attack rate to assign severity.
        
        attack_prob          : model's confidence this packet is an attack (0-1)
        attack_rate_in_window: % of recent packets that were attacks (0-1)
        """
        # Confidence score (0-100)
        confidence = attack_prob * 100

        # Context score — how many recent packets were also attacks
        context_score = attack_rate_in_window * 100

        # Combined threat score
        threat_score = (confidence * 0.6) + (context_score * 0.4)

        if threat_score >= 80:
            return "CRITICAL", "🔴"
        elif threat_score >= 60:
            return "HIGH",     "🟠"
        elif threat_score >= 40:
            return "MEDIUM",   "🟡"
        else:
            return "LOW",      "🟢"

    # ── Process One Packet ────────────────────────────────────────────────────
    def process(self, prediction, attack_prob, shap_top_features, packet_id=None):
        """
        Process a single packet prediction and return full context-aware alert.

        prediction       : 0 (Normal) or 1 (Attack)
        attack_prob      : model probability of being attack (0.0 - 1.0)
        shap_top_features: list of (feature_name, shap_value) tuples
        packet_id        : optional packet identifier
        """
        self.packet_count += 1
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Add to sliding window history
        self.history.append(prediction)

        # Calculate attack rate in current window
        if len(self.history) > 0:
            attack_rate = sum(self.history) / len(self.history)
        else:
            attack_rate = 0.0

        # Build result dict
        result = {
            'packet_id'      : packet_id or self.packet_count,
            'timestamp'      : timestamp,
            'prediction'     : prediction,
            'label'          : 'Attack' if prediction == 1 else 'Normal',
            'confidence'     : round(attack_prob * 100, 1),
            'attack_rate'    : round(attack_rate * 100, 1),
            'top_features'   : shap_top_features,
            'severity'       : None,
            'severity_icon'  : None,
            'is_alert'       : False
        }

        if prediction == 1:
            self.attack_count += 1
            severity, icon = self.get_severity(attack_prob, attack_rate)
            result['severity']      = severity
            result['severity_icon'] = icon
            result['is_alert']      = True

            # Log the alert
            self.alert_log.append(result)

        return result

    # ── Window Stats ──────────────────────────────────────────────────────────
    def get_window_stats(self):
        """Returns stats about the current sliding window."""
        if len(self.history) == 0:
            return {'attack_rate': 0, 'normal_rate': 100, 'window_size': 0}

        attack_rate = sum(self.history) / len(self.history) * 100
        return {
            'attack_rate'  : round(attack_rate, 1),
            'normal_rate'  : round(100 - attack_rate, 1),
            'window_size'  : len(self.history),
            'total_packets': self.packet_count,
            'total_attacks': self.attack_count
        }

    # ── Threat Level for Dashboard Header ────────────────────────────────────
    def get_overall_threat_level(self):
        """Returns overall system threat level based on recent window."""
        stats = self.get_window_stats()
        rate  = stats['attack_rate']

        if rate >= 60:
            return "CRITICAL", "🔴", "#FF3333"
        elif rate >= 40:
            return "HIGH",     "🟠", "#FF8C00"
        elif rate >= 20:
            return "MEDIUM",   "🟡", "#FFD700"
        else:
            return "LOW",      "🟢", "#00CC44"

    # ── Recent Alerts ─────────────────────────────────────────────────────────
    def get_recent_alerts(self, n=10):
        """Returns the last n alerts."""
        return self.alert_log[-n:]

    # ── Reset ─────────────────────────────────────────────────────────────────
    def reset(self):
        self.history.clear()
        self.alert_log.clear()
        self.packet_count = 0
        self.attack_count = 0


# ── Quick Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Context Engine...")
    engine = ContextEngine(window_size=50)

    # Simulate some packets
    test_cases = [
        (0, 0.05, [('dns.qry.name.len', 0.01)]),   # Normal
        (1, 0.95, [('dns.qry.name.len', 7.2)]),     # Critical attack
        (1, 0.88, [('mqtt.msg', 3.1)]),              # High attack
        (0, 0.12, [('tcp.len', 0.05)]),              # Normal
        (1, 0.72, [('mqtt.conack.flags', 0.9)]),     # Medium attack
    ]

    for pred, prob, feats in test_cases:
        result = engine.process(pred, prob, feats)
        if result['is_alert']:
            print(f"  {result['severity_icon']} [{result['timestamp']}] "
                  f"ATTACK | Confidence: {result['confidence']}% | "
                  f"Severity: {result['severity']} | "
                  f"Window Attack Rate: {result['attack_rate']}%")
        else:
            print(f"  ✅ [{result['timestamp']}] NORMAL | "
                  f"Confidence: {result['confidence']}%")

    print("\nWindow Stats:")
    stats = engine.get_window_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    level, icon, color = engine.get_overall_threat_level()
    print(f"\nOverall Threat Level: {icon} {level}")
    print("✅ Context Engine working correctly!")