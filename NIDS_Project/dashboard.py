import customtkinter as ctk
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import pickle
import threading
import time
import os
from collections import deque
from sklearn.preprocessing import LabelEncoder
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from context_engine import ContextEngine

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = r'D:\NIDS_Project\data'
MODEL_DIR = r'D:\NIDS_Project\models'

# ── Theme ─────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

COLORS = {
    'bg'       : '#0a0e1a',
    'panel'    : '#111827',
    'card'     : '#1a2235',
    'border'   : '#1e3a5f',
    'text'     : '#e2e8f0',
    'subtext'  : '#94a3b8',
    'green'    : '#00ff88',
    'red'      : '#ff4444',
    'orange'   : '#ff8c00',
    'yellow'   : '#ffd700',
    'blue'     : '#3b82f6',
    'critical' : '#ff3333',
    'high'     : '#ff8c00',
    'medium'   : '#ffd700',
    'low'      : '#00cc44',
}

SEVERITY_COLORS = {
    'CRITICAL' : COLORS['critical'],
    'HIGH'     : COLORS['high'],
    'MEDIUM'   : COLORS['medium'],
    'LOW'      : COLORS['low'],
    'Normal'   : COLORS['green'],
}


class NIDSDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("🛡️ NIDS — Network Intrusion Detection System")
        self.geometry("1400x900")
        self.configure(fg_color=COLORS['bg'])
        self.resizable(True, True)

        # ── State ─────────────────────────────────────────────────────────────
        self.running       = False
        self.speed         = 0.3
        self.packet_count  = 0
        self.attack_count  = 0
        self.normal_count  = 0
        self.rate_history  = deque(maxlen=30)
        self.time_labels   = deque(maxlen=30)
        self.tick          = 0
        self.sample_idx    = 0
        self.shap_precomputed = {}

        # ── Load model and data ───────────────────────────────────────────────
        self._load_model_and_data()

        # ── Context engine ────────────────────────────────────────────────────
        self.context = ContextEngine(window_size=50)

        # ── Build UI (controls first so they are never hidden) ────────────────
        self._build_controls()
        self._build_header()
        self._build_stat_cards()
        self._build_body()

    # ══════════════════════════════════════════════════════════════════════════
    # DATA LOADING
    # ══════════════════════════════════════════════════════════════════════════
    def _load_model_and_data(self):
        print("Loading model...")
        self.model = xgb.XGBClassifier()
        self.model.load_model(os.path.join(MODEL_DIR, 'xgboost_ids_model.json'))

        print("Loading SHAP explainer...")
        with open(os.path.join(MODEL_DIR, 'shap_explainer.pkl'), 'rb') as f:
            self.explainer = pickle.load(f)

        print("Loading data...")
        X_test  = pd.read_parquet(os.path.join(DATA_DIR, 'X_test.parquet'))
        X_train = pd.read_parquet(os.path.join(DATA_DIR, 'X_train.parquet'))
        X_val   = pd.read_parquet(os.path.join(DATA_DIR, 'X_val.parquet'))

        print("Encoding columns...")
        str_cols = X_train.select_dtypes(include=['object', 'str']).columns.tolist()
        for col in str_cols:
            n_unique = X_train[col].nunique()
            if n_unique <= 50:
                le = LabelEncoder()
                all_vals = pd.concat(
                    [X_train[col], X_val[col], X_test[col]]
                ).astype(str).unique()
                le.fit(all_vals)
                X_test[col] = le.transform(X_test[col].astype(str))
            else:
                X_test[col] = X_test[col].astype(str).apply(
                    lambda x: hash(x) % 100000
                )

        self.X_test  = X_test.reset_index(drop=True)
        self.columns = self.X_test.columns.tolist()
        print("✅ Everything loaded!")

        # Pre-compute SHAP for first 500 rows
        print("Pre-computing SHAP values (first 500 packets)...")
        sample = self.X_test.iloc[:500]
        shap_vals = self.explainer(sample)
        for i in range(len(sample)):
            feat_shap = list(zip(self.columns, np.abs(shap_vals.values[i])))
            feat_shap.sort(key=lambda x: x[1], reverse=True)
            self.shap_precomputed[i] = feat_shap
        print("✅ SHAP pre-computed!")

    # ══════════════════════════════════════════════════════════════════════════
    # UI BUILDERS
    # ══════════════════════════════════════════════════════════════════════════
    def _build_controls(self):
        """Built FIRST so it is always visible at the bottom."""
        ctrl = ctk.CTkFrame(self, fg_color=COLORS['panel'],
                            corner_radius=0, height=60)
        ctrl.pack(side='bottom', fill='x')
        ctrl.pack_propagate(False)

        self.btn_start = ctk.CTkButton(
            ctrl, text="▶  Start Simulation",
            width=170, height=38,
            fg_color=COLORS['green'], text_color='black',
            font=ctk.CTkFont(size=13, weight='bold'),
            command=self.start_simulation)
        self.btn_start.pack(side='left', padx=15, pady=11)

        self.btn_stop = ctk.CTkButton(
            ctrl, text="⏹  Stop",
            width=110, height=38,
            fg_color=COLORS['red'],
            font=ctk.CTkFont(size=13, weight='bold'),
            state='disabled',
            command=self.stop_simulation)
        self.btn_stop.pack(side='left', padx=5, pady=11)

        ctk.CTkButton(
            ctrl, text="🗑  Clear",
            width=110, height=38,
            fg_color=COLORS['card'],
            font=ctk.CTkFont(size=13),
            command=self.clear_all).pack(side='left', padx=5, pady=11)

        ctk.CTkLabel(ctrl, text="Speed:",
                     text_color=COLORS['subtext'],
                     font=ctk.CTkFont(size=12)).pack(side='left', padx=(25, 5))

        self.speed_slider = ctk.CTkSlider(
            ctrl, from_=0.05, to=1.5,
            width=160, number_of_steps=29,
            command=self._update_speed)
        self.speed_slider.set(0.3)
        self.speed_slider.pack(side='left', pady=11)

        self.speed_label = ctk.CTkLabel(
            ctrl, text="0.30s/pkt",
            text_color=COLORS['subtext'],
            font=ctk.CTkFont(size=12))
        self.speed_label.pack(side='left', padx=8)

        self.status_label = ctk.CTkLabel(
            ctrl, text="● Idle",
            text_color=COLORS['subtext'],
            font=ctk.CTkFont(size=13, weight='bold'))
        self.status_label.pack(side='right', padx=25)

    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color=COLORS['panel'],
                              corner_radius=0, height=58)
        header.pack(side='top', fill='x')
        header.pack_propagate(False)

        ctk.CTkLabel(
            header,
            text="🛡️  NIDS — Real-Time Intrusion Detection Dashboard",
            font=ctk.CTkFont(size=20, weight='bold'),
            text_color=COLORS['blue']
        ).pack(side='left', padx=20, pady=14)

        self.threat_label = ctk.CTkLabel(
            header, text="● THREAT: LOW",
            font=ctk.CTkFont(size=14, weight='bold'),
            text_color=COLORS['low'])
        self.threat_label.pack(side='right', padx=25)

    def _build_stat_cards(self):
        row = ctk.CTkFrame(self, fg_color=COLORS['bg'])
        row.pack(side='top', fill='x', padx=15, pady=(10, 5))

        self.card_total   = self._stat_card(row, "Total Packets", "0",    COLORS['blue'])
        self.card_normal  = self._stat_card(row, "Normal",        "0",    COLORS['green'])
        self.card_attacks = self._stat_card(row, "Attacks",       "0",    COLORS['red'])
        self.card_rate    = self._stat_card(row, "Attack Rate",   "0.0%", COLORS['orange'])
        self.card_window  = self._stat_card(row, "Window Rate",   "0.0%", COLORS['yellow'])

    def _stat_card(self, parent, title, value, color):
        card = ctk.CTkFrame(parent, fg_color=COLORS['card'],
                            corner_radius=10, border_width=1,
                            border_color=COLORS['border'])
        card.pack(side='left', expand=True, fill='x', padx=5)
        ctk.CTkLabel(card, text=title,
                     font=ctk.CTkFont(size=11),
                     text_color=COLORS['subtext']).pack(pady=(10, 0))
        lbl = ctk.CTkLabel(card, text=value,
                           font=ctk.CTkFont(size=26, weight='bold'),
                           text_color=color)
        lbl.pack(pady=(0, 10))
        return lbl

    def _build_body(self):
        body = ctk.CTkFrame(self, fg_color=COLORS['bg'])
        body.pack(side='top', fill='both', expand=True, padx=15, pady=(0, 8))
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        # ── Left column ───────────────────────────────────────────────────────
        left = ctk.CTkFrame(body, fg_color=COLORS['bg'])
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 8))
        left.rowconfigure(0, weight=2)
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        self._build_traffic_table(left)
        self._build_chart(left)

        # ── Right column ──────────────────────────────────────────────────────
        right = ctk.CTkFrame(body, fg_color=COLORS['bg'])
        right.grid(row=0, column=1, sticky='nsew')
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self._build_alert_panel(right)

    def _build_traffic_table(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=COLORS['panel'],
                             corner_radius=10, border_width=1,
                             border_color=COLORS['border'])
        frame.grid(row=0, column=0, sticky='nsew', pady=(0, 8))
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        ctk.CTkLabel(frame, text="📡  Live Traffic Feed",
                     font=ctk.CTkFont(size=14, weight='bold'),
                     text_color=COLORS['text']).grid(
                         row=0, column=0, sticky='w', padx=15, pady=(10, 5))

        # Header row
        hdr = ctk.CTkFrame(frame, fg_color=COLORS['border'], corner_radius=6)
        hdr.grid(row=1, column=0, sticky='ew', padx=10, pady=(0, 3))

        for col, w in [("Packet #", 80), ("Time", 75), ("Prediction", 100),
                       ("Confidence", 100), ("Severity", 95), ("Top Feature", 220)]:
            ctk.CTkLabel(hdr, text=col, width=w,
                         font=ctk.CTkFont(size=11, weight='bold'),
                         text_color=COLORS['subtext']).pack(
                             side='left', padx=5, pady=6)

        # Scrollable body
        self.table_frame = ctk.CTkScrollableFrame(
            frame, fg_color=COLORS['panel'])
        self.table_frame.grid(row=2, column=0, sticky='nsew', padx=10, pady=(0, 10))
        frame.rowconfigure(2, weight=1)

    def _build_chart(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=COLORS['panel'],
                             corner_radius=10, border_width=1,
                             border_color=COLORS['border'])
        frame.grid(row=1, column=0, sticky='nsew')

        ctk.CTkLabel(frame, text="📈  Attack Rate Over Time",
                     font=ctk.CTkFont(size=14, weight='bold'),
                     text_color=COLORS['text']).pack(
                         anchor='w', padx=15, pady=(10, 0))

        self.fig = Figure(figsize=(6, 2.2), facecolor=COLORS['panel'])
        self.ax  = self.fig.add_subplot(111)
        self.ax.set_facecolor(COLORS['card'])
        self.ax.tick_params(colors=COLORS['subtext'], labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color(COLORS['border'])
        self.fig.tight_layout(pad=1.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().pack(
            fill='both', expand=True, padx=10, pady=(0, 10))

    def _build_alert_panel(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=COLORS['panel'],
                             corner_radius=10, border_width=1,
                             border_color=COLORS['border'])
        frame.grid(row=0, column=0, sticky='nsew')
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        ctk.CTkLabel(frame, text="🚨  Alert Log  (Attacks Only)",
                     font=ctk.CTkFont(size=14, weight='bold'),
                     text_color=COLORS['red']).grid(
                         row=0, column=0, sticky='w', padx=15, pady=(10, 5))

        self.alert_frame = ctk.CTkScrollableFrame(
            frame, fg_color=COLORS['panel'])
        self.alert_frame.grid(
            row=1, column=0, sticky='nsew', padx=10, pady=(0, 10))

    # ══════════════════════════════════════════════════════════════════════════
    # SIMULATION
    # ══════════════════════════════════════════════════════════════════════════
    def start_simulation(self):
        self.running = True
        self.btn_start.configure(state='disabled')
        self.btn_stop.configure(state='normal')
        self.status_label.configure(
            text="● Running", text_color=COLORS['green'])
        threading.Thread(target=self._run_simulation, daemon=True).start()

    def stop_simulation(self):
        self.running = False
        self.btn_start.configure(state='normal')
        self.btn_stop.configure(state='disabled')
        self.status_label.configure(
            text="● Stopped", text_color=COLORS['orange'])

    def clear_all(self):
        self.stop_simulation()
        self.packet_count = 0
        self.attack_count = 0
        self.normal_count = 0
        self.tick         = 0
        self.sample_idx   = 0
        self.rate_history.clear()
        self.time_labels.clear()
        self.context.reset()

        for w in self.table_frame.winfo_children():
            w.destroy()
        for w in self.alert_frame.winfo_children():
            w.destroy()

        self.card_total.configure(text="0")
        self.card_normal.configure(text="0")
        self.card_attacks.configure(text="0")
        self.card_rate.configure(text="0.0%")
        self.card_window.configure(text="0.0%")
        self.threat_label.configure(
            text="● THREAT: LOW", text_color=COLORS['low'])
        self.status_label.configure(
            text="● Idle", text_color=COLORS['subtext'])
        self.ax.clear()
        self.canvas.draw()

    def _update_speed(self, val):
        self.speed = round(float(val), 2)
        self.speed_label.configure(text=f"{self.speed:.2f}s/pkt")

    def _run_simulation(self):
        while self.running:
            if self.sample_idx >= len(self.X_test):
                self.sample_idx = 0

            row = self.X_test.iloc[[self.sample_idx]]

            # Predict
            pred = int(self.model.predict(row)[0])
            prob = float(self.model.predict_proba(row)[0][1])

            # SHAP — use precomputed if available
            idx_key = self.sample_idx % 500
            if idx_key in self.shap_precomputed:
                top_feats = self.shap_precomputed[idx_key][:3]
            else:
                sv = self.explainer(row)
                fs = list(zip(self.columns, np.abs(sv.values[0])))
                fs.sort(key=lambda x: x[1], reverse=True)
                top_feats = fs[:3]

            top_feat = top_feats[0][0] if top_feats else "N/A"

            result = self.context.process(
                pred, prob, top_feats, packet_id=self.sample_idx)

            self.sample_idx   += 1
            self.packet_count += 1
            if pred == 1:
                self.attack_count += 1
            else:
                self.normal_count += 1

            self.after(0, self._update_ui, result, top_feat)
            time.sleep(self.speed)

    # ══════════════════════════════════════════════════════════════════════════
    # UI UPDATES
    # ══════════════════════════════════════════════════════════════════════════
    def _update_ui(self, result, top_feat):
        self._update_cards()
        self._add_table_row(result, top_feat)
        if result['is_alert']:
            self._add_alert_card(result)
        self._update_chart()
        self._update_threat_badge()

    def _update_cards(self):
        rate  = (self.attack_count / self.packet_count * 100
                 if self.packet_count > 0 else 0)
        stats = self.context.get_window_stats()
        self.card_total.configure(text=f"{self.packet_count:,}")
        self.card_normal.configure(text=f"{self.normal_count:,}")
        self.card_attacks.configure(text=f"{self.attack_count:,}")
        self.card_rate.configure(text=f"{rate:.1f}%")
        self.card_window.configure(text=f"{stats['attack_rate']}%")

    def _add_table_row(self, result, top_feat):
        is_attack = result['prediction'] == 1
        txt_color = COLORS['red'] if is_attack else COLORS['green']
        sev_color = SEVERITY_COLORS.get(
            result['severity'] or 'Normal', COLORS['green'])

        row = ctk.CTkFrame(self.table_frame, fg_color=COLORS['card'],
                           corner_radius=4)
        row.pack(fill='x', pady=1)

        for text, width, color in [
            (str(result['packet_id']),   80,  COLORS['subtext']),
            (result['timestamp'],         75,  COLORS['subtext']),
            (result['label'],            100,  txt_color),
            (f"{result['confidence']}%", 100,  txt_color),
            (result['severity'] or '—',   95,  sev_color),
            (top_feat[:30],              220,  COLORS['text']),
        ]:
            ctk.CTkLabel(row, text=text, width=width,
                         font=ctk.CTkFont(size=11),
                         text_color=color).pack(side='left', padx=5, pady=4)

        self.after(50, lambda: self.table_frame._parent_canvas.yview_moveto(1.0))

    def _add_alert_card(self, result):
        sev   = result['severity']
        color = SEVERITY_COLORS.get(sev, COLORS['red'])
        icon  = result['severity_icon']

        card = ctk.CTkFrame(self.alert_frame, fg_color=COLORS['card'],
                            corner_radius=8, border_width=1,
                            border_color=color)
        card.pack(fill='x', pady=3)

        # Header bar
        hdr = ctk.CTkFrame(card, fg_color=color, corner_radius=6)
        hdr.pack(fill='x', padx=6, pady=(6, 3))
        ctk.CTkLabel(hdr,
                     text=f"{icon} {sev} ALERT  —  Pkt #{result['packet_id']}",
                     font=ctk.CTkFont(size=12, weight='bold'),
                     text_color='black').pack(side='left', padx=8, pady=4)
        ctk.CTkLabel(hdr, text=result['timestamp'],
                     font=ctk.CTkFont(size=11),
                     text_color='black').pack(side='right', padx=8)

        # Stats line
        ctk.CTkLabel(card,
                     text=f"Confidence: {result['confidence']}%   |   "
                          f"Window Attack Rate: {result['attack_rate']}%",
                     font=ctk.CTkFont(size=11),
                     text_color=COLORS['subtext']).pack(
                         anchor='w', padx=10, pady=(4, 2))

        # SHAP reasons
        ctk.CTkLabel(card, text="🔍 Why flagged:",
                     font=ctk.CTkFont(size=11, weight='bold'),
                     text_color=COLORS['text']).pack(anchor='w', padx=10)

        for feat, val in result['top_features'][:3]:
            ctk.CTkLabel(card,
                         text=f"   • {feat[:35]:<35}  SHAP: {val:.4f}",
                         font=ctk.CTkFont(size=10, family='Courier'),
                         text_color=COLORS['yellow']).pack(anchor='w', padx=10)

        ctk.CTkLabel(card, text="").pack(pady=2)
        self.after(50, lambda: self.alert_frame._parent_canvas.yview_moveto(1.0))

    def _update_chart(self):
        self.tick += 1
        rate = (self.attack_count / self.packet_count * 100
                if self.packet_count > 0 else 0)
        self.rate_history.append(rate)
        self.time_labels.append(self.tick)

        self.ax.clear()
        self.ax.set_facecolor(COLORS['card'])
        self.ax.tick_params(colors=COLORS['subtext'], labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color(COLORS['border'])

        x = list(self.time_labels)
        y = list(self.rate_history)

        self.ax.plot(x, y, color=COLORS['red'], linewidth=1.5)
        self.ax.fill_between(x, y, alpha=0.2, color=COLORS['red'])
        self.ax.set_ylim(0, 100)
        self.ax.set_ylabel('Attack %', color=COLORS['subtext'], fontsize=8)
        self.ax.axhline(y=50, color=COLORS['orange'],
                        linestyle='--', linewidth=0.8, alpha=0.5)
        self.fig.tight_layout(pad=1.2)
        self.canvas.draw()

    def _update_threat_badge(self):
        level, icon, color = self.context.get_overall_threat_level()
        self.threat_label.configure(
            text=f"● THREAT: {level}", text_color=color)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = NIDSDashboard()
    app.mainloop()