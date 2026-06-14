import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import importlib
import numpy as np
import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import os

m1 = importlib.import_module("1")
m2 = importlib.import_module("2")
m3 = importlib.import_module("3")

class FintechDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("FinTech Risk Engine - Professional Edition")
        self.root.state('zoomed') 
        
        self.bg_lavender = "#E6E6FA"
        self.btn_pink = "#FFB6C1"
        self.btn_lavender = "#DCD0FF"
        self.text_dark = "#4B0082"
        self.plot_pink = "#FFC0CB"
        self.plot_lavender = "#B57EDC"

        self.root.configure(bg=self.bg_lavender)
        self.raw_df = None
        self.feature_names = ['Annual_Income', 'Credit_Score', 'Debt_Ratio']

        header = tk.Frame(self.root, bg=self.btn_lavender, height=80)
        header.pack(fill="x")
        tk.Label(header, text="FINTECH PORTFOLIO OPTIMIZATION DASHBOARD", 
                 font=("Helvetica", 24, "bold"), fg=self.text_dark, bg=self.btn_lavender).pack(pady=15)

        config_frame = tk.LabelFrame(self.root, text="System Configuration", font=("Arial", 12, "bold"), 
                                    fg=self.text_dark, bg=self.bg_lavender, padx=20, pady=10)
        config_frame.pack(pady=10, fill="x", padx=50)

        self.slider_label = tk.Label(config_frame, text="Records to Process:", fg=self.text_dark, bg=self.bg_lavender)
        self.slider_label.grid(row=0, column=0, sticky="w")
        
        self.record_slider = tk.Scale(config_frame, from_=5, to=100, orient=tk.HORIZONTAL, 
                                     bg=self.bg_lavender, fg=self.text_dark, length=300, 
                                     highlightthickness=0, troughcolor=self.btn_pink)
        self.record_slider.set(10)
        self.record_slider.grid(row=0, column=1, padx=20)

        self.upload_btn = tk.Button(config_frame, text="📁 UPLOAD CUSTOM CSV", command=self.load_data, 
                                   bg="white", fg=self.text_dark, font=("Arial", 10, "bold"), width=20)
        self.upload_btn.grid(row=0, column=2, padx=10)

        self.run_btn = tk.Button(config_frame, text="🚀 RUN ANALYSIS", command=self.run_pipeline, 
                                bg=self.btn_pink, fg="white", font=("Arial", 10, "bold"), width=15)
        self.run_btn.grid(row=0, column=3, padx=10)

        self.exit_btn = tk.Button(config_frame, text="❌ EXIT", command=self.root.quit, 
                                 bg=self.btn_lavender, fg=self.text_dark, font=("Arial", 10, "bold"), width=15)
        self.exit_btn.grid(row=0, column=4, padx=10)

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(pady=10, fill="x", padx=50)

        self.log_text = tk.Text(self.notebook, height=10, font=("Consolas", 11), bg="white", fg=self.text_dark)
        self.log_text.config(state='disabled')
        self.notebook.add(self.log_text, text=" System Status Log ")

        self.matrix_text = tk.Text(self.notebook, height=12, font=("Consolas", 11), bg="white", fg=self.text_dark)
        self.matrix_text.config(state='disabled')
        self.notebook.add(self.matrix_text, text=" Matrix & Basis View (Decimals) ")

        self.plot_container = tk.Frame(self.root, bg=self.bg_lavender)
        self.plot_container.pack(fill="both", expand=True, pady=10)

        # AUTO-LOAD DEFAULT CSV
        self.auto_load_default()

    def log_message(self, msg, target="log"):
        text_widget = self.log_text if target == "log" else self.matrix_text
        text_widget.config(state='normal')
        text_widget.insert(tk.END, f"{msg}\n")
        text_widget.config(state='disabled')
        text_widget.see(tk.END)

    def auto_load_default(self):
        default_file = "data.csv"
        if os.path.exists(default_file):
            self.raw_df = pd.read_csv(default_file)
            num_rows = len(self.raw_df)
            self.record_slider.config(to=num_rows)
            self.record_slider.set(num_rows)
            self.log_message(f"✅ System Initialized: Default dataset '{default_file}' auto-loaded.")
        else:
            self.log_message("⚠️ Warning: data.csv not found. Please upload a file.")

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.raw_df = pd.read_csv(file_path)
            self.feature_names = self.raw_df.columns[1:4].tolist()
            num_rows = len(self.raw_df)
            self.record_slider.config(to=num_rows)
            self.log_message(f"✅ Success: Loaded custom file '{os.path.basename(file_path)}'.")

    def format_matrix(self, mat):
        return np.array2string(mat, formatter={'float_kind':lambda x: "%.2f" % x}, separator=', ')

    # In main2.py
    def run_pipeline(self):
        for widget in self.plot_container.winfo_children(): widget.destroy()
        self.matrix_text.config(state='normal'); self.matrix_text.delete('1.0', tk.END); self.matrix_text.config(state='disabled')
        
        n = self.record_slider.get()
        try:
            if self.raw_df is not None:
                subset = self.raw_df.head(n)
                
                # Prepare the data from CSV
                annual = subset['Annual_Income'].values
                credit = subset['Credit_Score'].values
                debt = subset['Debt_Ratio'].values
                monthly = annual / 12  # Adding the redundancy for the test
                
                A_full = np.column_stack((annual, credit, debt, monthly))

                # CALLING 1.PY: All terminal printing happens inside this call now
                _, pivots = m1.data_architect_analysis(matrix_input=A_full)

                # Slice A to only include independent (Basis) columns
                A = A_full[:, pivots]
                b = subset['Risk_Score'].values

                # Labels for UI
                all_possible_features = ['Annual_Income', 'Credit_Score', 'Debt_Ratio', 'Monthly_Income']
                self.feature_names = [all_possible_features[i] for i in pivots]
            else:
                messagebox.showerror("Error", "No data source found.")
                return

            # Rest of the Pipeline
            m2_res = m2.run_part2(A, b=b, pivots=range(A.shape[1]))
            self.display_plots(A, m2_res, b)
            
            # UI Status Updates
            self.log_message("=== LINEAR ALGEBRA DATA VIEW (Decimals) ===", "matrix")
            self.log_message(f"\n1. INPUT MATRIX A ({n} records):", "matrix")
            self.log_message(self.format_matrix(A[:5, :]), "matrix")
            
            # ADDED: This displays the Orthogonal vectors on the screen
            self.log_message(f"\n2. ORTHONORMAL BASIS VECTORS (Q):", "matrix")
            self.log_message(self.format_matrix(m2_res["Q"][:5, :]), "matrix")
            
            self.log_message(f"✅ Success: Pipeline executed for {n} records.")
            self.notebook.select(1)

        except Exception as e:
            self.log_message(f"❌ PIPELINE ERROR: {str(e)}")

    def display_plots(self, A, m2_res, b):
        fig = plt.figure(figsize=(18, 8), facecolor=self.bg_lavender)
        plt.rcParams.update({'text.color': self.text_dark, 'axes.labelcolor': self.text_dark, 
                             'xtick.color': self.text_dark, 'ytick.color': self.text_dark})
        
        ax1 = fig.add_subplot(1, 3, 1); ax1.set_facecolor('white')
        show_n = min(20, len(b))
        ax1.scatter(range(show_n), b[:show_n], color=self.plot_pink, label='Actual Risk', s=120, zorder=3)
        ax1.plot(range(show_n), (A @ m2_res["least_squares_solution"])[:show_n], 
                color=self.plot_lavender, marker='o', linewidth=3, label='Predicted Risk', zorder=2)
        ax1.set_title("1. LEAST SQUARES PREDICTION\n(Model Accuracy Check)", fontsize=12, fontweight='bold', pad=20)
        ax1.set_xlabel("Borrower Row ID"); ax1.set_ylabel("Risk Score")
        ax1.legend()

        ax2 = fig.add_subplot(1, 3, 2)
        corr_matrix = np.corrcoef(m2_res["Q"], rowvar=False)
        current_labels = self.feature_names[:m2_res["Q"].shape[1]]
        sns.heatmap(corr_matrix, annot=True, cmap=sns.light_palette(self.plot_lavender, as_cmap=True), 
                    ax=ax2, cbar=False, xticklabels=current_labels, yticklabels=current_labels)
        ax2.set_title("2. FACTOR INDEPENDENCE\n(Gram-Schmidt Orthogonality)", fontsize=12, fontweight='bold', pad=20)

        ax3 = fig.add_subplot(1, 3, 3); ax3.set_facecolor('white')
        eig_vals, eig_vecs = np.linalg.eig(np.cov(m2_res["Q"], rowvar=False))
        projected = m2_res["Q"] @ eig_vecs[:, :2]
        scatter = ax3.scatter(projected[:, 0], projected[:, 1], c=projected[:, 0], cmap='RdPu', s=80, edgecolor=self.plot_lavender)
        ax3.set_title("3. BORROWER RISK CLUSTERS\n(Eigen-Analysis Results)", fontsize=12, fontweight='bold', pad=20)
        ax3.set_xlabel("Primary Risk Trend"); ax3.set_ylabel("Secondary Risk Trend")
        plt.colorbar(scatter, ax=ax3, label='Risk Intensity')

        plt.subplots_adjust(top=0.85, bottom=0.15, wspace=0.35)
        canvas = FigureCanvasTkAgg(fig, master=self.plot_container); canvas.draw(); canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

if __name__ == "__main__":
    root = tk.Tk(); app = FintechDashboard(root); root.mainloop()