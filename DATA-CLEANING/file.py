# Install required packages:
# pip install pandas numpy tkinter openpyxl xlrd matplotlib seaborn pillow

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

class AdvancedDataCleaningTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Data Cleaning Tool")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Data storage
        self.df = None
        self.original_df = None
        self.cleaning_history = []
        
        # Setup GUI
        self.setup_menu()
        self.setup_gui()
        
    def setup_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Data", command=self.load_data)
        file_menu.add_command(label="Save Cleaned Data", command=self.save_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Data Profile Report", command=self.generate_profile_report)
        tools_menu.add_command(label="Visualize Data", command=self.visualize_data)
        tools_menu.add_command(label="Export Cleaning Log", command=self.export_cleaning_log)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def setup_gui(self):
        """Setup main GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Left Panel - Controls
        self.setup_control_panel(main_frame)
        
        # Right Panel - Data View and Stats
        self.setup_data_panel(main_frame)
        
        # Bottom Panel - Log
        self.setup_log_panel(main_frame)
    
    def setup_control_panel(self, parent):
        """Setup control panel with cleaning options"""
        control_frame = ttk.LabelFrame(parent, text="Data Cleaning Controls", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Load Data Button
        ttk.Button(control_frame, text="📁 Load Data", command=self.load_data, 
                  width=25).pack(pady=5)
        
        ttk.Separator(control_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # Missing Values Section
        missing_frame = ttk.LabelFrame(control_frame, text="Missing Values", padding="5")
        missing_frame.pack(fill='x', pady=5)
        
        ttk.Button(missing_frame, text="Remove Rows with Missing", 
                  command=self.remove_missing_rows, width=25).pack(pady=2)
        ttk.Button(missing_frame, text="Fill Missing (Mean)", 
                  command=lambda: self.fill_missing('mean'), width=25).pack(pady=2)
        ttk.Button(missing_frame, text="Fill Missing (Median)", 
                  command=lambda: self.fill_missing('median'), width=25).pack(pady=2)
        ttk.Button(missing_frame, text="Fill Missing (Mode)", 
                  command=lambda: self.fill_missing('mode'), width=25).pack(pady=2)
        ttk.Button(missing_frame, text="Forward Fill", 
                  command=lambda: self.fill_missing('ffill'), width=25).pack(pady=2)
        
        # Duplicates Section
        dup_frame = ttk.LabelFrame(control_frame, text="Duplicates", padding="5")
        dup_frame.pack(fill='x', pady=5)
        
        ttk.Button(dup_frame, text="Remove Duplicates", 
                  command=self.remove_duplicates, width=25).pack(pady=2)
        ttk.Button(dup_frame, text="Mark Duplicates", 
                  command=self.mark_duplicates, width=25).pack(pady=2)
        
        # Outliers Section
        outlier_frame = ttk.LabelFrame(control_frame, text="Outliers", padding="5")
        outlier_frame.pack(fill='x', pady=5)
        
        ttk.Button(outlier_frame, text="Remove Outliers (IQR)", 
                  command=self.remove_outliers_iqr, width=25).pack(pady=2)
        ttk.Button(outlier_frame, text="Remove Outliers (Z-score)", 
                  command=self.remove_outliers_zscore, width=25).pack(pady=2)
        
        # Data Type Section
        dtype_frame = ttk.LabelFrame(control_frame, text="Data Types", padding="5")
        dtype_frame.pack(fill='x', pady=5)
        
        ttk.Button(dtype_frame, text="Auto-Detect Types", 
                  command=self.auto_detect_types, width=25).pack(pady=2)
        ttk.Button(dtype_frame, text="Convert to Numeric", 
                  command=self.convert_to_numeric, width=25).pack(pady=2)
        ttk.Button(dtype_frame, text="Convert to DateTime", 
                  command=self.convert_to_datetime, width=25).pack(pady=2)
        
        # Text Cleaning Section
        text_frame = ttk.LabelFrame(control_frame, text="Text Cleaning", padding="5")
        text_frame.pack(fill='x', pady=5)
        
        ttk.Button(text_frame, text="Trim Whitespace", 
                  command=self.trim_whitespace, width=25).pack(pady=2)
        ttk.Button(text_frame, text="Remove Special Characters", 
                  command=self.remove_special_chars, width=25).pack(pady=2)
        ttk.Button(text_frame, text="Standardize Case", 
                  command=self.standardize_case, width=25).pack(pady=2)
        
        # Utility Section
        util_frame = ttk.LabelFrame(control_frame, text="Utilities", padding="5")
        util_frame.pack(fill='x', pady=5)
        
        ttk.Button(util_frame, text="Reset to Original", 
                  command=self.reset_data, width=25).pack(pady=2)
        ttk.Button(util_frame, text="Undo Last Action", 
                  command=self.undo_last, width=25).pack(pady=2)
        ttk.Button(util_frame, text="💾 Save Cleaned Data", 
                  command=self.save_data, width=25).pack(pady=2)
    
    def setup_data_panel(self, parent):
        """Setup data view panel"""
        data_frame = ttk.Frame(parent)
        data_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        data_frame.columnconfigure(0, weight=1)
        data_frame.rowconfigure(0, weight=1)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(data_frame)
        self.notebook.pack(fill='both', expand=True)
        
        # Data View Tab
        self.data_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_tab, text="📊 Data View")
        
        # Treeview for data display
        tree_frame = ttk.Frame(self.data_tab)
        tree_frame.pack(fill='both', expand=True)
        
        # Scrollbars
        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal")
        
        self.tree = ttk.Treeview(tree_frame, yscrollcommand=vsb.set, 
                                 xscrollcommand=hsb.set)
        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)
        
        self.tree.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')
        hsb.pack(side='bottom', fill='x')
        
        # Statistics Tab
        self.stats_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_tab, text="📈 Statistics")
        
        self.stats_text = scrolledtext.ScrolledText(self.stats_tab, width=80, height=30,
                                                     font=('Courier', 10))
        self.stats_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Info Tab
        self.info_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.info_tab, text="ℹ️ Info")
        
        self.info_text = scrolledtext.ScrolledText(self.info_tab, width=80, height=30,
                                                    font=('Courier', 10))
        self.info_text.pack(fill='both', expand=True, padx=5, pady=5)
    
    def setup_log_panel(self, parent):
        """Setup activity log panel"""
        log_frame = ttk.LabelFrame(parent, text="Activity Log", padding="5")
        log_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, width=120,
                                                   font=('Courier', 9))
        self.log_text.pack(fill='both', expand=True)
        
        self.log_message("Welcome to Advanced Data Cleaning Tool! Load a dataset to begin.")
    
    # ============================================
    # Data Loading and Saving
    # ============================================
    
    def load_data(self):
        """Load data from file"""
        file_path = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # Load based on file extension
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == '.csv':
                self.df = pd.read_csv(file_path)
            elif ext in ['.xlsx', '.xls']:
                self.df = pd.read_excel(file_path)
            elif ext == '.json':
                self.df = pd.read_json(file_path)
            else:
                messagebox.showerror("Error", "Unsupported file format")
                return
            
            self.original_df = self.df.copy()
            self.cleaning_history = []
            
            self.log_message(f"✓ Loaded data from {os.path.basename(file_path)}")
            self.log_message(f"  Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
            
            self.update_display()
            self.update_statistics()
            self.update_info()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
            self.log_message(f"✗ Error loading data: {str(e)}")
    
    def save_data(self):
        """Save cleaned data"""
        if self.df is None:
            messagebox.showwarning("Warning", "No data to save!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Cleaned Data",
            defaultextension=".csv",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx"),
                ("JSON files", "*.json")
            ]
        )
        
        if not file_path:
            return
        
        try:
            ext = os.path.splitext(file_path)[1].lower()
            
            if ext == '.csv':
                self.df.to_csv(file_path, index=False)
            elif ext == '.xlsx':
                self.df.to_excel(file_path, index=False)
            elif ext == '.json':
                self.df.to_json(file_path, orient='records', indent=2)
            
            self.log_message(f"✓ Saved cleaned data to {os.path.basename(file_path)}")
            messagebox.showinfo("Success", "Data saved successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save data: {str(e)}")
            self.log_message(f"✗ Error saving data: {str(e)}")
    
    # ============================================
    # Cleaning Operations
    # ============================================
    
    def remove_missing_rows(self):
        """Remove rows with missing values"""
        if self.df is None:
            return
        
        self.save_state()
        initial_rows = len(self.df)
        self.df = self.df.dropna()
        removed = initial_rows - len(self.df)
        
        self.log_message(f"✓ Removed {removed} rows with missing values")
        self.update_display()
        self.update_statistics()
    
    def fill_missing(self, method):
        """Fill missing values"""
        if self.df is None:
            return
        
        self.save_state()
        
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            
            if method == 'mean':
                self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
            elif method == 'median':
                self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
            elif method == 'mode':
                for col in self.df.columns:
                    self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else None, inplace=True)
            elif method == 'ffill':
                self.df = self.df.fillna(method='ffill')
            
            self.log_message(f"✓ Filled missing values using {method} method")
            self.update_display()
            self.update_statistics()
            
        except Exception as e:
            self.log_message(f"✗ Error filling missing values: {str(e)}")
            self.undo_last()
    
    def remove_duplicates(self):
        """Remove duplicate rows"""
        if self.df is None:
            return
        
        self.save_state()
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = initial_rows - len(self.df)
        
        self.log_message(f"✓ Removed {removed} duplicate rows")
        self.update_display()
        self.update_statistics()
    
    def mark_duplicates(self):
        """Mark duplicate rows"""
        if self.df is None:
            return
        
        self.save_state()
        self.df['is_duplicate'] = self.df.duplicated()
        duplicates_count = self.df['is_duplicate'].sum()
        
        self.log_message(f"✓ Marked {duplicates_count} duplicate rows")
        self.update_display()
    
    def remove_outliers_iqr(self):
        """Remove outliers using IQR method"""
        if self.df is None:
            return
        
        self.save_state()
        initial_rows = len(self.df)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        
        removed = initial_rows - len(self.df)
        self.log_message(f"✓ Removed {removed} outlier rows using IQR method")
        self.update_display()
        self.update_statistics()
    
    def remove_outliers_zscore(self):
        """Remove outliers using Z-score method"""
        if self.df is None:
            return
        
        self.save_state()
        initial_rows = len(self.df)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
            self.df = self.df[z_scores < 3]
        
        removed = initial_rows - len(self.df)
        self.log_message(f"✓ Removed {removed} outlier rows using Z-score method")
        self.update_display()
        self.update_statistics()
    
    def auto_detect_types(self):
        """Automatically detect and convert data types"""
        if self.df is None:
            return
        
        self.save_state()
        self.df = self.df.infer_objects()
        
        self.log_message("✓ Auto-detected and converted data types")
        self.update_info()
    
    def convert_to_numeric(self):
        """Convert selected columns to numeric"""
        if self.df is None:
            return
        
        # Simple implementation - converts all possible columns
        self.save_state()
        
        for col in self.df.columns:
            try:
                self.df[col] = pd.to_numeric(self.df[col], errors='ignore')
            except:
                pass
        
        self.log_message("✓ Attempted numeric conversion on all columns")
        self.update_info()
    
    def convert_to_datetime(self):
        """Convert selected columns to datetime"""
        if self.df is None:
            return
        
        self.save_state()
        
        for col in self.df.columns:
            try:
                self.df[col] = pd.to_datetime(self.df[col], errors='ignore')
            except:
                pass
        
        self.log_message("✓ Attempted datetime conversion on all columns")
        self.update_info()
    
    def trim_whitespace(self):
        """Trim whitespace from text columns"""
        if self.df is None:
            return
        
        self.save_state()
        
        text_cols = self.df.select_dtypes(include=['object']).columns
        for col in text_cols:
            self.df[col] = self.df[col].str.strip()
        
        self.log_message("✓ Trimmed whitespace from text columns")
        self.update_display()
    
    def remove_special_chars(self):
        """Remove special characters from text columns"""
        if self.df is None:
            return
        
        self.save_state()
        
        text_cols = self.df.select_dtypes(include=['object']).columns
        for col in text_cols:
            self.df[col] = self.df[col].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
        
        self.log_message("✓ Removed special characters from text columns")
        self.update_display()
    
    def standardize_case(self):
        """Standardize case to title case"""
        if self.df is None:
            return
        
        self.save_state()
        
        text_cols = self.df.select_dtypes(include=['object']).columns
        for col in text_cols:
            self.df[col] = self.df[col].str.title()
        
        self.log_message("✓ Standardized text to title case")
        self.update_display()
    
    # ============================================
    # Utility Functions
    # ============================================
    
    def save_state(self):
        """Save current state for undo"""
        if self.df is not None:
            self.cleaning_history.append(self.df.copy())
    
    def undo_last(self):
        """Undo last operation"""
        if self.cleaning_history:
            self.df = self.cleaning_history.pop()
            self.log_message("↶ Undid last operation")
            self.update_display()
            self.update_statistics()
        else:
            messagebox.showinfo("Info", "Nothing to undo!")
    
    def reset_data(self):
        """Reset to original data"""
        if self.original_df is not None:
            self.df = self.original_df.copy()
            self.cleaning_history = []
            self.log_message("↺ Reset to original data")
            self.update_display()
            self.update_statistics()
            self.update_info()
        else:
            messagebox.showinfo("Info", "No original data to reset to!")
    
    # ============================================
    # Display Update Functions
    # ============================================
    
    def update_display(self):
        """Update treeview with current data"""
        if self.df is None:
            return
        
        # Clear existing
        self.tree.delete(*self.tree.get_children())
        
        # Configure columns
        self.tree['columns'] = list(self.df.columns)
        self.tree['show'] = 'headings'
        
        # Setup column headings
        for col in self.df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)
        
        # Insert data (first 1000 rows for performance)
        for idx, row in self.df.head(1000).iterrows():
            self.tree.insert('', 'end', values=list(row))
    
    def update_statistics(self):
        """Update statistics tab"""
        if self.df is None:
            return
        
        self.stats_text.delete(1.0, tk.END)
        
        stats_info = "=" * 80 + "\n"
        stats_info += "STATISTICAL SUMMARY\n"
        stats_info += "=" * 80 + "\n\n"
        stats_info += str(self.df.describe(include='all')) + "\n\n"
        
        stats_info += "=" * 80 + "\n"
        stats_info += "MISSING VALUES\n"
        stats_info += "=" * 80 + "\n"
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({'Missing': missing, 'Percentage': missing_pct})
        stats_info += str(missing_df[missing_df['Missing'] > 0]) + "\n\n"
        
        stats_info += "=" * 80 + "\n"
        stats_info += "DUPLICATES\n"
        stats_info += "=" * 80 + "\n"
        stats_info += f"Duplicate rows: {self.df.duplicated().sum()}\n"
        
        self.stats_text.insert(1.0, stats_info)
    
    def update_info(self):
        """Update info tab"""
        if self.df is None:
            return
        
        self.info_text.delete(1.0, tk.END)
        
        info = "=" * 80 + "\n"
        info += "DATASET INFORMATION\n"
        info += "=" * 80 + "\n\n"
        info += f"Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns\n\n"
        info += f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n"
        
        info += "Column Information:\n"
        info += "-" * 80 + "\n"
        for col in self.df.columns:
            info += f"{col:30} {str(self.df[col].dtype):15} Non-Null: {self.df[col].notna().sum()}\n"
        
        self.info_text.insert(1.0, info)
    
    def log_message(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
    
    # ============================================
    # Additional Features
    # ============================================
    
    def generate_profile_report(self):
        """Generate comprehensive data profile report"""
        if self.df is None:
            messagebox.showwarning("Warning", "No data loaded!")
            return
        
        report_window = tk.Toplevel(self.root)
        report_window.title("Data Profile Report")
        report_window.geometry("800x600")
        
        report_text = scrolledtext.ScrolledText(report_window, width=100, height=40,
                                                font=('Courier', 9))
        report_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        report = "=" * 80 + "\n"
        report += "COMPREHENSIVE DATA PROFILE REPORT\n"
        report += "=" * 80 + "\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += f"Dataset Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns\n"
        report += f"Total Memory: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n"
        
        report += "Data Quality Metrics:\n"
        report += "-" * 80 + "\n"
        report += f"Completeness: {(1 - self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1])) * 100:.2f}%\n"
        report += f"Duplicates: {self.df.duplicated().sum()} ({(self.df.duplicated().sum() / len(self.df)) * 100:.2f}%)\n\n"
        
        report += "Column Details:\n"
        report += "-" * 80 + "\n"
        for col in self.df.columns:
            report += f"\n{col}:\n"
            report += f"  Type: {self.df[col].dtype}\n"
            report += f"  Unique: {self.df[col].nunique()}\n"
            report += f"  Missing: {self.df[col].isnull().sum()} ({(self.df[col].isnull().sum() / len(self.df)) * 100:.2f}%)\n"
            if self.df[col].dtype in ['int64', 'float64']:
                report += f"  Mean: {self.df[col].mean():.2f}\n"
                report += f"  Std: {self.df[col].std():.2f}\n"
        
        report_text.insert(1.0, report)
    
    def visualize_data(self):
        """Create data visualizations"""
        if self.df is None:
            messagebox.showwarning("Warning", "No data loaded!")
            return
        
        viz_window = tk.Toplevel(self.root)
        viz_window.title("Data Visualizations")
        viz_window.geometry("1000x800")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Data Overview', fontsize=16)
        
        # Missing values heatmap
        if self.df.isnull().sum().sum() > 0:
            sns.heatmap(self.df.isnull(), cbar=False, ax=axes[0, 0], cmap='viridis')
            axes[0, 0].set_title('Missing Values Pattern')
        
        # Numeric columns distribution
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns[:4]
        for i, col in enumerate(numeric_cols):
            if i < 3:
                row = (i + 1) // 2
                col_idx = (i + 1) % 2
                self.df[col].hist(ax=axes[row, col_idx], bins=30)
                axes[row, col_idx].set_title(f'{col} Distribution')
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, viz_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
    
    def export_cleaning_log(self):
        """Export cleaning history log"""
        if not self.log_text.get(1.0, tk.END).strip():
            messagebox.showwarning("Warning", "No log to export!")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")]
        )
