"""
Combined ERF VCSEL Cavity Analyzer
Combines features from both original and improved versions:
- Load DM3 image files and make intensity linescan
- Show scan profile
- Set number of ERF fitting parameters (k1, k2, ..., k97)
- High-precision fitting with error < 1e-10
- Show scan profile and fitting curve with edge points
- Show layer thickness contribution
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector, Button
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import os
import sys
import json
from datetime import datetime
from typing import Tuple, List, Union, Dict

# Import scipy for optimization and signal processing.
# Note: import failure no longer calls sys.exit() at import time (that made the
# module impossible to import/reuse). The hard requirement is enforced in main().
try:
    from scipy.optimize import curve_fit
    from scipy.special import erf
    from scipy.signal import find_peaks
    HAS_SCIPY = True
    print("✓ SciPy loaded successfully")
except ImportError:
    HAS_SCIPY = False
    print("✗ SciPy not available. Please install: pip install scipy")

# Modular refactor: pure numerics / IO now live in the vcsel_analyzer package.
# The GUI class below delegates its numerical kernels to these tested modules.
from vcsel_analyzer.logging_setup import setup_logging
from vcsel_analyzer.core import units as _units
from vcsel_analyzer.core import erf_model as _erf_model
from vcsel_analyzer.core import thickness as _thickness
from vcsel_analyzer.core import fitting as _fitting
from vcsel_analyzer.io import dm3_loader as _dm3_loader

# Route the package's core-module log messages to stdout so their text matches
# the original print()-based console output.
setup_logging()

# Try to import required libraries for DM3 reading
try:
    import hyperspy.api as hs
    HAS_HYPERSPY = True
    print("✓ HyperSpy loaded successfully")
except ImportError:
    HAS_HYPERSPY = False
    print("Warning: HyperSpy not found. DM3 loading will be limited.")

try:
    import ncempy.io as ncempy_io
    HAS_NCEMPY = True
    print("✓ ncempy loaded successfully")
except ImportError:
    HAS_NCEMPY = False
    print("Warning: ncempy not found.")

# Configuration for ERF fitting (single source of truth in vcsel_analyzer.config)
from vcsel_analyzer.config import ERF_CONFIG


class CombinedVCSELAnalyzer:
    """
    Combined VCSEL cavity analyzer with comprehensive functionality.
    """
    
    def __init__(self):
        # Data storage
        self.image_data = None
        self.image_metadata = None
        self.signal = None  # Store HyperSpy signal object for pixel size extraction
        self.dm3_file_path = None   # path of loaded DM3 file (default export dir)
        self.text_data_path = None  # path of loaded text data (fallback export dir)
        self.linescan_profile = None
        self.linescan_positions = None
        self.current_line = None
        
        # GUI elements
        self.fig = None
        self.ax = None
        self.selector = None
        self.buttons = {}
        
        # ERF fitting parameters
        self.total_params = ERF_CONFIG.get('default_total_params', 97)
        self.erf_parameters = None

        self.fitted_parameters = None
        self.edge_positions = []
        self.layer_thicknesses = {'Material_A': [], 'Material_B': []}
        
        # Custom material names for legend display
        self.material_name_a = "Material A"  # Default name for Material A
        self.material_name_b = "Material B"  # Default name for Material B
        
        # Results storage
        self.fitting_results = {}
        self.final_loss = None
        
        # Interactive selection attributes
        self.selected_start_index = None
        self.selected_end_index = None
        self.selected_positions = None
        self.selected_profile = None
        self.selection_active = False
        self.selection_mode = "none"  # "none", "first_click", "interval_selected"
        self.click_count = 0
        self.first_click_position = None
        self.selection_start_pos = None
        self.selection_end_pos = None
        
        # Selection visual elements (matplotlib objects)
        self.selection_lines = []  # Vertical lines at boundaries
        self.selection_span = None  # Shaded region
        self.selection_text = None  # Selection info text
        self.selection_status = None  # Selection status indicator
        
        # Selection control for fitting
        self.use_selection_for_fitting = True  # Default to using selection when available
        
    def load_dm3_file(self, file_path: str = None) -> bool:
        """
        Load DM3 file using available libraries.
        """
        if file_path is None:
            root = tk.Tk()
            root.withdraw()
            file_path = filedialog.askopenfilename(
                title="Select DM3 Image File",
                filetypes=[("DM3 files", "*.dm3"), ("All files", "*.*")]
            )
            root.destroy()
            
        if not file_path:
            return False

        # Delegate loading to vcsel_analyzer.io.dm3_loader. This adds a fix for
        # HyperSpy returning a list of signals (multi-dataset DM3) and keeps the
        # same console output text via the package logger.
        loaded = _dm3_loader.load_dm3(file_path)
        if loaded is None:
            return False

        self.signal = loaded.signal          # None when loaded via ncempy
        self.image_data = loaded.data
        self.image_metadata = loaded.metadata
        self.dm3_file_path = file_path       # remember source for default export dir
        return True
    
    def create_interactive_interface(self):
        """
        Create the main interactive interface for linescan selection and ERF fitting.
        """
        if self.image_data is None:
            print("No image data loaded. Please load a DM3 file first.")
            return
            
        # Create main figure
        self.fig = plt.figure(figsize=(16, 12))
        
        # Image display
        self.ax_image = plt.subplot2grid((3, 2), (0, 0), colspan=2, rowspan=2)
        
        # Control panel
        self.ax_controls = plt.subplot2grid((3, 2), (2, 0), colspan=2)
        self.ax_controls.axis('off')
        
        # Display the image
        im = self.ax_image.imshow(self.image_data, cmap='gray', origin='lower')
        self.ax_image.set_title('VCSEL Cavity Structure - Select Linescan Region\n'
                               'Click and drag to define linescan area')
        plt.colorbar(im, ax=self.ax_image)
        
        # Initialize rectangle selector
        self.selector = RectangleSelector(
            self.ax_image, 
            self.on_select,
            useblit=True,
            button=[1],  # Only left mouse button
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )
        
        # Create control buttons
        self.create_control_buttons()
        
        plt.tight_layout()
        plt.show()
    
    def create_control_buttons(self):
        """
        Create control buttons for the interface.
        """
        button_width = 0.12
        button_height = 0.04
        
        # Main button configurations (first row)
        buttons_config = [
            ('Extract Profile', 0.05, self.extract_linescan_profile),
            ('Set Parameters', 0.20, self.configure_parameters),
            ('Fit ERF', 0.35, self.perform_erf_fitting),
            ('Show Results', 0.50, self.show_comprehensive_results),
            ('Export Data', 0.65, self.export_results),
            ('Load Text Data', 0.80, self.load_text_data)
        ]
        
        # Create main buttons (first row)
        for name, x_pos, callback in buttons_config:
            ax_button = plt.axes([x_pos, 0.02, button_width, button_height])
            button = Button(ax_button, name)
            button.on_clicked(callback)
            self.buttons[name] = button
        
        # Legend and thickness button configurations (second row)
        legend_buttons_config = [
            ('Set Material Names', 0.05, self.set_material_names),
            ('Legend Input 1', 0.20, self.show_legend_input_window_1),
            ('Legend Input 2', 0.35, self.show_legend_input_window_2)
        ]

        # Create legend buttons (second row)
        for name, x_pos, callback in legend_buttons_config:
            ax_button = plt.axes([x_pos, 0.07, button_width, button_height])
            button = Button(ax_button, name)
            button.on_clicked(callback)
            self.buttons[name] = button

            # Store reference to material names button for text updates
            if name == 'Set Material Names':
                self.material_names_button = button

        # Initialize material names button label to reflect current names
        try:
            self._update_material_names_button_text()
        except Exception as e:
            print(f"Warning: could not initialize material names button text: {e}")

    def on_select(self, eclick, erelease):
        """
        Callback for rectangle selection.
        """
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        
        # Ensure proper ordering
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        self.current_line = (x1, y1, x2, y2)
        print(f"Selected region: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"Width: {x2-x1} pixels, Height: {y2-y1} pixels")
    
    def extract_linescan_profile(self, event=None):
        """
        Extract the linescan profile from selected region.
        """
        if self.current_line is None:
            print("Please select a region first!")
            return
            
        x1, y1, x2, y2 = self.current_line
        
        # Extract the selected region
        region = self.image_data[y1:y2, x1:x2]
        
        # Create linescan profile by averaging
        if region.shape[0] > region.shape[1]:
            # Vertical linescan
            self.linescan_profile = np.mean(region, axis=1)
            positions = np.arange(y1, y2)
            direction = "vertical"
        else:
            # Horizontal linescan
            self.linescan_profile = np.mean(region, axis=0)
            positions = np.arange(x1, x2)
            direction = "horizontal"
        
        # Convert positions to nanometers
        pixel_size = self.get_pixel_size()
        self.linescan_positions = positions * pixel_size
        
        print(f"Extracted {direction} linescan profile:")
        print(f"  Points: {len(self.linescan_profile)}")
        print(f"  Range: {self.linescan_positions[0]:.2f} to {self.linescan_positions[-1]:.2f} nm")
        print(f"  Pixel size: {pixel_size:.3f} nm/pixel")
        
        # Show the linescan profile
        self.show_linescan_profile()
        
        # Save profile data
        self.save_linescan_profile()
    
    def _convert_scale_to_nm(self, scale, units):
        """
        Convert an axis scale given in `units` to nanometers.

        Returns the scale value expressed in nm (float), or None if `units`
        is not a usable length unit (e.g. reciprocal-space or angular units).
        Unrecognized units are assumed to already be in nm (with a warning),
        preserving backward-compatible behavior.
        """
        return _units.convert_scale_to_nm(scale, units)

    def get_pixel_size(self) -> float:
        """
        Extract pixel size from HyperSpy signal axes manager or metadata, or use default.
        """
        # First try to get pixel size from HyperSpy signal axes manager (most reliable)
        if hasattr(self, 'signal') and self.signal is not None:
            try:
                if hasattr(self.signal, 'axes_manager'):
                    axes_manager = self.signal.axes_manager
                    if hasattr(axes_manager, 'signal_axes') and len(axes_manager.signal_axes) > 0:
                        # Get the first signal axis (usually x-axis)
                        first_axis = axes_manager.signal_axes[0]
                        if hasattr(first_axis, 'scale') and hasattr(first_axis, 'units'):
                            units = first_axis.units
                            converted = self._convert_scale_to_nm(first_axis.scale, units)
                            if converted is not None:
                                print(f"Found pixel size from HyperSpy axes: {first_axis.scale} {units} -> {converted} nm")
                                return converted
                            print(f"Could not interpret HyperSpy axis units '{units}'; continuing to fallback.")
            except Exception as e:
                print(f"Could not extract pixel size from HyperSpy axes: {e}")
        
        # Fallback: Try to find pixel size in DM3 metadata (legacy method)
        if self.image_metadata:
            try:
                if 'ImageList' in self.image_metadata:
                    image_data = self.image_metadata['ImageList']['TagGroup0']['ImageData']
                    if 'Calibrations' in image_data:
                        cal = image_data['Calibrations']['Dimension']
                        if len(cal) > 0:
                            pixel_size = float(cal[0].get('Scale', 1.0))
                            unit = cal[0].get('Units', 'nm')
                            converted = self._convert_scale_to_nm(pixel_size, unit)
                            if converted is not None:
                                print(f"Found pixel size from metadata: {pixel_size} {unit} -> {converted} nm")
                                return converted
                            print(f"Could not interpret metadata units '{unit}'; continuing to fallback.")
            except Exception as e:
                print(f"Could not extract pixel size from metadata: {e}")
        
        # Default pixel size
        default_pixel_size = 0.246  # nm per pixel (common for HRTEM)
        print(f"Using default pixel size: {default_pixel_size} nm/pixel")
        return default_pixel_size
    
    def on_selection_click(self, event):
        """
        Handle mouse click events for interactive data selection.
        Implements two-click selection system for defining data intervals.
        Enhanced with selection mode activation/deactivation control.
        
        Args:
            event: matplotlib mouse click event
        """
        # Only handle left mouse button clicks within the plot area
        if event.button != 1 or event.inaxes is None:
            return
        
        # Check if selection mode is enabled (Task 6.1: selection mode activation/deactivation)
        if not getattr(self, 'selection_enabled', False):
            print("Selection mode is disabled. Enable selection to interact with the plot.")
            return
        
        # Only process clicks if we have linescan data
        if self.linescan_positions is None or self.linescan_profile is None:
            print("No linescan data available for selection")
            return
        
        # Get click position
        click_x = event.xdata
        if click_x is None:
            return
        
        print(f"Click detected at position: {click_x:.2f} nm")
        
        # Convert click position to data index
        click_index = self._position_to_index(click_x)
        if click_index is None:
            print(f"Click position {click_x:.2f} nm is outside data range")
            return
        
        # Handle click based on current selection state
        if self.selection_mode == "none":
            # First click - start selection
            if self._validate_click_position(click_x, click_index):
                self.selection_mode = "first_click"
                self.click_count = 1
                self.first_click_position = click_x
                self.selected_start_index = click_index
                self.selection_start_pos = click_x
                
                print(f"Selection started at position {click_x:.2f} nm (index {click_index})")
                print("Click again to complete selection")
            else:
                print("Invalid click position for selection start")
            
        elif self.selection_mode == "first_click":
            # Second click - complete selection
            if self._validate_click_position(click_x, click_index):
                self.click_count = 2
                self.selected_end_index = click_index
                self.selection_end_pos = click_x
                
                # Validate and order the selection
                if self._validate_selection():
                    self.selection_mode = "interval_selected"
                    self.selection_active = True
                    
                    # Extract selected data
                    self._extract_selected_data()
                    
                    # Update visual feedback
                    self._update_selection_visual_feedback()
                    
                    # Update status display
                    self._update_status_display()
                    
                    if self.selected_positions is not None:
                        print(f"Selection completed: {self.selection_start_pos:.2f} to {self.selection_end_pos:.2f} nm")
                        print(f"Selected {len(self.selected_positions)} data points")
                    else:
                        print("Selection completed but data extraction failed")
                else:
                    # Invalid selection - reset
                    self._clear_selection_state()
                    print("Invalid selection - please try again")
            else:
                print("Invalid click position for selection end")
                
        elif self.selection_mode == "interval_selected":
            # Task 7.2: Enhanced selection replacement capability
            if self._validate_click_position(click_x, click_index):
                print(f"Replacing existing selection (was: {self.selection_start_pos:.2f} to {self.selection_end_pos:.2f} nm)")
                
                # Store previous selection info for potential rollback
                prev_selection_data = {
                    'start_pos': self.selection_start_pos,
                    'end_pos': self.selection_end_pos,
                    'start_index': self.selected_start_index,
                    'end_index': self.selected_end_index,
                    'positions': self.selected_positions.copy() if self.selected_positions is not None else None,
                    'profile': self.selected_profile.copy() if self.selected_profile is not None else None
                }
                
                # Clear previous selection state and visuals
                self._clear_selection_state()
                
                # Start new selection
                self.selection_mode = "first_click"
                self.click_count = 1
                self.first_click_position = click_x
                self.selected_start_index = click_index
                self.selection_start_pos = click_x
                
                # Update visual feedback to show new selection start
                self._update_selection_visual_feedback_partial()
                
                # Update status display
                if hasattr(self, '_update_status_display'):
                    self._update_status_display()
                
                print(f"New selection started at position {click_x:.2f} nm (index {click_index})")
                print("Click again to complete new selection")
            else:
                print("Invalid click position for new selection start")
    
    def _position_to_index(self, position):
        """
        Convert a position value to the nearest data index with comprehensive validation.
        
        Args:
            position (float): Position in nm
            
        Returns:
            int or None: Data index, or None if position is outside range
        """
        if self.linescan_positions is None or len(self.linescan_positions) == 0:
            return None
        
        # Validate input position
        if not isinstance(position, (int, float)) or not np.isfinite(position):
            print(f"Invalid click position: {position}")
            return None
        
        # Get data range with boundary validation
        try:
            min_pos = np.min(self.linescan_positions)
            max_pos = np.max(self.linescan_positions)
        except Exception as e:
            print(f"Error determining data range: {e}")
            return None
        
        # Check if position is within data range with tolerance
        position_tolerance = (max_pos - min_pos) * 0.01  # 1% tolerance
        
        if position < (min_pos - position_tolerance):
            print(f"Click position {position:.2f} nm is below data range [{min_pos:.2f}, {max_pos:.2f}] nm")
            return None
        
        if position > (max_pos + position_tolerance):
            print(f"Click position {position:.2f} nm is above data range [{min_pos:.2f}, {max_pos:.2f}] nm")
            return None
        
        # Find nearest index with error handling
        try:
            distances = np.abs(self.linescan_positions - position)
            nearest_index = np.argmin(distances)
            
            # Validate the found index
            if nearest_index < 0 or nearest_index >= len(self.linescan_positions):
                print(f"Invalid index calculated: {nearest_index}")
                return None
            
            # Additional validation: check if the nearest position is reasonable
            nearest_position = self.linescan_positions[nearest_index]
            distance_to_nearest = abs(nearest_position - position)
            max_reasonable_distance = (max_pos - min_pos) * 0.1  # 10% of total range
            
            if distance_to_nearest > max_reasonable_distance:
                print(f"Click position {position:.2f} nm is too far from nearest data point {nearest_position:.2f} nm")
                return None
            
            return int(nearest_index)
            
        except Exception as e:
            print(f"Error finding nearest index for position {position:.2f}: {e}")
            return None
    
    def _validate_selection(self):
        """
        Comprehensive validation of the current selection with automatic ordering and boundary handling.
        
        Returns:
            bool: True if selection is valid, False otherwise
        """
        # Basic null checks
        if (self.selected_start_index is None or 
            self.selected_end_index is None or
            self.linescan_positions is None or
            self.linescan_profile is None):
            print("Selection validation failed: missing required data")
            return False
        
        # Validate data arrays
        if len(self.linescan_positions) == 0 or len(self.linescan_profile) == 0:
            print("Selection validation failed: empty data arrays")
            return False
        
        if len(self.linescan_positions) != len(self.linescan_profile):
            print("Selection validation failed: position and profile arrays have different lengths")
            return False
        
        # Ensure indices are within bounds
        max_index = len(self.linescan_positions) - 1
        
        if (self.selected_start_index < 0 or self.selected_start_index > max_index):
            print(f"Selection validation failed: start index {self.selected_start_index} out of bounds [0, {max_index}]")
            return False
            
        if (self.selected_end_index < 0 or self.selected_end_index > max_index):
            print(f"Selection validation failed: end index {self.selected_end_index} out of bounds [0, {max_index}]")
            return False
        
        # Automatically order start/end positions (left-to-right) based on position values
        start_pos = self.linescan_positions[self.selected_start_index]
        end_pos = self.linescan_positions[self.selected_end_index]
        
        if start_pos > end_pos:
            # Swap indices and positions for left-to-right ordering
            self.selected_start_index, self.selected_end_index = self.selected_end_index, self.selected_start_index
            self.selection_start_pos, self.selection_end_pos = self.selection_end_pos, self.selection_start_pos
            print(f"Selection automatically ordered left-to-right: {self.selection_start_pos:.2f} to {self.selection_end_pos:.2f} nm")
        
        # Validate minimum interval size (at least 3 data points)
        interval_size = abs(self.selected_end_index - self.selected_start_index) + 1
        min_required_points = 3
        
        if interval_size < min_required_points:
            print(f"Selection too small: {interval_size} points selected, minimum {min_required_points} required")
            print(f"Please select a larger interval (current: {abs(self.selection_end_pos - self.selection_start_pos):.2f} nm)")
            return False
        
        # Validate that selection doesn't exceed reasonable limits
        max_reasonable_points = len(self.linescan_positions)
        if interval_size > max_reasonable_points:
            print(f"Selection too large: {interval_size} points exceeds maximum {max_reasonable_points}")
            return False
        
        # Validate position consistency
        try:
            actual_start_pos = self.linescan_positions[self.selected_start_index]
            actual_end_pos = self.linescan_positions[self.selected_end_index]
            
            # Check if stored positions match actual positions (with small tolerance)
            pos_tolerance = abs(self.linescan_positions[-1] - self.linescan_positions[0]) * 0.001  # 0.1% tolerance
            
            if abs(actual_start_pos - self.selection_start_pos) > pos_tolerance:
                print(f"Position inconsistency detected at start: stored {self.selection_start_pos:.2f}, actual {actual_start_pos:.2f}")
                # Update stored position to match actual
                self.selection_start_pos = actual_start_pos
            
            if abs(actual_end_pos - self.selection_end_pos) > pos_tolerance:
                print(f"Position inconsistency detected at end: stored {self.selection_end_pos:.2f}, actual {actual_end_pos:.2f}")
                # Update stored position to match actual
                self.selection_end_pos = actual_end_pos
                
        except Exception as e:
            print(f"Error validating position consistency: {e}")
            return False
        
        # Validate that selected region contains valid data
        try:
            start_idx = min(self.selected_start_index, self.selected_end_index)
            end_idx = max(self.selected_start_index, self.selected_end_index)
            
            selected_profile_subset = self.linescan_profile[start_idx:end_idx+1]
            
            # Check for NaN or infinite values
            if np.any(~np.isfinite(selected_profile_subset)):
                print("Selection contains invalid data (NaN or infinite values)")
                return False
            
            # Check for reasonable data variation (not all zeros or constant)
            if np.all(selected_profile_subset == selected_profile_subset[0]):
                print("Warning: Selected region contains constant values")
                # Don't fail validation, but warn user
            
        except Exception as e:
            print(f"Error validating selected data: {e}")
            return False
        
        # All validation checks passed
        selection_length = abs(self.selection_end_pos - self.selection_start_pos)
        print(f"Selection validated: {interval_size} points, {selection_length:.2f} nm length")
        
        return True
    
    def _extract_selected_data(self):
        """
        Extract the selected data subset from the full dataset with comprehensive validation.
        """
        if (self.selected_start_index is None or 
            self.selected_end_index is None or
            self.linescan_positions is None or
            self.linescan_profile is None):
            print("Cannot extract selected data: missing selection or data")
            return
        
        try:
            # Ensure proper ordering of indices
            start_idx = min(self.selected_start_index, self.selected_end_index)
            end_idx = max(self.selected_start_index, self.selected_end_index)
            
            # Validate indices are within bounds
            max_index = len(self.linescan_positions) - 1
            if start_idx < 0 or end_idx > max_index:
                print(f"Cannot extract data: indices [{start_idx}, {end_idx}] out of bounds [0, {max_index}]")
                return
            
            # Extract selected positions and intensities (inclusive of end point)
            self.selected_positions = self.linescan_positions[start_idx:end_idx+1].copy()
            self.selected_profile = self.linescan_profile[start_idx:end_idx+1].copy()
            
            # Validate extracted data
            if len(self.selected_positions) == 0 or len(self.selected_profile) == 0:
                print("Error: extracted data is empty")
                self.selected_positions = None
                self.selected_profile = None
                return
            
            if len(self.selected_positions) != len(self.selected_profile):
                print("Error: extracted position and profile arrays have different lengths")
                self.selected_positions = None
                self.selected_profile = None
                return
            
            # Check for valid data
            if (np.any(~np.isfinite(self.selected_positions)) or 
                np.any(~np.isfinite(self.selected_profile))):
                print("Warning: extracted data contains invalid values")
            
            # Calculate selection statistics
            selection_length = self.selected_positions[-1] - self.selected_positions[0]
            intensity_range = np.max(self.selected_profile) - np.min(self.selected_profile)
            
            print(f"Extracted selected data: {len(self.selected_positions)} points")
            print(f"Position range: {self.selected_positions[0]:.2f} to {self.selected_positions[-1]:.2f} nm")
            print(f"Selection length: {selection_length:.2f} nm")
            print(f"Intensity range: {np.min(self.selected_profile):.1f} to {np.max(self.selected_profile):.1f}")
            
        except Exception as e:
            print(f"Error extracting selected data: {e}")
            self.selected_positions = None
            self.selected_profile = None
    
    def _validate_click_position(self, position, index):
        """
        Validate that a click position is suitable for selection.
        
        Args:
            position (float): Click position in nm
            index (int): Corresponding data index
            
        Returns:
            bool: True if position is valid for selection, False otherwise
        """
        if position is None or index is None:
            return False
        
        if self.linescan_positions is None or self.linescan_profile is None:
            return False
        
        # Check if index is within valid range
        if index < 0 or index >= len(self.linescan_positions):
            print(f"Click index {index} is outside valid range [0, {len(self.linescan_positions)-1}]")
            return False
        
        # For first click, any valid position is acceptable
        if self.selection_mode == "none":
            return True
        
        # For second click, ensure minimum distance from first click
        if self.selection_mode == "first_click" and self.selected_start_index is not None:
            min_points = 3  # Minimum interval size
            distance = abs(index - self.selected_start_index)
            
            if distance < (min_points - 1):  # -1 because we count inclusive
                actual_distance = abs(position - self.first_click_position)
                print(f"Second click too close to first click: {actual_distance:.2f} nm")
                print(f"Minimum selection size is {min_points} data points")
                return False
        
        return True
    
    def _clear_selection_state(self):
        """
        Clear the current selection state and reset all selection variables.
        """
        self.selection_mode = "none"
        self.click_count = 0
        self.first_click_position = None
        self.selected_start_index = None
        self.selected_end_index = None
        self.selection_start_pos = None
        self.selection_end_pos = None
        self.selected_positions = None
        self.selected_profile = None
        self.selection_active = False
        
        # Clear visual feedback
        self._clear_selection_visuals()
    
    def clear_selection(self):
        """
        Public method to clear the current selection and reset all selection state.
        Implements task 7.1: Create selection clearing and reset functionality.
        
        This method:
        - Removes current selection
        - Updates all visual indicators when selection is cleared
        - Resets selection state variables to initial values
        
        Requirements: 3.4, 3.3
        """
        if not self.selection_active and self.selection_mode == "none":
            print("No active selection to clear")
            return False
        
        print("Clearing current selection...")
        
        # Clear the selection state and visuals
        self._clear_selection_state()
        
        # Update status display if available
        if hasattr(self, '_update_status_display'):
            self._update_status_display()
        
        # Redraw the plot if available
        if hasattr(self, 'current_fig') and self.current_fig is not None:
            try:
                self.current_fig.canvas.draw_idle()
            except Exception as e:
                print(f"Warning: Could not redraw plot after clearing selection: {e}")
        
        print("✓ Selection cleared successfully")
        return True
    
    def replace_selection(self, new_start_pos, new_end_pos):
        """
        Replace the current selection with a new one, maintaining data integrity.
        Implements task 7.2: Add selection replacement capability.
        
        Args:
            new_start_pos (float): New selection start position in nm
            new_end_pos (float): New selection end position in nm
            
        Returns:
            bool: True if replacement was successful, False otherwise
            
        Requirements: 3.1, 3.2
        """
        if self.linescan_positions is None or self.linescan_profile is None:
            print("Cannot replace selection: no linescan data available")
            return False
        
        # Convert positions to indices
        new_start_index = self._position_to_index(new_start_pos)
        new_end_index = self._position_to_index(new_end_pos)
        
        if new_start_index is None or new_end_index is None:
            print("Cannot replace selection: invalid positions")
            return False
        
        # Store current selection for potential rollback
        old_selection = {
            'start_pos': self.selection_start_pos,
            'end_pos': self.selection_end_pos,
            'start_index': self.selected_start_index,
            'end_index': self.selected_end_index,
            'positions': self.selected_positions.copy() if self.selected_positions is not None else None,
            'profile': self.selected_profile.copy() if self.selected_profile is not None else None,
            'active': self.selection_active,
            'mode': self.selection_mode
        }
        
        print(f"Replacing selection: {old_selection['start_pos']:.2f}-{old_selection['end_pos']:.2f} nm → {new_start_pos:.2f}-{new_end_pos:.2f} nm")
        
        # Set new selection parameters
        self.selected_start_index = new_start_index
        self.selected_end_index = new_end_index
        self.selection_start_pos = new_start_pos
        self.selection_end_pos = new_end_pos
        self.selection_mode = "interval_selected"
        self.selection_active = True
        
        # Validate the new selection
        if not self._validate_selection():
            print("New selection validation failed, rolling back to previous selection")
            # Rollback to previous selection
            self.selection_start_pos = old_selection['start_pos']
            self.selection_end_pos = old_selection['end_pos']
            self.selected_start_index = old_selection['start_index']
            self.selected_end_index = old_selection['end_index']
            self.selected_positions = old_selection['positions']
            self.selected_profile = old_selection['profile']
            self.selection_active = old_selection['active']
            self.selection_mode = old_selection['mode']
            return False
        
        # Extract new selected data
        self._extract_selected_data()
        
        # Validate extracted data integrity
        if self.selected_positions is None or self.selected_profile is None:
            print("Data extraction failed for new selection, rolling back")
            # Rollback to previous selection
            self.selection_start_pos = old_selection['start_pos']
            self.selection_end_pos = old_selection['end_pos']
            self.selected_start_index = old_selection['start_index']
            self.selected_end_index = old_selection['end_index']
            self.selected_positions = old_selection['positions']
            self.selected_profile = old_selection['profile']
            self.selection_active = old_selection['active']
            self.selection_mode = old_selection['mode']
            return False
        
        # Update visual feedback when selection changes
        self._update_selection_visual_feedback()
        
        # Update status display
        if hasattr(self, '_update_status_display'):
            self._update_status_display()
        
        print(f"✓ Selection replaced successfully: {len(self.selected_positions)} points, {abs(new_end_pos - new_start_pos):.2f} nm")
        return True
    
    def _clear_selection_visuals(self):
        """
        Clear visual selection indicators from the plot.
        Enhanced to handle all visual elements including status indicator.
        """
        # Remove selection lines
        for line in self.selection_lines:
            try:
                line.remove()
            except Exception:
                pass
        self.selection_lines.clear()
        
        # Remove selection span
        if self.selection_span is not None:
            try:
                self.selection_span.remove()
            except Exception:
                pass
            self.selection_span = None
        
        # Remove selection text
        if self.selection_text is not None:
            try:
                self.selection_text.remove()
            except Exception:
                pass
            self.selection_text = None
        
        # Remove selection status indicator
        if hasattr(self, 'selection_status') and self.selection_status is not None:
            try:
                self.selection_status.remove()
            except Exception:
                pass
            self.selection_status = None
        
        # Redraw the plot
        if hasattr(self, 'current_ax') and self.current_ax is not None:
            try:
                self.current_ax.figure.canvas.draw_idle()
            except Exception:
                pass
    
    def _update_selection_visual_feedback(self):
        """
        Update visual feedback to show the current selection with enhanced highlighting.
        Implements task 3.1: selection highlighting on main plot with vertical lines and shaded region.
        """
        if (not hasattr(self, 'current_ax') or 
            self.current_ax is None or
            self.selection_start_pos is None or
            self.selection_end_pos is None):
            return
        
        try:
            # Clear existing visuals first
            self._clear_selection_visuals()
            
            # Task 3.1: Add vertical lines at selection boundaries using matplotlib axvline
            start_line = self.current_ax.axvline(x=self.selection_start_pos, 
                                               color='red', linestyle='--', 
                                               linewidth=2.5, alpha=0.9, 
                                               label='Selection Start',
                                               zorder=10)  # Ensure lines appear on top
            end_line = self.current_ax.axvline(x=self.selection_end_pos, 
                                             color='red', linestyle='--', 
                                             linewidth=2.5, alpha=0.9, 
                                             label='Selection End',
                                             zorder=10)
            
            self.selection_lines = [start_line, end_line]
            
            # Task 3.1: Create shaded region between selection points using axvspan
            self.selection_span = self.current_ax.axvspan(self.selection_start_pos, 
                                                        self.selection_end_pos,
                                                        alpha=0.25, color='lightblue',
                                                        label='Selected Region',
                                                        zorder=5)  # Behind lines but above data
            
            # Task 3.2: Add selection information display
            selection_length = abs(self.selection_end_pos - self.selection_start_pos)
            num_points = len(self.selected_positions) if self.selected_positions is not None else 0
            
            # Show start and end positions in numerical form
            info_text = (f"Selection Range:\n"
                        f"Start: {self.selection_start_pos:.2f} nm\n"
                        f"End: {self.selection_end_pos:.2f} nm\n"
                        f"Length: {selection_length:.2f} nm\n"
                        f"Data Points: {num_points}")
            
            # Position text in upper right corner with enhanced styling
            self.selection_text = self.current_ax.text(0.98, 0.98, info_text,
                                                     transform=self.current_ax.transAxes,
                                                     verticalalignment='top',
                                                     horizontalalignment='right',
                                                     bbox=dict(boxstyle='round,pad=0.5', 
                                                             facecolor='lightyellow', 
                                                             edgecolor='orange',
                                                             alpha=0.9),
                                                     fontsize=10,
                                                     fontweight='bold',
                                                     zorder=15)  # Ensure text appears on top
            
            # Add selection status indicator
            status_text = "✓ Data Interval Selected"
            self.selection_status = self.current_ax.text(0.02, 0.98, status_text,
                                                       transform=self.current_ax.transAxes,
                                                       verticalalignment='top',
                                                       horizontalalignment='left',
                                                       bbox=dict(boxstyle='round,pad=0.3', 
                                                               facecolor='lightgreen', 
                                                               edgecolor='green',
                                                               alpha=0.8),
                                                       fontsize=9,
                                                       fontweight='bold',
                                                       color='darkgreen',
                                                       zorder=15)
            
            # Update legend to show selection elements
            self.current_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), 
                                 ncol=3, frameon=True, fancybox=True, shadow=True)
            
            # Task 3.1: Implement dynamic visual updates - redraw the plot
            self.current_ax.figure.canvas.draw_idle()  # Use draw_idle for better performance
            
            print(f"✓ Visual feedback updated: {self.selection_start_pos:.2f} to {self.selection_end_pos:.2f} nm")
            
        except Exception as e:
            print(f"Error updating visual feedback: {e}")
            import traceback
            traceback.print_exc()

    def _update_selection_visual_feedback_partial(self):
        """
        Update visual feedback to show partial selection (start point only).
        Used during selection replacement to show the new selection start.
        Implements task 7.2: Update visual feedback when selection changes.
        """
        if (not hasattr(self, 'current_ax') or 
            self.current_ax is None or
            self.selection_start_pos is None):
            return
        
        try:
            # Clear existing visuals first
            self._clear_selection_visuals()
            
            # Show only the start line for partial selection
            start_line = self.current_ax.axvline(x=self.selection_start_pos, 
                                               color='orange', linestyle=':', 
                                               linewidth=2.0, alpha=0.8, 
                                               label='Selection Start (Partial)',
                                               zorder=10)
            
            self.selection_lines = [start_line]
            
            # Add partial selection status indicator
            status_text = "⚡ Starting New Selection..."
            self.selection_status = self.current_ax.text(0.02, 0.98, status_text,
                                                       transform=self.current_ax.transAxes,
                                                       verticalalignment='top',
                                                       horizontalalignment='left',
                                                       bbox=dict(boxstyle='round,pad=0.3', 
                                                               facecolor='lightyellow', 
                                                               edgecolor='orange',
                                                               alpha=0.8),
                                                       fontsize=9,
                                                       fontweight='bold',
                                                       color='darkorange',
                                                       zorder=15)
            
            # Show partial selection info
            info_text = f"New Selection Start:\n{self.selection_start_pos:.2f} nm\n(Click to complete)"
            self.selection_text = self.current_ax.text(0.98, 0.98, info_text,
                                                     transform=self.current_ax.transAxes,
                                                     verticalalignment='top',
                                                     horizontalalignment='right',
                                                     bbox=dict(boxstyle='round,pad=0.5', 
                                                             facecolor='lightcyan', 
                                                             edgecolor='orange',
                                                             alpha=0.9),
                                                     fontsize=10,
                                                     fontweight='bold',
                                                     zorder=15)
            
            # Redraw the plot
            self.current_ax.figure.canvas.draw_idle()
            
            print(f"✓ Partial visual feedback updated: start at {self.selection_start_pos:.2f} nm")
            
        except Exception as e:
            print(f"Error updating partial visual feedback: {e}")

    def show_linescan_profile(self):
        """
        Show the extracted linescan profile with interactive selection capability.
        Enhanced with selection mode controls and UI buttons for selection management.
        """
        if self.linescan_profile is None:
            return
            
        # Create figure with space for control buttons
        fig = plt.figure(figsize=(16, 12))
        
        # Main plot area (takes most of the space)
        ax = plt.subplot2grid((10, 1), (0, 0), rowspan=8)
        ax.plot(self.linescan_positions, self.linescan_profile, 'b.-', 
                linewidth=1, markersize=2, label='Linescan Profile')
        ax.set_xlabel('Position [nm]')
        ax.set_ylabel('Intensity')
        
        # Update title based on selection mode
        if hasattr(self, 'selection_enabled') and self.selection_enabled:
            ax.set_title('Extracted Linescan Profile from VCSEL Cavity\n'
                        'Selection Mode: ENABLED - Click twice to select data interval')
        else:
            ax.set_title('Extracted Linescan Profile from VCSEL Cavity\n'
                        'Selection Mode: DISABLED - Enable selection to interact')
        
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Store reference to current axes for selection functionality
        self.current_ax = ax
        
        # Initialize selection mode if not already set
        if not hasattr(self, 'selection_enabled'):
            self.selection_enabled = False
        
        # Connect mouse click event for interactive selection (always connected, but behavior depends on selection_enabled)
        self.click_connection = fig.canvas.mpl_connect('button_press_event', self.on_selection_click)
        
        # Store figure reference for button callbacks
        self.current_fig = fig
        
        # Create control button area
        button_area = plt.subplot2grid((10, 1), (8, 0), rowspan=2)
        button_area.axis('off')
        
        # Add UI control buttons for selection management (Task 6.2)
        self._create_selection_control_buttons(fig)
        
        # If there's an existing selection, show it
        if (self.selection_active and 
            self.selection_start_pos is not None and 
            self.selection_end_pos is not None):
            self._update_selection_visual_feedback()
        
        plt.tight_layout()
        plt.show()
    
    def _create_selection_control_buttons(self, fig):
        """
        Create UI control buttons for selection management.
        Implements task 6.2: Add UI control buttons for selection management.
        """
        button_width = 0.15
        button_height = 0.04
        button_y = 0.02
        
        # Button configurations with callbacks
        buttons_config = [
            ('Enable Selection', 0.05, self._toggle_selection_mode, 'lightgreen' if getattr(self, 'selection_enabled', False) else 'lightcoral'),
            ('Clear Selection', 0.22, self._clear_selection_callback, 'orange'),
            ('Plot Selected', 0.39, self._plot_selected_callback, 'lightblue'),
            ('Use for Fitting', 0.56, self._toggle_fitting_mode, 'lightgreen' if getattr(self, 'use_selection_for_fitting', True) else 'lightgray'),
            ('Show Full Data', 0.73, self._show_full_data_callback, 'lightyellow')
        ]
        
        # Store button references for later updates
        if not hasattr(self, 'selection_buttons'):
            self.selection_buttons = {}
        
        for name, x_pos, callback, color in buttons_config:
            # Create button axes
            ax_button = plt.axes([x_pos, button_y, button_width, button_height])
            
            # Create button with color
            from matplotlib.widgets import Button
            button = Button(ax_button, name, color=color)
            button.on_clicked(callback)
            
            # Store button reference
            self.selection_buttons[name] = button
        
        # Add status text area
        self.status_text_ax = plt.axes([0.05, 0.08, 0.85, 0.02])
        self.status_text_ax.axis('off')
        
        # Update status display
        self._update_status_display()
    
    def _toggle_selection_mode(self, event):
        """
        Toggle selection mode activation/deactivation.
        Implements task 6.1: Add selection mode activation/deactivation.
        """
        # Toggle selection mode
        self.selection_enabled = not getattr(self, 'selection_enabled', False)
        
        # Update button color and text
        if 'Enable Selection' in self.selection_buttons:
            button = self.selection_buttons['Enable Selection']
            if self.selection_enabled:
                button.color = 'lightgreen'
                button.label.set_text('Disable Selection')
            else:
                button.color = 'lightcoral'
                button.label.set_text('Enable Selection')
            # Force button redraw
            button.ax.set_facecolor(button.color)
        
        # Update plot title
        if hasattr(self, 'current_ax') and self.current_ax is not None:
            if self.selection_enabled:
                self.current_ax.set_title('Extracted Linescan Profile from VCSEL Cavity\n'
                                        'Selection Mode: ENABLED - Click twice to select data interval')
            else:
                self.current_ax.set_title('Extracted Linescan Profile from VCSEL Cavity\n'
                                        'Selection Mode: DISABLED - Enable selection to interact')
        
        # Update status display
        self._update_status_display()
        
        # Redraw the plot
        if hasattr(self, 'current_fig') and self.current_fig is not None:
            self.current_fig.canvas.draw_idle()
        
        print(f"Selection mode {'ENABLED' if self.selection_enabled else 'DISABLED'}")
    
    def _clear_selection_callback(self, event):
        """
        Clear Selection button callback.
        Implements task 6.2: Add "Clear Selection" button to reset selection state.
        """
        if self.selection_active or self.selection_mode != "none":
            self._clear_selection_state()
            self._update_status_display()
            
            # Redraw the plot
            if hasattr(self, 'current_fig') and self.current_fig is not None:
                self.current_fig.canvas.draw_idle()
            
            print("Selection cleared")
        else:
            print("No selection to clear")
    
    def _plot_selected_callback(self, event):
        """
        Plot Selected button callback.
        Implements task 6.2: Implement "Plot Selected" button to show selected interval.
        """
        if not self.selection_active or self.selected_positions is None:
            print("No selection available to plot")
            return
        
        # Call the existing plot_selected_interval method
        self.plot_selected_interval()
    
    def _toggle_fitting_mode(self, event):
        """
        Toggle whether to use selected data for fitting.
        Implements task 6.2: Add "Use Selected for Fitting" checkbox option.
        """
        # Toggle fitting mode
        self.use_selection_for_fitting = not getattr(self, 'use_selection_for_fitting', True)
        
        # Update button color and text
        if 'Use for Fitting' in self.selection_buttons:
            button = self.selection_buttons['Use for Fitting']
            if self.use_selection_for_fitting:
                button.color = 'lightgreen'
                button.label.set_text('Use for Fitting ✓')
            else:
                button.color = 'lightgray'
                button.label.set_text('Use Full Data')
            # Force button redraw
            button.ax.set_facecolor(button.color)
        
        # Update status display
        self._update_status_display()
        
        # Redraw the plot
        if hasattr(self, 'current_fig') and self.current_fig is not None:
            self.current_fig.canvas.draw_idle()
        
        fitting_mode = "selected data" if self.use_selection_for_fitting else "full dataset"
        print(f"Fitting mode set to use: {fitting_mode}")
    
    def _show_full_data_callback(self, event):
        """
        Show full data button callback - displays the complete linescan profile.
        """
        if self.linescan_profile is None:
            print("No linescan data available")
            return
        
        # Create a new figure showing the full data
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(self.linescan_positions, self.linescan_profile, 'b.-', 
                linewidth=1, markersize=2, label='Full Linescan Profile')
        
        # Highlight selected region if available
        if (self.selection_active and 
            self.selection_start_pos is not None and 
            self.selection_end_pos is not None):
            ax.axvspan(self.selection_start_pos, self.selection_end_pos,
                      alpha=0.3, color='yellow', label='Selected Region')
        
        ax.set_xlabel('Position [nm]')
        ax.set_ylabel('Intensity')
        ax.set_title('Complete Linescan Profile Overview')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        print("Full data overview displayed")
    
    def _update_status_display(self):
        """
        Update the status display text showing current selection and mode information.
        """
        if not hasattr(self, 'status_text_ax') or self.status_text_ax is None:
            return
        
        # Clear existing text
        self.status_text_ax.clear()
        self.status_text_ax.axis('off')
        
        # Build status message
        status_parts = []
        
        # Selection mode status
        if getattr(self, 'selection_enabled', False):
            status_parts.append("Selection: ENABLED")
        else:
            status_parts.append("Selection: DISABLED")
        
        # Current selection status
        if self.selection_active and self.selected_positions is not None:
            length = abs(self.selection_end_pos - self.selection_start_pos)
            points = len(self.selected_positions)
            status_parts.append(f"Selected: {self.selection_start_pos:.1f}-{self.selection_end_pos:.1f}nm ({points} pts, {length:.1f}nm)")
        else:
            status_parts.append("Selected: None")
        
        # Fitting mode status
        if getattr(self, 'use_selection_for_fitting', True):
            if self.selection_active:
                status_parts.append("Fitting: Will use SELECTED data")
            else:
                status_parts.append("Fitting: Will use FULL data (no selection)")
        else:
            status_parts.append("Fitting: Will use FULL data")
        
        # Display status text
        status_text = " | ".join(status_parts)
        self.status_text_ax.text(0.5, 0.5, status_text, 
                               transform=self.status_text_ax.transAxes,
                               ha='center', va='center',
                               fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor='lightblue', 
                                       alpha=0.7))
    
    def save_linescan_profile(self):
        """
        Save the linescan profile to a text file.
        """
        if self.linescan_profile is None:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'vcsel_linescan_profile_{timestamp}.txt'
        
        data = np.column_stack((self.linescan_positions, self.linescan_profile))
        header = f'Position[nm]\tIntensity\n# Extracted on {datetime.now()}\n# Points: {len(data)}'
        
        np.savetxt(filename, data, delimiter='\t', header=header, comments='')
        print(f"Linescan profile saved to: {filename}")
        return filename
    
    def load_text_data(self, event=None):
        """
        Load linescan data from text file.
        """
        root = tk.Tk()
        root.withdraw()
        
        file_path = filedialog.askopenfilename(
            title="Select linescan data file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                data = np.loadtxt(file_path, delimiter='\t')
                if data.shape[1] >= 2:
                    self.linescan_positions = data[:, 0]
                    self.linescan_profile = data[:, 1]
                    self.text_data_path = file_path  # fallback default export dir
                    print(f"Loaded data from {file_path}")
                    print(f"Points: {len(self.linescan_profile)}")
                    
                    # Show loaded profile
                    self.show_linescan_profile()
                else:
                    messagebox.showerror("Error", "File must have at least 2 columns")
            except Exception as e:
                messagebox.showerror("Error", f"Could not load file: {e}")
        
        root.destroy() 
   
    def configure_parameters(self, event=None):
        """
        Configure ERF fitting parameters.
        Uses a temporary Tk root for the number dialog, then opens the
        parameter window with its own mainloop.
        """
        # Use a temporary root for the number input dialog
        temp_root = tk.Tk()
        temp_root.withdraw()
        
        # Get total number of parameters
        max_params = ERF_CONFIG.get('max_layers', 200) * 3 + 1
        
        total_params = simpledialog.askinteger(
            "ERF Parameters",
            f"Total number of parameters k1, k2, ..., kN (1-{max_params}):",
            initialvalue=self.total_params,
            minvalue=1,
            maxvalue=max_params,
            parent=temp_root
        )
        
        # Destroy the temporary root immediately
        temp_root.destroy()
        
        if total_params:
            self.total_params = total_params
            print(f"Set total parameters to: {total_params}")
            
            try:
                self.initialize_parameters()
                # show_parameter_window creates its own Tk root + mainloop
                self.show_parameter_window()
            except Exception as e:
                print(f"ERROR in configure_parameters: {e}")
                import traceback
                traceback.print_exc()
    
    def initialize_parameters(self):
        """
        Initialize ERF parameters with smart defaults based on data analysis.
        When no linescan is loaded, use simple defaults so the parameter window
        can open and the user can load parameters from file or edit manually.
        """
        if self.linescan_profile is None:
            # Ensure parameter window can open: init with simple defaults
            self.erf_parameters = [1.0 if i == 0 else 0.0 for i in range(self.total_params)]
            print("No linescan profile yet. Parameters initialized to defaults (k1=1, rest=0).")
            print("You can load parameters from file or extract a profile first for smart defaults.")
            return
        
        # Reproducible random source for initial parameter estimates
        rng = np.random.RandomState(ERF_CONFIG.get('random_seed', 0))

        # Analyze the data to get better initial estimates
        baseline = np.mean(self.linescan_profile)
        intensity_range = np.max(self.linescan_profile) - np.min(self.linescan_profile)
        position_range = self.linescan_positions[-1] - self.linescan_positions[0]
        
        # Detect approximate edge positions using gradient analysis
        gradient = np.gradient(self.linescan_profile)
        gradient_abs = np.abs(gradient)
        
        # Find peaks in gradient (potential edge positions)
        if HAS_SCIPY:
            try:
                peaks, _ = find_peaks(gradient_abs, height=np.std(gradient_abs), distance=len(gradient_abs)//50)
                detected_positions = self.linescan_positions[peaks]
            except Exception:
                # Fallback to uniform distribution
                num_components = (self.total_params - 1) // 3
                detected_positions = np.linspace(self.linescan_positions[0] + position_range*0.1, 
                                               self.linescan_positions[-1] - position_range*0.1, 
                                               num_components)
        else:
            # Simple peak detection without scipy
            threshold = np.mean(gradient_abs) + 2 * np.std(gradient_abs)
            peaks = []
            for i in range(1, len(gradient_abs)-1):
                if (gradient_abs[i] > threshold and 
                    gradient_abs[i] > gradient_abs[i-1] and 
                    gradient_abs[i] > gradient_abs[i+1]):
                    peaks.append(i)
            
            if peaks:
                detected_positions = self.linescan_positions[peaks]
            else:
                # Fallback to uniform distribution
                num_components = (self.total_params - 1) // 3
                detected_positions = np.linspace(self.linescan_positions[0] + position_range*0.1, 
                                               self.linescan_positions[-1] - position_range*0.1, 
                                               num_components)
        
        print(f"Detected {len(detected_positions)} potential edge positions")
        
        # Initialize exactly total_params parameters
        self.erf_parameters = []
        
        for i in range(self.total_params):
            if i == 0:
                # First parameter is baseline (k1)
                param_value = baseline
            else:
                # Determine parameter type based on position
                param_index = i - 1  # 0-based index for non-baseline parameters
                param_type = param_index % 3  # 0=amplitude, 1=width, 2=position
                component_index = param_index // 3  # Which ERF component this belongs to
                
                if param_type == 0:  # Amplitude (k2, k5, k8, ...)
                    # Use realistic amplitude based on data variation
                    param_value = intensity_range * (0.3 + 0.4 * rng.random_sample()) * ((-1) ** component_index)
                elif param_type == 1:  # Width (k3, k6, k9, ...)
                    # Use more realistic width values (0.5 to 2.0)
                    param_value = 0.5 + 1.5 * rng.random_sample()
                else:  # Position (k4, k7, k10, ...)
                    # Use detected positions if available, otherwise distribute evenly
                    if component_index < len(detected_positions):
                        param_value = detected_positions[component_index]
                    else:
                        # Fallback to uniform distribution
                        num_positions = (self.total_params - 1) // 3
                        param_value = self.linescan_positions[0] + (component_index + 1) * position_range / (num_positions + 1)
            
            self.erf_parameters.append(param_value)
        
        print(f"Initialized {self.total_params} parameters (k1 to k{self.total_params})")
        print(f"Estimated ERF components: {(self.total_params - 1) // 3}")
        
        # Show parameter ranges for verification
        amplitudes = [self.erf_parameters[i] for i in range(1, len(self.erf_parameters), 3)]
        widths = [self.erf_parameters[i] for i in range(2, len(self.erf_parameters), 3)]
        positions = [self.erf_parameters[i] for i in range(3, len(self.erf_parameters), 3)]
        
        if amplitudes:
            print(f"  Amplitude range: {min(amplitudes):.1f} to {max(amplitudes):.1f}")
        if widths:
            print(f"  Width range: {min(widths):.2f} to {max(widths):.2f}")
        if positions:
            print(f"  Position range: {min(positions):.1f} to {max(positions):.1f} nm")
    
    def show_parameter_window(self):
        """
        Show parameter configuration window with k1, k2, k3... format.
        Uses its own Tk root + mainloop so buttons respond immediately,
        independent of matplotlib's event loop.
        """
        # Validate erf_parameters before creating window
        if self.erf_parameters is None:
            print("ERROR: erf_parameters is None. Cannot open parameter window.")
            return
        
        if len(self.erf_parameters) != self.total_params:
            self.erf_parameters = [1.0 if i == 0 else 0.0 for i in range(self.total_params)]
            print("Re-initialized parameters to defaults.")
        
        print(f"Creating parameter window with {self.total_params} parameters...")
        
        # Create independent Tk root for this window (runs its own event loop)
        param_root = tk.Tk()
        param_root.title(f"ERF Parameter Configuration (k1 to k{self.total_params})")
        param_root.geometry("900x700")
        param_root.resizable(True, True)
        self.config_window = param_root
        
        # Create scrollable frame
        canvas = tk.Canvas(param_root)
        scrollbar = tk.Scrollbar(param_root, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Enable mouse wheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", on_mousewheel)
        
        # Parameter entries
        self.param_entries = []
        
        # Create header
        tk.Label(scrollable_frame, text=f"Parameter Configuration (k1 to k{self.total_params})", 
                font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=4, pady=10)
        
        # Column headers
        tk.Label(scrollable_frame, text="Parameter", font=("Arial", 10, "bold")).grid(row=1, column=0, padx=5, sticky="w")
        tk.Label(scrollable_frame, text="Value", font=("Arial", 10, "bold")).grid(row=1, column=1, padx=5, sticky="w")
        tk.Label(scrollable_frame, text="Parameter", font=("Arial", 10, "bold")).grid(row=1, column=2, padx=5, sticky="w")
        tk.Label(scrollable_frame, text="Value", font=("Arial", 10, "bold")).grid(row=1, column=3, padx=5, sticky="w")
        
        # Create parameter entries in two columns
        params_per_column = (self.total_params + 1) // 2
        
        for i in range(self.total_params):
            param_index = i + 1
            
            if i < params_per_column:
                col_offset = 0
                row = i + 2
            else:
                col_offset = 2
                row = (i - params_per_column) + 2
            
            tk.Label(scrollable_frame, text=f"k{param_index}:").grid(
                row=row, column=col_offset, padx=5, sticky="w")
            
            entry = tk.Entry(scrollable_frame, width=15)
            entry.insert(0, f"{self.erf_parameters[i]:.6f}")
            entry.grid(row=row, column=col_offset+1, padx=5, sticky="w")
            self.param_entries.append(entry)
        
        # Add description
        description_row = max(params_per_column + 3, self.total_params - params_per_column + 3)
        description_text = (
            f"Total parameters: {self.total_params}\n"
            f"k1: Baseline\n"
            f"k2, k5, k8, ...: Amplitudes\n"
            f"k3, k6, k9, ...: Widths\n"
            f"k4, k7, k10, ...: Positions (edge points)"
        )
        tk.Label(scrollable_frame, text=description_text, 
                font=("Arial", 9), justify="left", fg="gray").grid(
                row=description_row, column=0, columnspan=4, pady=10, sticky="w")
        
        # Buttons frame
        button_frame = tk.Frame(param_root)
        button_frame.pack(side="bottom", fill="x", padx=10, pady=10)
        
        tk.Button(button_frame, text="Apply", command=self.apply_parameter_config).pack(side="left", padx=5)
        tk.Button(button_frame, text="Reset", command=self.reset_parameters).pack(side="left", padx=5)
        
        if self.fitted_parameters is not None and len(self.fitted_parameters) > 0:
            tk.Button(button_frame, text="Load Fitted Parameters", 
                     command=self.load_fitted_parameters_to_config,
                     bg="lightgreen").pack(side="left", padx=5)
        
        tk.Button(button_frame, text="Save to File", command=self.save_parameters_to_file).pack(side="left", padx=5)
        tk.Button(button_frame, text="Load from File", command=self.load_parameters_from_file).pack(side="left", padx=5)
        
        def close_window():
            param_root.quit()
            param_root.destroy()
        
        tk.Button(button_frame, text="Close", command=close_window).pack(side="right", padx=5)
        param_root.protocol("WM_DELETE_WINDOW", close_window)
        
        # Pack scrollable elements
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Update scroll region once
        param_root.update_idletasks()
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.configure(scrollregion=canvas.bbox("all"))
        
        print(f"Parameter window opened with {len(self.param_entries)} entries.")
        
        # Run own event loop - window is fully interactive
        # matplotlib window pauses until this window is closed
        param_root.mainloop()
        
        self.config_window = None
        print("Parameter window closed.")
    
    def _on_config_window_close(self, root_to_destroy=None):
        """
        Handle parameter configuration window close event.
        Destroys the config window and optionally the temporary Tk root.
        """
        if hasattr(self, 'config_window') and self.config_window:
            try:
                self.config_window.destroy()
            except Exception:
                pass
            self.config_window = None
        
        if root_to_destroy is not None:
            try:
                root_to_destroy.destroy()
            except Exception:
                pass
    
    def load_fitted_parameters_to_config(self):
        """
        Load the fitted parameters from the last successful fit into the parameter configuration window.
        Follows the same pattern as load_parameters_from_file for consistency.
        """
        # Validate fitted parameters exist
        if not hasattr(self, 'fitted_parameters') or self.fitted_parameters is None:
            error_msg = "No fitted parameters available.\n\nPlease perform ERF fitting first:\n1. Extract linescan profile\n2. Configure parameters\n3. Click 'Fit ERF'\n4. Then try loading fitted parameters"
            messagebox.showwarning("No Fitted Parameters", error_msg)
            print("✗ Cannot load fitted parameters: no fitted parameters available")
            return
        
        if len(self.fitted_parameters) == 0:
            error_msg = "Fitted parameters array is empty.\n\nPlease perform ERF fitting first."
            messagebox.showwarning("No Fitted Parameters", error_msg)
            print("✗ Cannot load fitted parameters: empty array")
            return
        
        # Validate parameter window exists
        if not hasattr(self, 'param_entries') or not self.param_entries:
            error_msg = "Parameter configuration window not properly initialized."
            messagebox.showerror("Configuration Error", 
                               f"{error_msg}\n\nPlease close and reopen the parameter window.")
            print(f"✗ Cannot load fitted parameters: {error_msg}")
            return
        
        # Validate parameter count before operations
        if not hasattr(self, 'total_params') or self.total_params <= 0:
            error_msg = "Invalid parameter count configuration."
            messagebox.showerror("Configuration Error", 
                               f"{error_msg}\n\nPlease reconfigure parameters.")
            print(f"✗ Cannot load fitted parameters: {error_msg}")
            return
        
        try:
            print(f"Starting fitted parameter load operation...")
            print(f"  Fitted parameters: {len(self.fitted_parameters)}")
            print(f"  GUI entries: {len(self.param_entries)}")
            
            # Check parameter count mismatch
            if len(self.fitted_parameters) != len(self.param_entries):
                error_msg = (f"Parameter count mismatch:\n\n"
                           f"Fitted parameters: {len(self.fitted_parameters)}\n"
                           f"Current configuration: {len(self.param_entries)}\n\n"
                           f"This may occur if you changed the parameter count after fitting.\n\n"
                           f"Please reconfigure with {len(self.fitted_parameters)} parameters.")
                messagebox.showerror("Configuration Mismatch", error_msg)
                print(f"✗ Parameter count mismatch: fitted={len(self.fitted_parameters)}, entries={len(self.param_entries)}")
                return
            
            # Validate fitted parameters are numeric
            invalid_params = []
            for i, param in enumerate(self.fitted_parameters):
                if not isinstance(param, (int, float)) or not np.isfinite(param):
                    invalid_params.append(f"k{i+1}: {param}")
            
            if invalid_params:
                error_msg = f"Fitted parameters contain invalid values:\n\n" + "\n".join(invalid_params[:10])
                if len(invalid_params) > 10:
                    error_msg += f"\n... and {len(invalid_params) - 10} more"
                error_msg += "\n\nPlease perform ERF fitting again."
                messagebox.showerror("Invalid Fitted Parameters", error_msg)
                print(f"✗ Invalid fitted parameters detected:")
                for error in invalid_params:
                    print(f"  - {error}")
                return
            
            # Update parameter entries with fitted values - batch update for performance
            try:
                # Hide window during batch update
                if hasattr(self, 'config_window') and self.config_window:
                    self.config_window.withdraw()
                
                for i, entry in enumerate(self.param_entries):
                    if i < len(self.fitted_parameters):
                        entry.delete(0, tk.END)
                        entry.insert(0, f"{self.fitted_parameters[i]:.6f}")
                
                # Show window again
                if hasattr(self, 'config_window') and self.config_window:
                    self.config_window.deiconify()
                    self.config_window.update()
                
                print(f"✓ Updated {len(self.param_entries)} GUI entries with fitted values")
            except Exception as gui_error:
                if hasattr(self, 'config_window') and self.config_window:
                    self.config_window.deiconify()
                error_msg = f"Error updating GUI entries:\n{str(gui_error)}\n\nSome parameters may not have been loaded."
                messagebox.showerror("GUI Update Error", error_msg)
                print(f"✗ Error updating GUI: {gui_error}")
                return
            
            # Automatically apply the loaded parameters to update self.erf_parameters
            print(f"Applying fitted parameters...")
            apply_success = self.apply_parameter_config()
            
            if not apply_success:
                # Apply failed - show error
                messagebox.showwarning("Parameters Loaded with Issues", 
                                     f"⚠ Loaded fitted parameters into GUI,\n"
                                     "but automatic application failed.\n\n"
                                     "The parameters are displayed in the GUI.\n"
                                     "Please review them and click 'Apply' manually.\n\n"
                                     "Check the console for error details.")
                print(f"⚠ Fitted parameters loaded but apply failed")
                return
            
            # Brief success message - skip lengthy dialogs for speed
            print(f"✓ Fitted parameters loaded: {len(self.fitted_parameters)} values")
            
        except Exception as e:
            error_msg = f"Unexpected error loading fitted parameters:\n{str(e)}\n\nPlease check console for details."
            messagebox.showerror("Error Loading Parameters", error_msg)
            print(f"✗ Unexpected error loading fitted parameters: {e}")
            import traceback
            traceback.print_exc()

    def save_parameters_to_file(self):
        """
        Save current parameters to a file for later use with enhanced state synchronization.
        """
        # Validate parameter window exists
        if not hasattr(self, 'param_entries') or not self.param_entries:
            error_msg = "Parameter configuration window not properly initialized."
            messagebox.showerror("Configuration Error", 
                               f"{error_msg}\n\nPlease open the parameter configuration window first:\n"
                               "1. Click 'Set Parameters' button\n"
                               "2. Configure your parameters\n"
                               "3. Try saving again")
            print(f"✗ Cannot save parameters: {error_msg}")
            return
        
        # Validate parameter count before operations
        if not hasattr(self, 'total_params') or self.total_params <= 0:
            error_msg = "Invalid parameter count configuration."
            messagebox.showerror("Configuration Error", 
                               f"{error_msg}\n\nPlease reconfigure parameters.")
            print(f"✗ Cannot save parameters: {error_msg}")
            return
        
        try:
            print(f"Starting parameter save operation...")
            print(f"  Expected parameters: {self.total_params}")
            
            # Use verification method to check state synchronization before save
            if hasattr(self, 'erf_parameters') and self.erf_parameters is not None:
                if len(self.erf_parameters) == len(self.param_entries):
                    is_synced, mismatched_indices = self.verify_parameter_state_sync(auto_sync=False)
                    
                    if not is_synced:
                        # Ask user which values to save
                        response = messagebox.askyesnocancel(
                            "Parameter State Mismatch",
                            f"⚠ GUI and internal parameter states differ!\n\n"
                            f"Mismatched parameters: {len(mismatched_indices)}\n\n"
                            f"Which values would you like to save?\n\n"
                            f"• Click 'Yes' to save GUI values (what you see)\n"
                            f"• Click 'No' to sync GUI to internal state, then save\n"
                            f"• Click 'Cancel' to abort save operation\n\n"
                            f"Note: Internal state reflects the last 'Apply' or load operation."
                        )
                        
                        if response is None:  # Cancel
                            print("✗ Save cancelled due to state mismatch")
                            return
                        elif response is False:  # No - sync to internal state
                            print("Syncing GUI to internal state before save...")
                            self.verify_parameter_state_sync(auto_sync=True)
                            print("✓ GUI synchronized to internal state")
                        else:  # Yes - save GUI values
                            print("⚠ Saving GUI values (displayed values take precedence)")
                            print("  Note: These may differ from internal state used for fitting")
            
            # Get current parameters from the GUI with validation
            current_params = []
            invalid_entries = []
            
            for i, entry in enumerate(self.param_entries):
                param_name = f"k{i+1}"
                try:
                    value_str = entry.get().strip()
                    
                    if not value_str:
                        invalid_entries.append(f"{param_name}: empty value")
                        continue
                    
                    value = float(value_str)
                    
                    # Validate numeric value
                    if not np.isfinite(value):
                        invalid_entries.append(f"{param_name}: non-finite value")
                        continue
                    
                    current_params.append(value)
                    
                except ValueError as ve:
                    invalid_entries.append(f"{param_name}: invalid value '{entry.get()}'")
                except Exception as e:
                    invalid_entries.append(f"{param_name}: error reading value ({str(e)})")
            
            # Report validation errors
            if invalid_entries:
                error_msg = "Cannot save parameters with invalid values:\n\n" + "\n".join(invalid_entries[:10])
                if len(invalid_entries) > 10:
                    error_msg += f"\n... and {len(invalid_entries) - 10} more errors"
                error_msg += "\n\nPlease correct these values and try again."
                messagebox.showerror("Invalid Parameters", error_msg)
                print(f"✗ Parameter save failed - validation errors:")
                for error in invalid_entries:
                    print(f"  - {error}")
                return
            
            # Validate parameter count
            if len(current_params) != self.total_params:
                error_msg = f"Parameter count mismatch: expected {self.total_params}, got {len(current_params)}"
                messagebox.showerror("Configuration Error", 
                                   f"{error_msg}\n\nThis indicates a configuration problem.\n"
                                   "Please close and reopen the parameter window.")
                print(f"✗ {error_msg}")
                return
            
            print(f"✓ Parameter validation passed ({len(current_params)} parameters)")
            
            # Calculate parameter ranges for metadata
            try:
                amplitudes = [current_params[i] for i in range(1, len(current_params), 3) if i < len(current_params)]
                widths = [current_params[i] for i in range(2, len(current_params), 3) if i < len(current_params)]
                positions = [current_params[i] for i in range(3, len(current_params), 3) if i < len(current_params)]
            except Exception as calc_error:
                print(f"⚠ Warning: Could not calculate parameter ranges: {calc_error}")
                amplitudes, widths, positions = [], [], []
            
            # Ask user for filename
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_filename = f"vcsel_parameters_{timestamp}.txt"
            
            # Flush pending events before opening dialog
            if hasattr(self, 'config_window') and self.config_window:
                self.config_window.update()
            
            try:
                filename = filedialog.asksaveasfilename(
                    title="Save Parameters",
                    defaultextension=".txt",
                    filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                    initialfile=default_filename
                )
            except Exception as dialog_error:
                print(f"✗ Error opening file dialog: {dialog_error}")
                return
            
            if not filename:
                print("✗ Save cancelled by user")
                return
            
            # Write file with enhanced metadata and comprehensive error handling
            try:
                with open(filename, 'w') as f:
                    # Header with comprehensive metadata
                    f.write(f"# VCSEL ERF Parameters - Saved {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"# Total parameters: {len(current_params)}\n")
                    f.write(f"# ERF components: {(len(current_params) - 1) // 3}\n")
                    
                    # Include fitting error if available
                    if hasattr(self, 'final_loss') and self.final_loss is not None:
                        f.write(f"# Final fitting error: {self.final_loss:.2e}\n")
                    
                    # Parameter range summary in header
                    f.write(f"#\n")
                    f.write(f"# Parameter Range Summary:\n")
                    f.write(f"#   Baseline (k1): {current_params[0]:.6f}\n")
                    
                    if amplitudes:
                        f.write(f"#   Amplitudes (k2, k5, k8, ...): [{min(amplitudes):.2f}, {max(amplitudes):.2f}]\n")
                        f.write(f"#     Count: {len(amplitudes)}\n")
                    
                    if widths:
                        f.write(f"#   Widths (k3, k6, k9, ...): [{min(widths):.4f}, {max(widths):.4f}]\n")
                        f.write(f"#     Count: {len(widths)}\n")
                    
                    if positions:
                        f.write(f"#   Positions (k4, k7, k10, ...): [{min(positions):.2f}, {max(positions):.2f}] nm\n")
                        f.write(f"#     Count: {len(positions)}\n")
                        f.write(f"#     Span: {max(positions) - min(positions):.2f} nm\n")
                    
                    f.write(f"#\n")
                    f.write(f"# Format: k<N>: <value>\n")
                    f.write(f"# All values use consistent 12-digit precision\n")
                    f.write("\n")
                    
                    # Write parameters with consistent formatting
                    for i, param in enumerate(current_params):
                        f.write(f"k{i+1}: {param:.12f}\n")
                
            except PermissionError:
                error_msg = f"Permission denied writing to:\n{filename}\n\nPlease check file permissions or choose a different location."
                messagebox.showerror("Permission Error", error_msg)
                print(f"✗ Permission denied: {filename}")
                return
            except IOError as io_error:
                error_msg = f"I/O error writing to file:\n{filename}\n\nError: {str(io_error)}\n\nPlease check disk space and try again."
                messagebox.showerror("I/O Error", error_msg)
                print(f"✗ I/O error saving parameters: {io_error}")
                return
            except Exception as write_error:
                error_msg = f"Unexpected error writing to file:\n{filename}\n\nError: {str(write_error)}"
                messagebox.showerror("Write Error", error_msg)
                print(f"✗ Unexpected error saving parameters: {write_error}")
                return
            
            # Success feedback
            messagebox.showinfo("Parameters Saved", 
                              f"✓ Parameters saved successfully!\n\n"
                              f"File: {os.path.basename(filename)}\n"
                              f"Location: {os.path.dirname(filename)}\n\n"
                              f"Total parameters: {len(current_params)}\n"
                              f"ERF components: {(len(current_params) - 1) // 3}\n\n"
                              "You can now load these parameters in future sessions.")
            
            # Detailed console output
            print(f"✓ Parameters saved successfully to {filename}")
            print(f"  Total parameters: {len(current_params)} (k1 to k{len(current_params)})")
            print(f"  ERF components: {(len(current_params) - 1) // 3}")
            
            if amplitudes:
                print(f"  Amplitude range: [{min(amplitudes):.2f}, {max(amplitudes):.2f}]")
            if widths:
                print(f"  Width range: [{min(widths):.4f}, {max(widths):.4f}]")
            if positions:
                print(f"  Position range: [{min(positions):.2f}, {max(positions):.2f}] nm")
                print(f"  Position span: {max(positions) - min(positions):.2f} nm")
                
        except Exception as e:
            error_msg = f"Unexpected error during save operation:\n{str(e)}\n\nPlease check console for details."
            messagebox.showerror("Save Error", error_msg)
            print(f"✗ Unexpected error saving parameters: {e}")
            import traceback
            traceback.print_exc()

    def load_parameters_from_file(self):
        """
        Load parameters from a file and automatically apply them.
        """
        # Validate parameter window exists
        if not hasattr(self, 'param_entries') or not self.param_entries:
            error_msg = "Parameter configuration window not properly initialized."
            messagebox.showerror("Configuration Error", 
                               f"{error_msg}\n\nPlease open the parameter configuration window first:\n"
                               "1. Click 'Set Parameters' button\n"
                               "2. Then try loading parameters again")
            print(f"✗ Cannot load parameters: {error_msg}")
            return
        
        # Validate parameter count before operations
        if not hasattr(self, 'total_params') or self.total_params <= 0:
            error_msg = "Invalid parameter count configuration."
            messagebox.showerror("Configuration Error", 
                               f"{error_msg}\n\nPlease reconfigure parameters.")
            print(f"✗ Cannot load parameters: {error_msg}")
            return
        
        try:
            print(f"Loading parameters...")
            
            # Flush pending events before opening dialog
            if hasattr(self, 'config_window') and self.config_window:
                self.config_window.update()
            
            # Open file dialog
            try:
                filename = filedialog.askopenfilename(
                    title="Load Parameters",
                    filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
                )
            except Exception as dialog_error:
                print(f"✗ Error opening file dialog: {dialog_error}")
                return
            
            if not filename:
                print("✗ Load cancelled by user")
                return
            
            # Check if file exists
            if not os.path.exists(filename):
                error_msg = f"File not found:\n{filename}\n\nPlease check the file path and try again."
                messagebox.showerror("File Not Found", error_msg)
                print(f"✗ File not found: {filename}")
                return
            
            # Check if file is readable
            if not os.access(filename, os.R_OK):
                error_msg = f"Permission denied reading:\n{filename}\n\nPlease check file permissions."
                messagebox.showerror("Permission Error", error_msg)
                print(f"✗ Permission denied: {filename}")
                return
            
            # Read and parse file
            loaded_params = []
            parse_errors = []
            line_number = 0
            
            try:
                with open(filename, 'r') as f:
                    for line in f:
                        line_number += 1
                        line = line.strip()
                        
                        # Skip empty lines and comments
                        if not line or line.startswith('#'):
                            continue
                        
                        try:
                            if ':' in line:
                                # Format: k1: value
                                parts = line.split(':', 1)
                                if len(parts) == 2:
                                    value_str = parts[1].strip()
                                    value = float(value_str)
                                    
                                    # Validate numeric value
                                    if not np.isfinite(value):
                                        parse_errors.append(f"Line {line_number}: non-finite value")
                                        continue
                                    
                                    loaded_params.append(value)
                                else:
                                    parse_errors.append(f"Line {line_number}: invalid format")
                            else:
                                # Simple format: just the value
                                try:
                                    value = float(line)
                                    
                                    # Validate numeric value
                                    if not np.isfinite(value):
                                        parse_errors.append(f"Line {line_number}: non-finite value")
                                        continue
                                    
                                    loaded_params.append(value)
                                except ValueError:
                                    parse_errors.append(f"Line {line_number}: cannot parse '{line}'")
                                    continue
                        except ValueError as ve:
                            parse_errors.append(f"Line {line_number}: invalid number format")
                        except Exception as line_error:
                            parse_errors.append(f"Line {line_number}: {str(line_error)}")
                            
            except PermissionError:
                error_msg = f"Permission denied reading:\n{filename}\n\nPlease check file permissions."
                messagebox.showerror("Permission Error", error_msg)
                print(f"✗ Permission denied: {filename}")
                return
            except IOError as io_error:
                error_msg = f"I/O error reading file:\n{filename}\n\nError: {str(io_error)}"
                messagebox.showerror("I/O Error", error_msg)
                print(f"✗ I/O error loading parameters: {io_error}")
                return
            except Exception as read_error:
                error_msg = f"Unexpected error reading file:\n{filename}\n\nError: {str(read_error)}"
                messagebox.showerror("Read Error", error_msg)
                print(f"✗ Unexpected error reading file: {read_error}")
                return
            
            # Report parse errors if any
            if parse_errors:
                print(f"⚠ Warning: {len(parse_errors)} parse errors encountered:")
                for error in parse_errors[:5]:
                    print(f"  - {error}")
                if len(parse_errors) > 5:
                    print(f"  ... and {len(parse_errors) - 5} more errors")
            
            # Validate loaded parameters
            if len(loaded_params) == 0:
                error_msg = f"No valid parameters found in file:\n{filename}\n\n"
                if parse_errors:
                    error_msg += f"Parse errors: {len(parse_errors)}\n\n"
                error_msg += "Please check the file format:\n"
                error_msg += "- Each line should contain: k<N>: <value>\n"
                error_msg += "- Or just: <value>\n"
                error_msg += "- Comments start with #"
                messagebox.showerror("No Parameters Found", error_msg)
                print(f"✗ No valid parameters found in {filename}")
                return
            
            print(f"✓ Successfully parsed {len(loaded_params)} parameters from file")
            
            # Check parameter count mismatch
            if len(loaded_params) != len(self.param_entries):
                response = messagebox.askyesno(
                    "Parameter Count Mismatch",
                    f"File contains {len(loaded_params)} parameters,\n"
                    f"but current configuration has {len(self.param_entries)} parameters.\n\n"
                    f"What would you like to do?\n\n"
                    f"• Click 'Yes' to load {min(len(loaded_params), len(self.param_entries))} parameters\n"
                    f"• Click 'No' to cancel the operation\n\n"
                    f"Note: Remaining parameters will keep their current values."
                )
                if not response:
                    print("✗ Load cancelled due to parameter count mismatch")
                    return
                print(f"⚠ Parameter count mismatch: loading {min(len(loaded_params), len(self.param_entries))} parameters")
            
            # Load parameters into GUI - batch update with window hidden to prevent flicker
            num_loaded = min(len(loaded_params), len(self.param_entries))
            try:
                # Hide window during batch update for performance
                if hasattr(self, 'config_window') and self.config_window:
                    self.config_window.withdraw()
                
                for i, entry in enumerate(self.param_entries):
                    if i < len(loaded_params):
                        entry.delete(0, tk.END)
                        entry.insert(0, f"{loaded_params[i]:.6f}")
                
                # Show window again
                if hasattr(self, 'config_window') and self.config_window:
                    self.config_window.deiconify()
                    self.config_window.update()
                
                print(f"✓ Updated {num_loaded} GUI entries")
            except Exception as gui_error:
                # Make sure window is visible even if error occurs
                if hasattr(self, 'config_window') and self.config_window:
                    self.config_window.deiconify()
                error_msg = f"Error updating GUI entries:\n{str(gui_error)}\n\nSome parameters may not have been loaded."
                messagebox.showerror("GUI Update Error", error_msg)
                print(f"✗ Error updating GUI: {gui_error}")
                return
            
            # Automatically apply the loaded parameters to update self.erf_parameters
            print(f"Applying loaded parameters...")
            apply_success = self.apply_parameter_config()
            
            if apply_success:
                # Brief success message - no blocking dialog for speed
                print(f"✓ Parameters loaded: {num_loaded} values from {os.path.basename(filename)}")
            else:
                # Only show dialog on error
                print(f"⚠ Parameters loaded but apply failed")
                messagebox.showwarning("Apply Failed", 
                                     f"Loaded {num_loaded} parameters but apply failed.\n"
                                     "Please click 'Apply' manually.")
                
        except Exception as e:
            error_msg = f"Unexpected error during load operation:\n{str(e)}\n\nPlease check console for details."
            messagebox.showerror("Load Error", error_msg)
            print(f"✗ Unexpected error loading parameters: {e}")
            import traceback
            traceback.print_exc()

    def apply_parameter_config(self) -> bool:
        """
        Apply parameter configuration from GUI.
        
        Returns:
            bool: True if parameters were successfully applied, False otherwise
        """
        # Validate that parameter entries exist
        if not hasattr(self, 'param_entries') or not self.param_entries:
            error_msg = "Parameter configuration window not properly initialized."
            print(f"✗ {error_msg}")
            messagebox.showerror("Configuration Error", error_msg)
            return False
        
        try:
            new_params = []
            invalid_entries = []
            
            # Comprehensive validation for all parameter entries
            for i, entry in enumerate(self.param_entries):
                param_name = f"k{i+1}"
                try:
                    value_str = entry.get().strip()
                    
                    # Check if entry is empty
                    if not value_str:
                        invalid_entries.append(f"{param_name}: empty value")
                        continue
                    
                    # Try to convert to float
                    value = float(value_str)
                    
                    # Check for invalid numeric values
                    if not np.isfinite(value):
                        invalid_entries.append(f"{param_name}: non-finite value ({value_str})")
                        continue
                    
                    new_params.append(value)
                    
                except ValueError:
                    invalid_entries.append(f"{param_name}: invalid numeric value '{entry.get()}'")
            
            # If there are invalid entries, show specific error messages
            if invalid_entries:
                error_msg = "Invalid parameter values detected:\n" + "\n".join(invalid_entries[:10])
                if len(invalid_entries) > 10:
                    error_msg += f"\n... and {len(invalid_entries) - 10} more errors"
                print(f"✗ Parameter validation failed:")
                for error in invalid_entries:
                    print(f"  - {error}")
                messagebox.showerror("Invalid Parameters", error_msg)
                return False
            
            # Validate parameter count matches expected
            if len(new_params) != self.total_params:
                error_msg = f"Parameter count mismatch: expected {self.total_params}, got {len(new_params)}"
                print(f"✗ {error_msg}")
                messagebox.showerror("Configuration Error", error_msg)
                return False
            
            # Update internal parameter array
            self.erf_parameters = new_params
            
            # Brief console output (detailed output slows things down)
            print(f"✓ Parameters applied: {len(new_params)} values")
            
            return True
            
        except Exception as e:
            error_msg = f"Unexpected error applying parameters: {str(e)}"
            print(f"✗ {error_msg}")
            messagebox.showerror("Error", error_msg)
            return False
    
    def reset_parameters(self):
        """
        Reset parameters to default values.
        Optimized: batch GUI updates to prevent multiple redraws.
        """
        print("Resetting parameters...")
        
        # Re-initialize parameters
        self.initialize_parameters()
        
        if not hasattr(self, 'param_entries') or not self.param_entries:
            return
        
        # Batch update GUI entries - freeze window during update
        if hasattr(self, 'config_window') and self.config_window:
            self.config_window.withdraw()  # Hide window during update
        
        try:
            for i, entry in enumerate(self.param_entries):
                if i < len(self.erf_parameters):
                    entry.delete(0, tk.END)
                    entry.insert(0, f"{self.erf_parameters[i]:.6f}")
        finally:
            if hasattr(self, 'config_window') and self.config_window:
                self.config_window.deiconify()  # Show window again
                self.config_window.update()
        
        print(f"✓ Parameters reset ({len(self.erf_parameters)} values)")
    
    def verify_parameter_state_sync(self, auto_sync: bool = False) -> Tuple[bool, List[int]]:
        """
        Verify that GUI entries and internal parameter state are synchronized.
        
        This method checks if the values displayed in the GUI parameter entries
        match the internal self.erf_parameters array. This is important because:
        - GUI entries can be edited by the user without clicking "Apply"
        - Internal parameters can be updated by fitting operations
        - Mismatches can lead to confusion about which values will be used
        
        Args:
            auto_sync: If True, automatically sync GUI to internal state when mismatch found
        
        Returns:
            Tuple[bool, List[int]]: (is_synchronized, list_of_mismatched_indices)
                - is_synchronized: True if all values match, False if any mismatch
                - list_of_mismatched_indices: List of parameter indices (0-based) that don't match
        
        Requirements: 3.3, 3.4
        """
        # Validate prerequisites
        if not hasattr(self, 'param_entries') or not self.param_entries:
            print("⚠ Warning: Parameter entries not initialized")
            return False, []
        
        if not hasattr(self, 'erf_parameters') or self.erf_parameters is None:
            print("⚠ Warning: Internal parameters not initialized")
            return False, []
        
        if len(self.param_entries) != len(self.erf_parameters):
            print(f"⚠ Warning: Parameter count mismatch - GUI: {len(self.param_entries)}, Internal: {len(self.erf_parameters)}")
            return False, list(range(len(self.param_entries)))
        
        # Check each parameter for synchronization
        mismatched_indices = []
        tolerance_relative = 1e-6  # Relative tolerance for floating point comparison
        tolerance_absolute = 1e-10  # Absolute tolerance for values near zero
        
        for i, entry in enumerate(self.param_entries):
            try:
                # Get GUI value
                gui_value_str = entry.get().strip()
                if not gui_value_str:
                    mismatched_indices.append(i)
                    continue
                
                gui_value = float(gui_value_str)
                internal_value = self.erf_parameters[i]
                
                # Check for non-finite values
                if not np.isfinite(gui_value) or not np.isfinite(internal_value):
                    mismatched_indices.append(i)
                    continue
                
                # Calculate tolerance based on magnitude
                tolerance = max(abs(internal_value) * tolerance_relative, tolerance_absolute)
                
                # Check if values match within tolerance
                if abs(gui_value - internal_value) > tolerance:
                    mismatched_indices.append(i)
                    
            except ValueError:
                # GUI entry contains invalid value
                mismatched_indices.append(i)
            except Exception as e:
                print(f"⚠ Warning: Error checking parameter k{i+1}: {e}")
                mismatched_indices.append(i)
        
        is_synchronized = len(mismatched_indices) == 0
        
        # Report results
        if is_synchronized:
            print(f"✓ Parameter state verification: All {len(self.param_entries)} parameters synchronized")
        else:
            print(f"⚠ Parameter state verification: {len(mismatched_indices)} mismatches detected")
            
            # Show details for first few mismatches
            num_to_show = min(5, len(mismatched_indices))
            for idx in mismatched_indices[:num_to_show]:
                try:
                    gui_val = self.param_entries[idx].get()
                    internal_val = self.erf_parameters[idx]
                    print(f"  - k{idx+1}: GUI='{gui_val}', Internal={internal_val:.6f}")
                except Exception:
                    print(f"  - k{idx+1}: Error reading values")
            
            if len(mismatched_indices) > num_to_show:
                print(f"  ... and {len(mismatched_indices) - num_to_show} more mismatches")
            
            # Auto-sync if requested
            if auto_sync:
                print(f"Auto-syncing GUI to internal state...")
                try:
                    for idx in mismatched_indices:
                        self.param_entries[idx].delete(0, tk.END)
                        self.param_entries[idx].insert(0, f"{self.erf_parameters[idx]:.6f}")
                    print(f"✓ Synced {len(mismatched_indices)} parameters to internal state")
                    return True, []  # After sync, they're synchronized
                except Exception as sync_error:
                    print(f"✗ Error during auto-sync: {sync_error}")
                    return False, mismatched_indices
        
        return is_synchronized, mismatched_indices
    
    def perform_erf_fitting(self, event=None):
        """
        Perform high-precision ERF fitting using scipy.optimize.curve_fit.
        Model: k1 + k2*erf(k3*(x-k4)) + k5*erf(k6*(x-k7)) + ... + k95*erf(k96*(x-k97))
        
        Uses selected data interval if available, otherwise uses complete dataset.
        """
        if self.linescan_profile is None:
            print("Please extract a linescan profile first!")
            return
        
        if self.erf_parameters is None:
            print("Please configure ERF parameters first!")
            return
        
        # Determine which data to use for fitting
        use_selected_data = (self.selection_active and 
                           self.selected_positions is not None and 
                           self.selected_profile is not None and
                           getattr(self, 'use_selection_for_fitting', True))
        
        if use_selected_data:
            x_data = np.array(self.selected_positions, dtype=np.float64)
            y_data = np.array(self.selected_profile, dtype=np.float64)
            data_source = "selected interval"
            print("Using SELECTED DATA for ERF fitting")
            print(f"Selected range: {x_data[0]:.2f} to {x_data[-1]:.2f} nm")
        else:
            x_data = np.array(self.linescan_positions, dtype=np.float64)
            y_data = np.array(self.linescan_profile, dtype=np.float64)
            data_source = "complete dataset"
            if self.selection_active:
                print("Selection available but using COMPLETE DATASET for ERF fitting")
            else:
                print("Using COMPLETE DATASET for ERF fitting (no selection active)")
        
        print("Starting high-precision ERF fitting with scipy...")
        print(f"Using {self.total_params} parameters (k1 to k{self.total_params})")
        print(f"Target error tolerance: {ERF_CONFIG.get('target_error', 1e-10):.2e}")
        print(f"Model: k1 + k2*erf(k3*(x-k4)) + k5*erf(k6*(x-k7)) + ... + k{self.total_params-2}*erf(k{self.total_params-1}*(x-k{self.total_params}))")
        print(f"Data source: {data_source}")
        
        # Preserve original complete dataset for reference
        self._preserve_original_data()
        
        # Prepare data
        
        print(f"\nData preparation:")
        print(f"  Data points: {len(x_data)}")
        print(f"  X range: [{x_data.min():.2f}, {x_data.max():.2f}] nm")
        print(f"  Y range: [{y_data.min():.2f}, {y_data.max():.2f}]")
        
        # Create the ERF model function
        def erf_model(x, *params):
            """
            ERF model: k1 + k2*erf(k3*(x-k4)) + k5*erf(k6*(x-k7)) + ...
            Each ERF function represents one layer edge.
            """
            result = params[0]  # k1 (baseline)
            
            # Add ERF components: k2*erf(k3*(x-k4)), k5*erf(k6*(x-k7)), etc.
            num_erf_components = (len(params) - 1) // 3
            for i in range(num_erf_components):
                amplitude = params[3*i + 1]  # k2, k5, k8, ...
                width = params[3*i + 2]      # k3, k6, k9, ...
                position = params[3*i + 3]   # k4, k7, k10, ...
                
                result += amplitude * erf(width * (x - position))
            
            return result
        
        # Prepare initial parameters
        initial_params = np.array(self.erf_parameters, dtype=np.float64)
        
        print(f"\nInitial parameters:")
        print(f"  k1 (baseline): {initial_params[0]:.2f}")
        print(f"  Number of ERF components: {(len(initial_params) - 1) // 3}")
        
        # Show parameter ranges
        if len(initial_params) > 1:
            amplitudes = initial_params[1::3]  # k2, k5, k8, ...
            widths = initial_params[2::3]      # k3, k6, k9, ...
            positions = initial_params[3::3]   # k4, k7, k10, ...
            
            print(f"  Amplitude range: [{amplitudes.min():.1f}, {amplitudes.max():.1f}]")
            print(f"  Width range: [{widths.min():.3f}, {widths.max():.3f}]")
            print(f"  Position range: [{positions.min():.1f}, {positions.max():.1f}] nm")
        
        # Perform curve fitting with high precision
        print(f"\nStarting scipy curve_fit optimization...")
        print(f"  maxfev: {ERF_CONFIG.get('maxfev', 1000000)}")
        print(f"  ftol: {ERF_CONFIG.get('ftol', 1e-12):.2e}")
        print(f"  xtol: {ERF_CONFIG.get('xtol', 1e-12):.2e}")
        print(f"  gtol: {ERF_CONFIG.get('gtol', 1e-12):.2e}")
        
        try:
            # Delegate the curve_fit kernel to vcsel_analyzer.core.fitting.
            # Default path (bounded=False) is identical to the original call:
            # Levenberg-Marquardt with the ERF_CONFIG tolerances.
            fit_result = _fitting.fit_erf(x_data, y_data, initial_params, ERF_CONFIG)
            fitted_params = np.asarray(fit_result.params, dtype=np.float64)

            print("✓ Curve fitting completed successfully!")
            fit_elapsed = fit_result.elapsed_s
            print(f"  curve_fit elapsed: {fit_elapsed:.1f} s")

            # Final error metrics (computed inside fit_erf)
            mse = fit_result.mse
            rmse = fit_result.rmse
            max_error = fit_result.max_error
            
            print(f"\nFitting results:")
            print(f"  MSE: {mse:.2e}")
            print(f"  RMSE: {rmse:.2e}")
            print(f"  Max error: {max_error:.2e}")
            
            # Check if target error achieved
            target_error = ERF_CONFIG.get('target_error', 1e-10)
            if mse < target_error:
                print(f"✓ Target error achieved: {mse:.2e} < {target_error:.2e}")
            else:
                print(f"⚠ Target error not achieved: {mse:.2e} >= {target_error:.2e}")
            
            # Store results
            self.fitted_parameters = fitted_params.tolist()
            self.final_loss = mse
            
            # Store information about data source used for fitting
            self.fitting_used_selection = use_selected_data
            if use_selected_data:
                self.fitting_data_range = [float(x_data[0]), float(x_data[-1])]
            else:
                self.fitting_data_range = [float(self.linescan_positions[0]), float(self.linescan_positions[-1])]
            
            # Notify user about parameter saving option
            print(f"✓ Fitted parameters saved (Final error: {mse:.2e})")
            print(f"  Data used: {data_source}")
            if use_selected_data:
                print(f"  Selected range: {self.fitting_data_range[0]:.2f} to {self.fitting_data_range[1]:.2f} nm")
            print("  → Use 'Set Parameters' → 'Load Fitted Parameters' to use these for next scan")
            
            # Extract edge positions and layer thicknesses
            self.extract_fitting_results()
            
            # Show comprehensive results
            self.show_comprehensive_results()
            
        except Exception as e:
            print(f"✗ Curve fitting failed: {e}")
            if hasattr(self, 'config_window'):
                messagebox.showerror("Fitting Error", f"ERF fitting failed:\n{str(e)}")
            return
    

    
    def build_erf_model_numpy(self, x, parameters):
        """
        Build the ERF model from parameters using numpy.
        Model: k1 + k2*erf(k3*(x-k4)) + k5*erf(k6*(x-k7)) + ...

        Delegates to vcsel_analyzer.core.erf_model.build_erf_model.
        """
        return _erf_model.build_erf_model(x, parameters)
    
    def extract_fitting_results(self):
        """
        Extract edge positions and layer thicknesses from fitted parameters.
        """
        if self.fitted_parameters is None:
            return
        
        # Extract sorted edges and per-component descriptors (delegated to
        # vcsel_analyzer.core.thickness).
        self.edge_positions, _, _, _ = _thickness.edges_from_params(self.fitted_parameters)

        # Calculate layer thicknesses (delegated)
        thicknesses = _thickness.layer_thicknesses(self.edge_positions)

        # Assign to alternating materials (Material_A and Material_B) (delegated)
        self.layer_thicknesses, _ = _thickness.assign_materials(thicknesses, [])
        
        # Store comprehensive results including selection information
        self.fitting_results = {
            'fitted_parameters': self.fitted_parameters,
            'edge_positions': self.edge_positions,
            'layer_thicknesses': self.layer_thicknesses,
            'total_params': len(self.fitted_parameters),
            'total_layers': len(thicknesses),
            'final_loss': self.final_loss,
            'scan_range': [float(self.linescan_positions[0]), float(self.linescan_positions[-1])],
            'used_selection': getattr(self, 'fitting_used_selection', False),
            'fitting_data_range': getattr(self, 'fitting_data_range', [float(self.linescan_positions[0]), float(self.linescan_positions[-1])]),
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"\nFitting Results:")
        print(f"  Total parameters: {len(self.fitted_parameters)}")
        print(f"  Edge positions found: {len(self.edge_positions)}")
        print(f"  Layer thicknesses calculated: {len(thicknesses)}")
        print(f"  Material_A layers: {len(self.layer_thicknesses['Material_A'])}")
        print(f"  Material_B layers: {len(self.layer_thicknesses['Material_B'])}")
        print(f"  Final fitting error: {self.final_loss:.2e}")
        
        # Legend windows are available from interface buttons (input-driven)
    
    def _plot_fitting_results(self, ax, y_fit):
        """Draw scan profile, fitted ERF curve and edge points onto `ax`."""
        # Plot scan profile
        ax.plot(self.linescan_positions, self.linescan_profile, 'b.',
                label='Scan Profile', markersize=3, alpha=0.7)

        # Plot fitted curve
        ax.plot(self.linescan_positions, y_fit, 'g-',
                label='ERF Fitted Profile', linewidth=2)

        # Plot edge points (k4, k7, k10, ..., k97)
        if self.edge_positions:
            # Calculate y values for edge positions
            y_edge = self.build_erf_model_numpy(np.array(self.edge_positions), self.fitted_parameters)

            ax.plot(self.edge_positions, y_edge, 'ro',
                    label='Edge Points', markersize=8, markerfacecolor='red',
                    markeredgecolor='darkred', markeredgewidth=2)

            # Add edge point labels as a single, evenly-spaced row across the top
            # of the axes, each connected to its edge point by a thin leader line.
            # This keeps E1..E32 readable and clearly in left-to-right order even
            # when the edges are densely packed.
            n_edges = len(self.edge_positions)
            trans = ax.get_xaxis_transform()  # x: data coords, y: axes fraction
            x_min = float(np.min(self.linescan_positions))
            x_max = float(np.max(self.linescan_positions))
            x_span = (x_max - x_min) if (x_max > x_min) else 1.0
            for i, (x_pos, y_pos) in enumerate(zip(self.edge_positions, y_edge)):
                frac = (0.02 + 0.96 * i / (n_edges - 1)) if n_edges > 1 else 0.5
                label_x = x_min + frac * x_span
                ax.annotate(f'E{i+1}',
                            xy=(x_pos, y_pos), xycoords='data',
                            xytext=(label_x, 0.97), textcoords=trans,
                            fontsize=6, color='red', fontweight='bold',
                            ha='center', va='top', annotation_clip=False,
                            arrowprops=dict(arrowstyle='-', color='red',
                                            lw=0.5, alpha=0.4))

        ax.set_xlabel('Position [nm]')
        ax.set_ylabel('Intensity')
        # Create title with MSE and RMSE (RMSE = sqrt(MSE))
        title_base = f'VCSEL Cavity ERF Fitting Results\n'
        mse = self.final_loss
        if mse is not None and mse >= 0:
            rmse = np.sqrt(mse)
            title_params = f'Parameters: k1 to k{len(self.fitted_parameters)}, MSE: {mse:.2e} (RMSE: {rmse:.2e})'
        else:
            title_params = f'Parameters: k1 to k{len(self.fitted_parameters)}, MSE: N/A'

        # Show the fitted data range only when a sub-interval was selected
        if getattr(self, 'fitting_used_selection', False):
            data_range = getattr(self, 'fitting_data_range', [0, 0])
            title_selection = f'\nData: Selected Interval ({data_range[0]:.1f} - {data_range[1]:.1f} nm)'
        else:
            title_selection = ''

        ax.set_title(title_base + title_params + title_selection)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_residuals(self, ax, y_fit):
        """Draw fitting residuals onto `ax`."""
        residuals = self.linescan_profile - y_fit
        ax.plot(self.linescan_positions, residuals, 'g-', linewidth=1)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        ax.set_xlabel('Position [nm]')
        ax.set_ylabel('Residuals')
        ax.set_title('Fitting Residuals')
        ax.grid(True, alpha=0.3)

    def _plot_layer_thickness(self, ax):
        """Draw per-material layer thicknesses onto `ax`."""
        print("  [Plot] Entering Layer Thickness plot section...")

        for mat_key, fmt_str in [
            ('Material_A', 'bo-'),
            ('Material_B', 'go-')
        ]:
            mat_name = self.material_name_a if mat_key == 'Material_A' else self.material_name_b
            vals = []
            if (hasattr(self, 'layer_thicknesses') and
                isinstance(self.layer_thicknesses, dict)):
                vals = self.layer_thicknesses.get(mat_key, [])

            if not vals:
                print(f"  [Plot] {mat_key}: no thickness data, skipping")
                continue

            x_vals = list(range(1, len(vals) + 1))

            ax.plot(x_vals, vals, fmt_str, label=f'{mat_name} layers',
                    markersize=8, linewidth=2)

            # Annotate each data point with its thickness value
            text_color = 'blue' if mat_key == 'Material_A' else 'green'
            y_offset = 8 if mat_key == 'Material_A' else -14
            for j, x_pt in enumerate(x_vals):
                val = vals[j]
                ax.annotate(f'{val:.2f}', (x_pt, val),
                            xytext=(0, y_offset), textcoords='offset points',
                            fontsize=6, color=text_color, ha='center',
                            fontweight='bold')

            print(f"  [Plot] {mat_key}: plotted {len(vals)} layers")

        ax.set_xlabel('Layer Number')
        ax.set_ylabel('Thickness [nm]')
        ax.set_title('VCSEL Cavity Layer Thickness Measurements')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def show_comprehensive_results(self, event=None):
        """
        Show comprehensive results with scan profile, fitting curve, edge points, and layer thicknesses.
        Enhanced with comprehensive error handling.

        In addition to the combined overview figure, each panel (fitting
        results, residuals, layer thickness) is also shown in its own separate
        window.
        """
        try:
            # Validate required data
            if self.fitted_parameters is None:
                print("No fitting results available. Please perform ERF fitting first.")
                messagebox.showwarning("No Results", "No fitting results available. Please perform ERF fitting first.")
                return
            
            if self.linescan_positions is None or self.linescan_profile is None:
                print("No linescan data available.")
                messagebox.showerror("Data Error", "No linescan data available for display.")
                return
            
            print("Generating comprehensive results display...")
            
            # Generate fitted curve using numpy with error handling
            try:
                y_fit = self.build_erf_model_numpy(self.linescan_positions, self.fitted_parameters)
            except Exception as fit_error:
                print(f"✗ Error generating fitted curve: {fit_error}")
                messagebox.showerror("Fitting Error", f"Error generating fitted curve:\n{str(fit_error)}")
                return
        
            # Create comprehensive visualization with error handling
            try:
                fig = plt.figure(figsize=(20, 12))
            except Exception as fig_error:
                print(f"✗ Error creating figure: {fig_error}")
                messagebox.showerror("Plot Error", f"Error creating visualization:\n{str(fig_error)}")
                return
        
            # Plot 1: Scan profile and fitting curve with edge points
            try:
                # --- Combined overview figure (unchanged 3-subplot layout) ---
                ax1 = plt.subplot(2, 2, (1, 2))
                self._plot_fitting_results(ax1, y_fit)

                ax2 = plt.subplot(2, 2, 3)
                self._plot_residuals(ax2, y_fit)

                ax3 = plt.subplot(2, 2, 4)
                self._plot_layer_thickness(ax3)

                plt.tight_layout()

                # --- Additionally, each panel as its own standalone figure ---
                # Display only; use the matplotlib toolbar to save if desired.
                fig_results = plt.figure(figsize=(12, 7))
                self._plot_fitting_results(fig_results.add_subplot(111), y_fit)
                fig_results.tight_layout()

                fig_residuals = plt.figure(figsize=(12, 5))
                self._plot_residuals(fig_residuals.add_subplot(111), y_fit)
                fig_residuals.tight_layout()

                fig_thickness = plt.figure(figsize=(10, 7))
                self._plot_layer_thickness(fig_thickness.add_subplot(111))
                fig_thickness.tight_layout()

                # Show the combined figure and the three individual figures
                plt.show()

                print("✓ Comprehensive results visualization displayed successfully")
                print("  (combined overview + 3 individual figures)")
                
            except Exception as plot_error:
                print(f"✗ Error creating plots: {plot_error}")
                messagebox.showerror("Plotting Error", 
                                   f"Error creating visualization plots:\n{str(plot_error)}\n\n"
                                   "Will show text results only.")
            
            # Show detailed results window (always try this even if plots fail)
            try:
                self.show_detailed_results_window()
            except Exception as window_error:
                print(f"✗ Error showing detailed results window: {window_error}")
                messagebox.showerror("Window Error", 
                                   f"Error displaying results window:\n{str(window_error)}")
                
        except Exception as general_error:
            print(f"✗ General error in show_comprehensive_results: {general_error}")
            messagebox.showerror("Results Error", 
                               f"Error displaying results:\n{str(general_error)}\n\n"
                               "Check console for detailed error information.")
    
    def show_detailed_results_window(self):
        """
        Show detailed results in a separate window with comprehensive error handling.
        """
        try:
            results_window = tk.Toplevel()
            results_window.title("Detailed VCSEL Cavity Analysis Results")
            results_window.geometry("900x700")
            
            # Create text widget with scrollbar
            text_frame = tk.Frame(results_window)
            text_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            text_widget = tk.Text(text_frame, wrap="word", font=("Courier", 10))
            scrollbar = tk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)
            
            # Generate detailed results text with error handling
            try:
                results_text = self.generate_detailed_results_text()
                if not results_text or len(results_text.strip()) == 0:
                    results_text = self._generate_fallback_results_text()
                    
            except Exception as text_error:
                print(f"✗ Error generating detailed results text: {text_error}")
                results_text = self._generate_error_results_text(text_error)
            
            # Insert text with error handling
            try:
                text_widget.insert("1.0", results_text)
                text_widget.config(state="disabled")
            except Exception as insert_error:
                print(f"✗ Error inserting text into widget: {insert_error}")
                # Try with simplified text
                try:
                    simple_text = f"Error displaying results: {str(insert_error)}\n\nBasic Information:\n"
                    if hasattr(self, 'fitted_parameters') and self.fitted_parameters:
                        simple_text += f"Parameters fitted: {len(self.fitted_parameters)}\n"
                    if hasattr(self, 'final_loss') and self.final_loss:
                        simple_text += f"Final error: {self.final_loss:.2e}\n"
                    text_widget.insert("1.0", simple_text)
                    text_widget.config(state="disabled")
                except Exception:
                    text_widget.insert("1.0", "Critical error: Unable to display results")
                    text_widget.config(state="disabled")
            
            # Pack widgets with error handling
            try:
                text_widget.pack(side="left", fill="both", expand=True)
                scrollbar.pack(side="right", fill="y")
            except Exception as pack_error:
                print(f"✗ Error packing widgets: {pack_error}")
            
            # Add close button with error handling
            try:
                button_frame = tk.Frame(results_window)
                button_frame.pack(pady=5)
                
                tk.Button(button_frame, text="Close", 
                         command=results_window.destroy).pack(side="left", padx=5)
                
                # Add export button if results are available
                if hasattr(self, 'fitting_results') and self.fitting_results:
                    tk.Button(button_frame, text="Export Results", 
                             command=self.export_results).pack(side="left", padx=5)
                
                # Add retry button for text generation
                tk.Button(button_frame, text="Refresh", 
                         command=lambda: self._refresh_results_window(text_widget)).pack(side="left", padx=5)
                         
            except Exception as button_error:
                print(f"✗ Error creating buttons: {button_error}")
                # Minimal close button
                tk.Button(results_window, text="Close", 
                         command=results_window.destroy).pack(pady=5)
            
            print("✓ Results window displayed successfully")
            
        except Exception as window_error:
            print(f"✗ Critical error creating results window: {window_error}")
            # Show error dialog as fallback
            try:
                messagebox.showerror("Results Display Error", 
                                   f"Unable to display results window:\n{str(window_error)}\n\n"
                                   "Check console for detailed error information.")
            except Exception:
                print("✗ Unable to show error dialog - critical GUI failure")

    def _generate_fallback_results_text(self):
        """
        Generate a simplified fallback results text when main generation fails.
        """
        text = "="*60 + "\n"
        text += "VCSEL ANALYSIS RESULTS (Simplified View)\n"
        text += "="*60 + "\n\n"
        
        try:
            if hasattr(self, 'fitted_parameters') and self.fitted_parameters:
                text += f"Fitted Parameters: {len(self.fitted_parameters)} total\n"
                text += f"First few parameters: {self.fitted_parameters[:5]}\n\n"
            
            if hasattr(self, 'final_loss') and self.final_loss is not None:
                text += f"Final Fitting Error: {self.final_loss:.2e}\n\n"
            
            if hasattr(self, 'edge_positions') and self.edge_positions:
                text += f"Edge Positions Found: {len(self.edge_positions)}\n"
                text += f"Positions: {[f'{pos:.1f}' for pos in self.edge_positions[:10]]}\n\n"
            
            if hasattr(self, 'layer_thicknesses') and self.layer_thicknesses and isinstance(self.layer_thicknesses, dict):
                material_a = self.layer_thicknesses.get('Material_A', [])
                material_b = self.layer_thicknesses.get('Material_B', [])
                text += f"Layer Thicknesses:\n"
                text += f"  {self.material_name_a}: {len(material_a)} layers\n"
                text += f"  {self.material_name_b}: {len(material_b)} layers\n\n"
            
            text += "Note: This is a simplified view due to an error in detailed text generation.\n"
            text += "Try refreshing or check the console for error details.\n"
            
        except Exception as e:
            text += f"Error generating fallback text: {str(e)}\n"
        
        return text

    def _generate_error_results_text(self, error):
        """
        Generate error information text when results generation completely fails.
        """
        text = "="*60 + "\n"
        text += "ERROR DISPLAYING RESULTS\n"
        text += "="*60 + "\n\n"
        
        text += f"Error Details: {str(error)}\n\n"
        
        text += "Available Data Summary:\n"
        text += "-"*30 + "\n"
        
        try:
            text += f"Has fitted_parameters: {hasattr(self, 'fitted_parameters') and self.fitted_parameters is not None}\n"
            text += f"Has final_loss: {hasattr(self, 'final_loss') and self.final_loss is not None}\n"
            text += f"Has fitting_results: {hasattr(self, 'fitting_results') and bool(self.fitting_results)}\n"
            text += f"Has edge_positions: {hasattr(self, 'edge_positions') and bool(self.edge_positions)}\n"
            text += f"Has layer_thicknesses: {hasattr(self, 'layer_thicknesses') and bool(self.layer_thicknesses)}\n"
        except Exception as summary_error:
            text += f"Error generating data summary: {str(summary_error)}\n"
        
        text += "\nRecommendations:\n"
        text += "1. Check console output for detailed error information\n"
        text += "2. Try performing ERF fitting again\n"
        text += "3. Verify input data quality\n"
        text += "4. Use 'Refresh' button to retry text generation\n"
        
        return text

    def _refresh_results_window(self, text_widget):
        """
        Refresh the results window by regenerating the text.
        """
        try:
            text_widget.config(state="normal")
            text_widget.delete("1.0", tk.END)
            
            # Try to regenerate results text
            try:
                results_text = self.generate_detailed_results_text()
                if not results_text or len(results_text.strip()) == 0:
                    results_text = self._generate_fallback_results_text()
            except Exception as e:
                results_text = self._generate_error_results_text(e)
            
            text_widget.insert("1.0", results_text)
            text_widget.config(state="disabled")
            print("✓ Results window refreshed successfully")
            
        except Exception as refresh_error:
            print(f"✗ Error refreshing results window: {refresh_error}")
            try:
                text_widget.config(state="normal")
                text_widget.delete("1.0", tk.END)
                text_widget.insert("1.0", f"Refresh failed: {str(refresh_error)}")
                text_widget.config(state="disabled")
            except Exception:
                pass
    
    def generate_detailed_results_text(self) -> str:
        """
        Generate detailed results text.
        """
        # Enhanced type checking for fitting_results
        if not self.fitting_results or not isinstance(self.fitting_results, dict):
            return "No results available."
        
        text = "="*80 + "\n"
        text += "VCSEL CAVITY ERF FITTING RESULTS - DETAILED ANALYSIS\n"
        text += "="*80 + "\n\n"
        
        # Use defensive programming for all fitting_results access
        timestamp = self.fitting_results.get('timestamp', 'Unknown')
        
        # Validate total_params is numeric
        raw_total_params = self.fitting_results.get('total_params', len(self.fitting_results.get('fitted_parameters', [])))
        if isinstance(raw_total_params, (int, float)) and not np.isnan(raw_total_params):
            total_params = int(raw_total_params)
        else:
            fitted_params = self.fitting_results.get('fitted_parameters', [])
            total_params = len(fitted_params) if isinstance(fitted_params, list) else 0
        
        # Validate final_loss is numeric
        raw_final_loss = self.fitting_results.get('final_loss', 0.0)
        if isinstance(raw_final_loss, (int, float)) and not np.isnan(raw_final_loss) and not np.isinf(raw_final_loss):
            final_loss = raw_final_loss
        else:
            final_loss = 0.0
            
        scan_range = self.fitting_results.get('scan_range', [0.0, 100.0])
        # Validate scan_range is a list of numbers
        if not isinstance(scan_range, list) or len(scan_range) != 2:
            scan_range = [0.0, 100.0]
        else:
            # Ensure both values are numeric
            try:
                scan_range = [float(scan_range[0]), float(scan_range[1])]
            except (ValueError, TypeError):
                scan_range = [0.0, 100.0]
        
        text += f"Analysis Date: {timestamp}\n"
        text += f"Total Parameters: {total_params} (k1 to k{total_params})\n"
        text += f"Final Fitting Error: {final_loss:.2e}\n"
        text += f"Scan Range: {scan_range[0]:.2f} - {scan_range[1]:.2f} nm\n"
        text += f"Total Scan Length: {scan_range[1] - scan_range[0]:.2f} nm\n"
        
        # Add selection information
        used_selection = self.fitting_results.get('used_selection', False)
        fitting_data_range = self.fitting_results.get('fitting_data_range', scan_range)
        
        if used_selection:
            text += f"Data Source: Selected Interval\n"
            text += f"Fitting Range: {fitting_data_range[0]:.2f} - {fitting_data_range[1]:.2f} nm\n"
            text += f"Fitting Length: {fitting_data_range[1] - fitting_data_range[0]:.2f} nm\n"
        else:
            text += f"Data Source: Complete Dataset\n"
        
        text += "\n"
        
        # Edge positions (k4, k7, k10, ...)
        text += "EDGE POSITIONS (from position parameters):\n"
        text += "-"*50 + "\n"
        edge_positions = self.fitting_results.get('edge_positions', [])
        if edge_positions:
            edge_param_indices = [4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97]
            for i, pos in enumerate(edge_positions):
                param_num = edge_param_indices[i] if i < len(edge_param_indices) else f"k{4 + i*3}"
                text += f"Edge {i+1:2d} ({param_num}): {pos:8.2f} nm\n"
        else:
            text += "No edge positions available\n"
        text += "\n"
        
        # Layer thicknesses
        text += "LAYER THICKNESS ANALYSIS:\n"
        text += "-"*50 + "\n"
        
        layer_thicknesses = self.fitting_results.get('layer_thicknesses', {})
        # Ensure layer_thicknesses is a dict before calling .get()
        if not isinstance(layer_thicknesses, dict):
            layer_thicknesses = {}
        material_a_layers = layer_thicknesses.get('Material_A', [])
        material_b_layers = layer_thicknesses.get('Material_B', [])
        
        text += f"{self.material_name_a} Layers ({len(material_a_layers)} total):\n"
        if len(material_a_layers) > 0:
            # Validate layer thickness data
            valid_thicknesses = [t for t in material_a_layers if isinstance(t, (int, float)) and not np.isnan(t) and not np.isinf(t) and t > 0]
            
            if valid_thicknesses:
                for i, thickness in enumerate(valid_thicknesses):
                    text += f"  Layer {i+1:2d}: {thickness:6.2f} nm\n"
                
                # Statistical calculations with validation
                try:
                    mean_val = np.mean(valid_thicknesses)
                    std_val = np.std(valid_thicknesses)
                    min_val = np.min(valid_thicknesses)
                    max_val = np.max(valid_thicknesses)
                    
                    text += f"  Average:   {mean_val:6.2f} nm\n"
                    text += f"  Std Dev:   {std_val:6.2f} nm\n"
                    text += f"  Min:       {min_val:6.2f} nm\n"
                    text += f"  Max:       {max_val:6.2f} nm\n"
                    
                    # Add quality indicators
                    if std_val / mean_val < 0.1:  # CV < 10%
                        text += f"  Quality:   ✓ Uniform (CV: {100*std_val/mean_val:.1f}%)\n"
                    else:
                        text += f"  Quality:   ⚠ Variable (CV: {100*std_val/mean_val:.1f}%)\n"
                        
                except Exception as e:
                    text += f"  Statistics: Error calculating ({str(e)})\n"
            else:
                text += f"  No valid {self.material_name_a} layer thickness data\n"
        else:
            text += f"  No {self.material_name_a} layer data available\n"
        text += "\n"
        
        text += f"{self.material_name_b} Layers ({len(material_b_layers)} total):\n"
        if len(material_b_layers) > 0:
            # Validate layer thickness data
            valid_thicknesses = [t for t in material_b_layers if isinstance(t, (int, float)) and not np.isnan(t) and not np.isinf(t) and t > 0]
            
            if valid_thicknesses:
                for i, thickness in enumerate(valid_thicknesses):
                    text += f"  Layer {i+1:2d}: {thickness:6.2f} nm\n"
                
                # Statistical calculations with validation
                try:
                    mean_val = np.mean(valid_thicknesses)
                    std_val = np.std(valid_thicknesses)
                    min_val = np.min(valid_thicknesses)
                    max_val = np.max(valid_thicknesses)
                    
                    text += f"  Average:   {mean_val:6.2f} nm\n"
                    text += f"  Std Dev:   {std_val:6.2f} nm\n"
                    text += f"  Min:       {min_val:6.2f} nm\n"
                    text += f"  Max:       {max_val:6.2f} nm\n"
                    
                    # Add quality indicators
                    if std_val / mean_val < 0.1:  # CV < 10%
                        text += f"  Quality:   ✓ Uniform (CV: {100*std_val/mean_val:.1f}%)\n"
                    else:
                        text += f"  Quality:   ⚠ Variable (CV: {100*std_val/mean_val:.1f}%)\n"
                        
                except Exception as e:
                    text += f"  Statistics: Error calculating ({str(e)})\n"
            else:
                text += f"  No valid {self.material_name_b} layer thickness data\n"
        else:
            text += f"  No {self.material_name_b} layer data available\n"
        text += "\n"
        
        # Fitted parameters summary
        text += "FITTED PARAMETERS SUMMARY:\n"
        text += "-"*50 + "\n"
        fitted_parameters = self.fitting_results.get('fitted_parameters', [])
        
        # Validate fitted_parameters is a list
        if not isinstance(fitted_parameters, list):
            fitted_parameters = []
        
        if len(fitted_parameters) > 0:
            try:
                baseline = float(fitted_parameters[0])
                text += f"k1 (Baseline): {baseline:8.2f}\n\n"
            except (ValueError, TypeError):
                text += f"k1 (Baseline): Invalid value\n\n"
            
            # Show first few ERF components
            num_components = min(5, (len(fitted_parameters) - 1) // 3)
            for i in range(num_components):
                if 3*i + 3 < len(fitted_parameters):
                    try:
                        amp = float(fitted_parameters[3*i + 1])
                        width = float(fitted_parameters[3*i + 2])
                        center = float(fitted_parameters[3*i + 3])
                        
                        text += f"ERF Component {i+1}:\n"
                        text += f"  k{3*i + 2} (Amplitude): {amp:8.2f}\n"
                        text += f"  k{3*i + 3} (Width):     {width:8.2f}\n"
                        text += f"  k{3*i + 4} (Position):  {center:8.2f} nm\n\n"
                    except (ValueError, TypeError):
                        text += f"ERF Component {i+1}: Invalid parameter values\n\n"
            
            if num_components < (len(fitted_parameters) - 1) // 3:
                remaining = (len(fitted_parameters) - 1) // 3 - num_components
                text += f"... and {remaining} more ERF components\n\n"
        else:
            text += "No fitted parameters available\n\n"
        
        # Layer structure summary
        text += "LAYER STRUCTURE SUMMARY:\n"
        text += "-"*50 + "\n"
        
        # Calculate total structure information
        total_material_a = len([t for t in material_a_layers if isinstance(t, (int, float)) and not np.isnan(t) and not np.isinf(t) and t > 0])
        total_material_b = len([t for t in material_b_layers if isinstance(t, (int, float)) and not np.isnan(t) and not np.isinf(t) and t > 0])
        total_layers = total_material_a + total_material_b
        
        if total_layers > 0:
            text += f"Total Layers Detected: {total_layers}\n"
            text += f"  {self.material_name_a} layers: {total_material_a}\n"
            text += f"  {self.material_name_b} layers: {total_material_b}\n"
            
            # Calculate total thickness
            total_thickness = 0
            if material_a_layers:
                valid_material_a = [t for t in material_a_layers if isinstance(t, (int, float)) and not np.isnan(t) and not np.isinf(t) and t > 0]
                total_thickness += sum(valid_material_a)
            if material_b_layers:
                valid_material_b = [t for t in material_b_layers if isinstance(t, (int, float)) and not np.isnan(t) and not np.isinf(t) and t > 0]
                total_thickness += sum(valid_material_b)
            
            text += f"Total Structure Thickness: {total_thickness:.2f} nm\n"
            
            # Layer alternation check
            if total_material_a > 0 and total_material_b > 0:
                if abs(total_material_a - total_material_b) <= 1:
                    text += "Layer Alternation: ✓ Regular (expected for VCSEL)\n"
                else:
                    text += "Layer Alternation: ⚠ Irregular\n"
            
            # Average layer thickness
            if total_layers > 0:
                avg_layer_thickness = total_thickness / total_layers
                text += f"Average Layer Thickness: {avg_layer_thickness:.2f} nm\n"
                
                # Typical VCSEL layer thickness range check
                if 50 < avg_layer_thickness < 200:
                    text += "Thickness Range: ✓ Typical for VCSEL structures\n"
                elif avg_layer_thickness < 50:
                    text += "Thickness Range: ⚠ Thin layers (check resolution)\n"
                else:
                    text += "Thickness Range: ⚠ Thick layers (unusual for VCSEL)\n"
        else:
            text += "No valid layer structure detected\n"
        
        text += "\n"
        
        # Quality assessment
        text += "FITTING QUALITY ASSESSMENT:\n"
        text += "-"*50 + "\n"
        target_error = ERF_CONFIG.get('target_error', 1e-10)
        if final_loss < target_error:
            text += f"✓ EXCELLENT: Error {final_loss:.2e} < {target_error:.2e}\n"
        else:
            text += f"⚠ ACCEPTABLE: Error {final_loss:.2e} >= {target_error:.2e}\n"
        
        if self.linescan_positions is not None and len(self.linescan_positions) > 0:
            text += f"Total data points fitted: {len(self.linescan_positions)}\n"
            if len(fitted_parameters) > 0:
                text += f"Parameters per data point: {len(fitted_parameters) / len(self.linescan_positions):.3f}\n"
        else:
            text += "No linescan position data available\n"
        
        return text
    
    def set_material_names(self, event=None):
        """
        Show dialog to set custom material names for the main results legend.
        """
        try:
            # Visual style constants (8px spacing scale)
            BG = '#f5f6f8'
            font_title = ("Segoe UI", 15, "bold")
            font_label = ("Segoe UI", 11, "bold")
            font_entry = ("Segoe UI", 11)
            font_hint = ("Segoe UI", 9)

            # Create input window as its OWN Tk root running its own
            # mainloop() (see end of this method). matplotlib's default backend
            # is not forced to Tk, so there may be no Tk event loop running;
            # a bare Toplevel would render but never receive keyboard input.
            input_window = tk.Tk()
            input_window.title("Set Material Names")
            # No fixed initial size: the window is auto-sized to its content
            # (see adjust_window_size below). A small minsize keeps it sane.
            input_window.minsize(320, 280)
            input_window.resizable(True, True)
            input_window.configure(bg=BG)
            
            # Create main frame
            main_frame = tk.Frame(input_window, bg=BG)
            main_frame.pack(fill="both", expand=True, padx=24, pady=24)
            
            # Title
            title_label = tk.Label(main_frame, text="Set Material Names for Results Legend", 
                                 font=font_title, bg=BG, fg='#1f3a5f')
            title_label.pack(pady=(0, 24))
            
            # Material A input
            material_a_frame = tk.Frame(main_frame, bg=BG)
            material_a_frame.pack(fill="x", pady=(0, 16))
            
            tk.Label(material_a_frame, text="Material A Name:", font=font_label, 
                    bg=BG, fg='#333333').pack(anchor="w")
            material_a_entry = tk.Entry(material_a_frame, font=font_entry, width=30)
            material_a_entry.pack(fill="x", pady=(6, 0))
            material_a_entry.insert(0, self.material_name_a)
            
            # Material B input
            material_b_frame = tk.Frame(main_frame, bg=BG)
            material_b_frame.pack(fill="x", pady=(0, 16))
            
            tk.Label(material_b_frame, text="Material B Name:", font=font_label, 
                    bg=BG, fg='#333333').pack(anchor="w")
            material_b_entry = tk.Entry(material_b_frame, font=font_entry, width=30)
            material_b_entry.pack(fill="x", pady=(6, 0))
            material_b_entry.insert(0, self.material_name_b)
            
            # Divider between inputs and the instructions/actions area
            divider = tk.Frame(main_frame, bg='#e0e0e0', height=1)
            divider.pack(fill="x", pady=(8, 16))
            
            # Instructions
            instructions = tk.Label(main_frame, 
                                  text="These names will appear in the main results legend\nwhen you click 'Show Results'.", 
                                  font=font_hint, bg=BG, fg='#888888', justify='center')
            instructions.pack(pady=(0, 16))
            
            # Button frame
            button_frame = tk.Frame(main_frame, bg=BG)
            button_frame.pack(fill="x", pady=(8, 0))

            def close_window():
                """Exit this window's own event loop and destroy it."""
                input_window.quit()
                input_window.destroy()

            def apply_material_names():
                """Apply the new material names."""
                try:
                    new_name_a = material_a_entry.get().strip()
                    new_name_b = material_b_entry.get().strip()
                    
                    if not new_name_a:
                        messagebox.showwarning("Input Error", "Please enter a name for Material A.", parent=input_window)
                        return
                    if not new_name_b:
                        messagebox.showwarning("Input Error", "Please enter a name for Material B.", parent=input_window)
                        return
                    
                    # Update the material names
                    old_name_a = self.material_name_a
                    old_name_b = self.material_name_b
                    self.material_name_a = new_name_a
                    self.material_name_b = new_name_b
                    
                    # Show detailed confirmation with before/after
                    confirmation_msg = f"✓ Material Names Successfully Updated!\n\n"
                    confirmation_msg += f"Previous Names:\n"
                    confirmation_msg += f"  Material A: '{old_name_a}'\n"
                    confirmation_msg += f"  Material B: '{old_name_b}'\n\n"
                    confirmation_msg += f"New Names:\n"
                    confirmation_msg += f"  Material A: '{new_name_a}'\n"
                    confirmation_msg += f"  Material B: '{new_name_b}'\n\n"
                    confirmation_msg += f"These names will now appear in:\n"
                    confirmation_msg += f"• Main results legend (Show Results button)\n"
                    confirmation_msg += f"• Layer thickness displays\n"
                    confirmation_msg += f"• Plot legends and charts\n"
                    confirmation_msg += f"• Export file headers"
                    
                    messagebox.showinfo("Material Names Updated", confirmation_msg, parent=input_window)
                    print(f"✓ Material names updated: A='{old_name_a}' → '{new_name_a}', B='{old_name_b}' → '{new_name_b}'")
                    
                    # Update the button text to show current names
                    self._update_material_names_button_text()
                    
                    # Close the window (and exit its event loop) last
                    close_window()
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Error updating material names:\n{str(e)}", parent=input_window)

            # Action buttons (right-aligned: primary Apply on the far right)
            apply_button = tk.Button(button_frame, text="Apply", command=apply_material_names,
                     font=("Segoe UI", 10, "bold"), bg='#2d6cdf', fg='white',
                     activebackground='#255bbf', activeforeground='white',
                     relief='flat', bd=0, width=12, cursor='hand2')
            apply_button.pack(side="right", ipady=4)
            
            cancel_button = tk.Button(button_frame, text="Cancel", command=close_window,
                     font=("Segoe UI", 10, "bold"), bg='#e6e8eb', fg='#333333',
                     activebackground='#d8dadd', relief='flat', bd=0, width=12, cursor='hand2')
            cancel_button.pack(side="right", padx=(0, 8), ipady=4)
            
            # Adaptively size the window to its content (on open and while typing).
            # Width is driven directly by the pixel width of the longest text; the
            # title is allowed to wrap within the chosen width so it never forces
            # the window wider. Height is then recomputed from the content.
            from tkinter import font as tkfont
            entry_font_obj = tkfont.Font(family="Segoe UI", size=11)

            def adjust_window_size(event=None):
                try:
                    name_a = material_a_entry.get() or self.material_name_a
                    name_b = material_b_entry.get() or self.material_name_b
                    text_px = max(entry_font_obj.measure(name_a),
                                  entry_font_obj.measure(name_b),
                                  entry_font_obj.measure("Material A Name:"),
                                  200)

                    screen_w = input_window.winfo_screenwidth()
                    screen_h = input_window.winfo_screenheight()
                    # content pixels + margins (frame padx*2 + entry inner padding)
                    win_w = max(340, min(text_px + 88, int(screen_w * 0.9)))

                    # Let the title wrap within the chosen width instead of
                    # forcing the window wider.
                    title_label.config(wraplength=max(120, win_w - 60))

                    input_window.update_idletasks()
                    req_h = input_window.winfo_reqheight()
                    win_h = max(260, min(req_h, int(screen_h * 0.9)))
                    input_window.geometry(f"{win_w}x{win_h}")
                except Exception as _size_err:
                    print(f"Warning: could not auto-size material names window: {_size_err}")

            # Fit to initial content, then keep in sync as the user types
            adjust_window_size()
            material_a_entry.bind('<KeyRelease>', adjust_window_size)
            material_b_entry.bind('<KeyRelease>', adjust_window_size)

            # Add Enter key binding to apply changes
            def on_enter_key(event):
                apply_material_names()
            
            input_window.bind('<Return>', on_enter_key)
            material_a_entry.bind('<Return>', on_enter_key)
            material_b_entry.bind('<Return>', on_enter_key)
            
            # Bring the window to the front and focus the first entry.
            # This dialog uses its OWN tk.Tk() root + mainloop() (below), so it
            # is fully interactive regardless of matplotlib's active backend.
            input_window.lift()
            input_window.focus_set()
            material_a_entry.focus_set()
            
            print("✓ Material names input window displayed")
            
            # Run this window's own event loop so it receives input even when
            # matplotlib's backend is not Tk. matplotlib pauses until it closes.
            input_window.protocol("WM_DELETE_WINDOW", close_window)
            input_window.mainloop()
            
        except Exception as e:
            print(f"✗ Error creating material names input window: {e}")
            messagebox.showerror("Error", f"Error creating material names input window:\n{str(e)}")

    def _update_material_names_button_text(self):
        """Update the Set Material Names button text to show current names."""
        try:
            if hasattr(self, 'material_names_button'):
                current_text = f"Set Material Names\n(Current: {self.material_name_a[:10]}{'...' if len(self.material_name_a) > 10 else ''} | {self.material_name_b[:10]}{'...' if len(self.material_name_b) > 10 else ''})"
                self.material_names_button.label.set_text(current_text)
                # Force redraw of the button
                if hasattr(self, 'fig') and self.fig:
                    self.fig.canvas.draw_idle()
        except Exception as e:
            print(f"Warning: Could not update material names button text: {e}")

    def show_legend_input_window_1(self, event=None):
        """
        Show first legend input window for custom material/content.
        """
        self._show_legend_input_window(window_number=1, title="Legend Input Window 1", color_scheme="blue")
    
    def show_legend_input_window_2(self, event=None):
        """
        Show second legend input window for custom material/content.
        """
        self._show_legend_input_window(window_number=2, title="Legend Input Window 2", color_scheme="green")
    
    def _show_legend_input_window(self, window_number, title, color_scheme):
        """
        Show a generic legend input window that accepts custom material name and layer data.
        
        Args:
            window_number (int): Window identifier (1 or 2)
            title (str): Window title
            color_scheme (str): Color scheme ('blue' or 'green')
        """
        try:
            # Create input window
            input_window = tk.Toplevel()
            input_window.title(f"{title} - Material Input")
            input_window.minsize(460, 420)
            input_window.resizable(True, True)
            input_window.configure(bg='white')
            
            # Create main frame
            main_frame = tk.Frame(input_window, bg='white')
            main_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            # Title
            title_label = tk.Label(main_frame, text=f"{title} - Input", 
                                 font=("Arial", 16, "bold"), bg='white', 
                                 fg='blue' if color_scheme == 'blue' else 'green')
            title_label.pack(pady=(0, 20))
            
            # Material name input
            material_frame = tk.Frame(main_frame, bg='white')
            material_frame.pack(fill="x", pady=(0, 10))
            
            tk.Label(material_frame, text="Material Name:", font=("Arial", 12, "bold"), 
                    bg='white').pack(anchor="w")
            material_entry = tk.Entry(material_frame, font=("Arial", 11), width=40)
            material_entry.pack(fill="x", pady=(5, 0))
            material_entry.insert(0, "Enter material name")

            # Quantity and units input
            qty_units_frame = tk.Frame(main_frame, bg='white')
            qty_units_frame.pack(fill="x", pady=(0, 10))

            tk.Label(qty_units_frame, text="Quantity Name:", font=("Arial", 12, "bold"), bg='white').grid(row=0, column=0, sticky="w")
            quantity_entry = tk.Entry(qty_units_frame, font=("Arial", 11), width=24)
            quantity_entry.grid(row=0, column=1, padx=(8,0))
            quantity_entry.insert(0, "Layer Thickness")

            tk.Label(qty_units_frame, text="Units:", font=("Arial", 12, "bold"), bg='white').grid(row=0, column=2, padx=(16,0), sticky="w")
            units_entry = tk.Entry(qty_units_frame, font=("Arial", 11), width=10)
            units_entry.grid(row=0, column=3, padx=(8,0))
            units_entry.insert(0, "nm")

            # Output name input (for export filename/title)
            output_frame = tk.Frame(main_frame, bg='white')
            output_frame.pack(fill="x", pady=(0, 10))
            tk.Label(output_frame, text="Output Name (optional):", font=("Arial", 12, "bold"), bg='white').pack(anchor="w")
            output_name_entry = tk.Entry(output_frame, font=("Arial", 11), width=40)
            output_name_entry.pack(fill="x", pady=(5, 0))
            output_name_entry.insert(0, "")
            
            # Layer thickness input
            thickness_frame = tk.Frame(main_frame, bg='white')
            thickness_frame.pack(fill="both", expand=True, pady=(10, 0))
            
            tk.Label(thickness_frame, text="Layer Thicknesses (nm):", font=("Arial", 12, "bold"), 
                    bg='white').pack(anchor="w")
            tk.Label(thickness_frame, text="Enter one thickness per line:", font=("Arial", 10), 
                    bg='white', fg='gray').pack(anchor="w")
            
            # Text widget for thickness input
            text_frame = tk.Frame(thickness_frame, bg='white')
            text_frame.pack(fill="both", expand=True, pady=(5, 0))
            
            thickness_text = tk.Text(text_frame, wrap="word", font=("Courier", 11), 
                                   bg='#f8f8f8', relief='sunken', bd=2, height=8)
            thickness_scrollbar = tk.Scrollbar(text_frame, orient="vertical", command=thickness_text.yview)
            thickness_text.configure(yscrollcommand=thickness_scrollbar.set)
            
            # Pack text widgets
            thickness_text.pack(side="left", fill="both", expand=True)
            thickness_scrollbar.pack(side="right", fill="y")
            
            # Add sample data
            sample_data = "45.2\n38.7\n42.1\n39.8\n41.5\n40.3\n43.7\n37.9"
            thickness_text.insert("1.0", sample_data)
            
            # Button frame
            button_frame = tk.Frame(main_frame, bg='white')
            button_frame.pack(pady=(20, 0))

            def generate_legend():
                """Generate and show legend based on input data."""
                try:
                    material_name = material_entry.get().strip()
                    if not material_name or material_name == "Enter material name":
                        messagebox.showwarning("Input Error", "Please enter a valid material name.")
                        return
                    quantity_name = quantity_entry.get().strip() or "Layer Thickness"
                    units = units_entry.get().strip() or "nm"
                    output_name = output_name_entry.get().strip()
                    
                    thickness_data = thickness_text.get("1.0", "end-1c").strip()
                    if not thickness_data:
                        messagebox.showwarning("Input Error", "Please enter layer thickness data.")
                        return
                    
                    # Parse thickness data
                    try:
                        thicknesses = []
                        for line in thickness_data.split('\n'):
                            line = line.strip()
                            if line:
                                thicknesses.append(float(line))
                        
                        if not thicknesses:
                            messagebox.showwarning("Input Error", "No valid thickness values found.")
                            return
                            
                    except ValueError as e:
                        messagebox.showerror("Input Error", f"Invalid thickness data. Please enter numeric values only.\nError: {str(e)}")
                        return
                    
                    # Close input window and show legend
                    input_window.destroy()
                    self._show_custom_legend_window(material_name, thicknesses, window_number, color_scheme, quantity_name, units, output_name)
                    
                except Exception as e:
                    messagebox.showerror("Error", f"Error generating legend:\n{str(e)}")
            
            def clear_sample_data():
                """Clear the sample data from input fields."""
                material_entry.delete(0, tk.END)
                thickness_text.delete("1.0", tk.END)

            def quick_fill_from_results(material_key):
                """Populate inputs using current analyzer results for a given material."""
                try:
                    # Validate results presence
                    if not hasattr(self, 'layer_thicknesses') or self.layer_thicknesses is None:
                        messagebox.showwarning("No Results", "No layer thickness results available. Please run fitting first or paste values manually.")
                        return
                    if material_key not in self.layer_thicknesses:
                        messagebox.showwarning("Material Not Found", f"No results found for '{material_key}'. Available keys: {list(self.layer_thicknesses.keys())}")
                        return
                    values = self.layer_thicknesses.get(material_key, [])
                    # Filter valid numeric values
                    valid_vals = []
                    for v in values:
                        try:
                            fv = float(v)
                            if not np.isnan(fv) and not np.isinf(fv) and fv > 0:
                                valid_vals.append(fv)
                        except Exception:
                            continue
                    if not valid_vals:
                        messagebox.showwarning("Empty Results", f"No valid thickness values found for '{material_key}'.")
                        return
                    # Populate fields
                    material_entry.delete(0, tk.END)
                    material_entry.insert(0, material_key)
                    thickness_text.delete("1.0", tk.END)
                    thickness_text.insert("1.0", "\n".join(f"{v:.3f}" for v in valid_vals))
                    # Suggest output name if empty
                    if not output_name_entry.get().strip():
                        output_name_entry.insert(0, f"{material_key}_legend")
                    print(f"✓ Quick-filled legend input for {material_key} with {len(valid_vals)} values")
                except Exception as e:
                    messagebox.showerror("Quick-Fill Error", f"Error populating inputs from results:\n{str(e)}")
            
            # Buttons
            tk.Button(button_frame, text="Clear", command=clear_sample_data,
                     font=("Arial", 10), bg='lightgray').pack(side="left", padx=5)
            
            tk.Button(button_frame, text="Cancel", command=input_window.destroy,
                     font=("Arial", 10), bg='lightcoral').pack(side="left", padx=5)
            
            tk.Button(button_frame, text="Generate Legend", command=generate_legend,
                     font=("Arial", 10), bg='lightblue' if color_scheme == 'blue' else 'lightgreen').pack(side="left", padx=5)

            # Adaptively size the window to its content (on open and while typing).
            def adjust_window_size(event=None):
                try:
                    # Let the thickness text height follow the number of entered lines.
                    last_index = thickness_text.index('end-1c')
                    n_lines = int(last_index.split('.')[0])
                    thickness_text.config(height=max(6, min(n_lines, 24)))

                    input_window.update_idletasks()
                    req_w = input_window.winfo_reqwidth()
                    req_h = input_window.winfo_reqheight()
                    screen_w = input_window.winfo_screenwidth()
                    screen_h = input_window.winfo_screenheight()
                    win_w = max(460, min(req_w, int(screen_w * 0.9)))
                    win_h = max(420, min(req_h, int(screen_h * 0.9)))
                    input_window.geometry(f"{win_w}x{win_h}")
                except Exception as _size_err:
                    print(f"Warning: could not auto-size legend input window: {_size_err}")

            # Fit to initial content, then keep in sync as the user types
            adjust_window_size()
            material_entry.bind('<KeyRelease>', adjust_window_size)
            output_name_entry.bind('<KeyRelease>', adjust_window_size)
            thickness_text.bind('<KeyRelease>', adjust_window_size)

            print(f"✓ {title} input window displayed successfully")
            
        except Exception as e:
            print(f"✗ Error creating {title} input window: {e}")
            messagebox.showerror("Input Window Error", f"Error creating {title} input window:\n{str(e)}")
    
    def _show_custom_legend_window(self, material_name, layer_thicknesses, window_number, color_scheme, quantity_name="Layer Thickness", units="nm", output_name=""):
        """
        Show custom legend window with user-provided material and thickness data.
        
        Args:
            material_name (str): Name of the material
            layer_thicknesses (list): List of layer thickness values
            window_number (int): Window identifier (1 or 2)
            color_scheme (str): Color scheme ('blue' or 'green')
            quantity_name (str): Measurement/quantity name (e.g., Layer Thickness)
            units (str): Units for the values (e.g., nm)
            output_name (str): Optional name for export filename base
        """
        try:
            # Create legend window
            legend_window = tk.Toplevel()
            legend_window.title(f"{material_name} {quantity_name} Legend")
            legend_window.minsize(420, 320)
            legend_window.resizable(True, True)
            legend_window.configure(bg='white')
            
            # Create main frame
            main_frame = tk.Frame(legend_window, bg='white')
            main_frame.pack(fill="both", expand=True, padx=20, pady=20)
            
            # Title
            title_label = tk.Label(main_frame, text=f"{material_name} {quantity_name} Legend", 
                                 font=("Arial", 16, "bold"), bg='white', 
                                 fg='blue' if color_scheme == 'blue' else 'green')
            title_label.pack(pady=(0, 20))
            
            # Create scrollable text widget for legend content
            text_frame = tk.Frame(main_frame, bg='white')
            text_frame.pack(fill="both", expand=True)
            
            bg_color = '#f0f8ff' if color_scheme == 'blue' else '#f0fff0'
            fg_color = 'darkblue' if color_scheme == 'blue' else 'darkgreen'
            
            text_widget = tk.Text(text_frame, wrap="word", font=("Courier", 11), 
                                bg=bg_color, fg=fg_color, relief='sunken', bd=2)
            scrollbar = tk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)
            
            # Generate legend content
            legend_content = self._generate_custom_legend_content(material_name, layer_thicknesses, quantity_name, units)
            text_widget.insert("1.0", legend_content)
            text_widget.config(state="disabled")
            
            # Pack widgets
            text_widget.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            # Adaptively size the window to the generated legend content (one-time).
            try:
                content_lines = legend_content.split('\n')
                n_lines = len(content_lines)
                max_line_len = max((len(line) for line in content_lines), default=0)
                text_widget.config(width=max(48, min(max_line_len + 2, 110)),
                                   height=max(10, min(n_lines, 40)))
                legend_window.update_idletasks()
                req_w = legend_window.winfo_reqwidth()
                req_h = legend_window.winfo_reqheight()
                screen_w = legend_window.winfo_screenwidth()
                screen_h = legend_window.winfo_screenheight()
                win_w = max(420, min(req_w, int(screen_w * 0.9)))
                win_h = max(320, min(req_h, int(screen_h * 0.9)))
                legend_window.geometry(f"{win_w}x{win_h}")
            except Exception as _size_err:
                print(f"Warning: could not auto-size custom legend window: {_size_err}")

            # Add buttons
            button_frame = tk.Frame(main_frame, bg='white')
            button_frame.pack(pady=(20, 0))
            
            tk.Button(button_frame, text="Close", command=legend_window.destroy,
                     font=("Arial", 10), bg='lightgray').pack(side="left", padx=5)
            
            # Determine export base name
            export_base = output_name if output_name else f"{material_name}_{quantity_name}"
            tk.Button(button_frame, text="Export Legend", 
                     command=lambda: self._export_legend_content(export_base, legend_content),
                     font=("Arial", 10), bg='lightblue' if color_scheme == 'blue' else 'lightgreen').pack(side="left", padx=5)
            
            tk.Button(button_frame, text="Edit Input", 
                     command=lambda: [legend_window.destroy(), self._show_legend_input_window(window_number, f"Legend Input Window {window_number}", color_scheme)],
                     font=("Arial", 10), bg='lightyellow').pack(side="left", padx=5)
            
            print(f"✓ {material_name} legend window displayed successfully")
            
        except Exception as e:
            print(f"✗ Error creating {material_name} legend window: {e}")
            messagebox.showerror("Legend Window Error", f"Error creating {material_name} legend window:\n{str(e)}")
    
    def _export_legend_content(self, output_name, content):
        """
        Export legend content to a text file.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = (output_name or "legend").strip()
            # Sanitize filename base
            safe_base = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in base)
            if len(safe_base) == 0:
                safe_base = "legend"
            filename = f"{safe_base}_{timestamp}.txt"
            
            with open(filename, 'w') as f:
                f.write(content)
            
            print(f"✓ Legend exported to {filename}")
            messagebox.showinfo("Export Success", f"Legend exported to:\n{filename}")
            
        except Exception as e:
            print(f"✗ Error exporting legend: {e}")
            messagebox.showerror("Export Error", f"Error exporting legend:\n{str(e)}")

    def _generate_custom_legend_content(self, material_name, layer_thicknesses, quantity_name="Layer Thickness", units="nm"):
        """
        Generate legend content for any material using provided values.
        
        Args:
            material_name (str): Name of the material to display
            layer_thicknesses (list[float]): Measurement values (e.g., thicknesses)
            quantity_name (str): Measurement/quantity name to display
            units (str): Units for the values
        Returns:
            str: Formatted legend content
        """
        content = "="*60 + "\n"
        content += f"{material_name.upper()} {quantity_name.upper()} LEGEND\n"
        content += "="*60 + "\n\n"

        # Basic info
        try:
            total_layers = len(layer_thicknesses) if layer_thicknesses is not None else 0
        except Exception:
            total_layers = 0

        content += f"Material: {material_name}\n"
        content += f"Quantity: {quantity_name} ({units})\n"
        content += f"Total Layers: {total_layers}\n"
        content += f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Details section
        content += "LAYER DETAILS:\n"
        content += "-"*40 + "\n"

        # Validate and filter thicknesses
        valid_thicknesses = []
        try:
            for t in (layer_thicknesses or []):
                if isinstance(t, (int, float)) and not np.isnan(t) and not np.isinf(t) and t > 0:
                    valid_thicknesses.append(float(t))
        except Exception:
            pass

        if valid_thicknesses:
            for i, thickness in enumerate(valid_thicknesses):
                content += f"Layer {i+1:2d}: {thickness:8.3f} {units}\n"

            content += "\n" + "-"*40 + "\n"
            content += "STATISTICAL SUMMARY:\n"
            content += "-"*40 + "\n"

            mean_val = float(np.mean(valid_thicknesses))
            std_val = float(np.std(valid_thicknesses))
            min_val = float(np.min(valid_thicknesses))
            max_val = float(np.max(valid_thicknesses))

            content += f"Average {quantity_name}: {mean_val:8.3f} {units}\n"
            content += f"Standard Deviation: {std_val:8.3f} {units}\n"
            content += f"Minimum {quantity_name}: {min_val:8.3f} {units}\n"
            content += f"Maximum {quantity_name}: {max_val:8.3f} {units}\n"
            content += f"{quantity_name} Range: {max_val - min_val:8.3f} {units}\n"

            cv = (std_val / mean_val) * 100 if mean_val > 0 else 0.0
            content += f"Coefficient of Variation: {cv:6.2f}%\n\n"

            content += "QUALITY ASSESSMENT:\n"
            content += "-"*40 + "\n"
            if cv < 5:
                content += "✓ EXCELLENT: Very uniform layer thickness\n"
            elif cv < 10:
                content += "✓ GOOD: Reasonably uniform layer thickness\n"
            elif cv < 20:
                content += "⚠ MODERATE: Some thickness variation\n"
            else:
                content += "⚠ POOR: High thickness variation\n"

            content += f"\nTotal {material_name} {quantity_name}: {sum(valid_thicknesses):.3f} {units}\n"
        else:
            content += "No valid thickness data available\n"

        content += "\n" + "="*60 + "\n"
        content += "Legend generated by VCSEL ERF Analyzer\n"
        content += "="*60

        return content
    
    def extract_selected_data(self):
        """
        Extract data subset from selection indices with comprehensive validation.
        
        Task 4.1: Write extract_selected_data method to get data subset from selection indices
        - Implement validation of extracted data (check for NaN, sufficient points)
        - Add method to calculate selected interval statistics (length, point count)
        
        Returns:
            dict: Dictionary containing extracted data and statistics, or None if extraction fails
        """
        # Validate selection state
        if not self.selection_active:
            print("No active selection available for data extraction")
            return None
        
        if (self.selected_start_index is None or 
            self.selected_end_index is None or
            self.linescan_positions is None or
            self.linescan_profile is None):
            print("Cannot extract selected data: missing selection or data")
            return None
        
        try:
            # Ensure proper ordering of indices
            start_idx = min(self.selected_start_index, self.selected_end_index)
            end_idx = max(self.selected_start_index, self.selected_end_index)
            
            # Validate indices are within bounds
            max_index = len(self.linescan_positions) - 1
            if start_idx < 0 or end_idx > max_index:
                print(f"Cannot extract data: indices [{start_idx}, {end_idx}] out of bounds [0, {max_index}]")
                return None
            
            # Extract selected positions and intensities (inclusive of end point)
            selected_positions = self.linescan_positions[start_idx:end_idx+1].copy()
            selected_profile = self.linescan_profile[start_idx:end_idx+1].copy()
            
            # Task 4.1: Implement validation of extracted data (check for NaN, sufficient points)
            if len(selected_positions) == 0 or len(selected_profile) == 0:
                print("Error: extracted data is empty")
                return None
            
            if len(selected_positions) != len(selected_profile):
                print("Error: extracted position and profile arrays have different lengths")
                return None
            
            # Check for NaN or infinite values
            if (np.any(~np.isfinite(selected_positions)) or 
                np.any(~np.isfinite(selected_profile))):
                print("Warning: extracted data contains invalid values (NaN or infinite)")
                # Remove invalid data points
                valid_mask = np.isfinite(selected_positions) & np.isfinite(selected_profile)
                selected_positions = selected_positions[valid_mask]
                selected_profile = selected_profile[valid_mask]
                
                if len(selected_positions) == 0:
                    print("Error: no valid data points after removing NaN/infinite values")
                    return None
            
            # Check for sufficient points (minimum 3 for meaningful analysis)
            min_required_points = 3
            if len(selected_positions) < min_required_points:
                print(f"Error: insufficient data points ({len(selected_positions)} < {min_required_points})")
                return None
            
            # Task 4.1: Add method to calculate selected interval statistics (length, point count)
            selection_length = selected_positions[-1] - selected_positions[0]
            point_count = len(selected_positions)
            intensity_range = np.max(selected_profile) - np.min(selected_profile)
            intensity_mean = np.mean(selected_profile)
            intensity_std = np.std(selected_profile)
            
            # Calculate data quality metrics
            data_density = point_count / selection_length if selection_length > 0 else 0
            signal_to_noise = intensity_mean / intensity_std if intensity_std > 0 else float('inf')
            
            # Create comprehensive statistics dictionary
            statistics = {
                'start_position': float(selected_positions[0]),
                'end_position': float(selected_positions[-1]),
                'selection_length': float(selection_length),
                'point_count': int(point_count),
                'data_density': float(data_density),  # points per nm
                'intensity_range': float(intensity_range),
                'intensity_mean': float(intensity_mean),
                'intensity_std': float(intensity_std),
                'intensity_min': float(np.min(selected_profile)),
                'intensity_max': float(np.max(selected_profile)),
                'signal_to_noise': float(signal_to_noise),
                'start_index': int(start_idx),
                'end_index': int(end_idx)
            }
            
            # Create result dictionary
            result = {
                'positions': selected_positions,
                'intensities': selected_profile,
                'statistics': statistics,
                'extraction_timestamp': datetime.now().isoformat(),
                'is_valid': True
            }
            
            # Update instance variables for compatibility
            self.selected_positions = selected_positions
            self.selected_profile = selected_profile
            
            print(f"✓ Selected data extracted successfully:")
            print(f"  Position range: {statistics['start_position']:.2f} to {statistics['end_position']:.2f} nm")
            print(f"  Selection length: {statistics['selection_length']:.2f} nm")
            print(f"  Data points: {statistics['point_count']}")
            print(f"  Data density: {statistics['data_density']:.1f} points/nm")
            print(f"  Intensity range: {statistics['intensity_min']:.1f} to {statistics['intensity_max']:.1f}")
            print(f"  Signal-to-noise ratio: {statistics['signal_to_noise']:.2f}")
            
            return result
            
        except Exception as e:
            print(f"Error extracting selected data: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def plot_selected_interval(self):
        """
        Create separate plot window for selected interval with proper labels and annotations.
        
        Task 4.2: Implement plot_selected_interval method to display selected data
        - Add proper axis labels and title indicating selected region
        - Include interval boundaries and statistics in plot annotations
        
        Returns:
            bool: True if plot was created successfully, False otherwise
        """
        # Extract selected data if not already done
        if self.selected_positions is None or self.selected_profile is None:
            extracted_data = self.extract_selected_data()
            if extracted_data is None:
                print("Cannot plot selected interval: data extraction failed")
                return False
        
        try:
            # Task 4.2: Create separate plot window for selected interval
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot selected data with enhanced styling
            ax.plot(self.selected_positions, self.selected_profile, 'b.-', 
                   linewidth=2, markersize=4, alpha=0.8, 
                   label='Selected Data Interval')

            # Task 4.2: Add proper axis labels and title indicating selected region
            ax.set_xlabel('Position [nm]', fontsize=12, fontweight='bold')
            ax.set_ylabel('Intensity', fontsize=12, fontweight='bold')
            
            # Calculate statistics for title
            start_pos = self.selected_positions[0]
            end_pos = self.selected_positions[-1]
            length = end_pos - start_pos
            point_count = len(self.selected_positions)
            
            title = (f'Selected Data Interval\n'
                    f'Range: {start_pos:.2f} - {end_pos:.2f} nm '
                    f'(Length: {length:.2f} nm, Points: {point_count})')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            # Task 4.2: Include interval boundaries and statistics in plot annotations
            
            # Add vertical lines at boundaries
            ax.axvline(x=start_pos, color='red', linestyle='--', linewidth=2, 
                      alpha=0.7, label=f'Start: {start_pos:.2f} nm')
            ax.axvline(x=end_pos, color='red', linestyle='--', linewidth=2, 
                      alpha=0.7, label=f'End: {end_pos:.2f} nm')
            
            # Calculate additional statistics
            intensity_min = np.min(self.selected_profile)
            intensity_max = np.max(self.selected_profile)
            intensity_mean = np.mean(self.selected_profile)
            intensity_std = np.std(self.selected_profile)
            data_density = point_count / length if length > 0 else 0
            
            # Create statistics annotation box
            stats_text = (
                f'Selection Statistics:\n'
                f'• Start Position: {start_pos:.2f} nm\n'
                f'• End Position: {end_pos:.2f} nm\n'
                f'• Interval Length: {length:.2f} nm\n'
                f'• Data Points: {point_count}\n'
                f'• Data Density: {data_density:.1f} pts/nm\n'
                f'• Intensity Range: {intensity_min:.1f} - {intensity_max:.1f}\n'
                f'• Mean Intensity: {intensity_mean:.1f} ± {intensity_std:.1f}'
            )
            
            # Position statistics box in upper right corner
            ax.text(0.98, 0.98, stats_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.5', 
                            facecolor='lightblue', 
                            edgecolor='navy',
                            alpha=0.9),
                   fontsize=10,
                   fontfamily='monospace')
            
            # Add selection info annotation in upper left
            selection_info = (
                f'Selected Region\n'
                f'From full dataset:\n'
                f'{len(self.linescan_positions)} total points'
            )
            
            ax.text(0.02, 0.98, selection_info,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   horizontalalignment='left',
                   bbox=dict(boxstyle='round,pad=0.3', 
                            facecolor='lightyellow', 
                            edgecolor='orange',
                            alpha=0.8),
                   fontsize=9,
                   fontweight='bold')
            
            # Enhance plot appearance
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
            
            # Set axis limits with some padding
            x_padding = length * 0.05  # 5% padding
            y_range = intensity_max - intensity_min
            y_padding = y_range * 0.1  # 10% padding
            
            ax.set_xlim(start_pos - x_padding, end_pos + x_padding)
            ax.set_ylim(intensity_min - y_padding, intensity_max + y_padding)
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fig.text(0.99, 0.01, f'Generated: {timestamp}', 
                    ha='right', va='bottom', fontsize=8, alpha=0.7)
            
            plt.tight_layout()
            plt.show()
            
            print(f"✓ Selected interval plot displayed successfully")
            print(f"  Plotted {point_count} data points from {start_pos:.2f} to {end_pos:.2f} nm")
            
            return True
            
        except Exception as e:
            print(f"Error creating selected interval plot: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _preserve_original_data(self):
        """
        Preserve original complete dataset for reference during fitting operations.
        
        This method ensures that the original linescan data is always available
        even when fitting is performed on selected intervals.
        """
        if not hasattr(self, 'original_linescan_positions'):
            # Store original data only if not already preserved
            if self.linescan_positions is not None:
                self.original_linescan_positions = self.linescan_positions.copy()
                print("✓ Original position data preserved for reference")
            else:
                self.original_linescan_positions = None
                
        if not hasattr(self, 'original_linescan_profile'):
            # Store original profile data only if not already preserved
            if self.linescan_profile is not None:
                self.original_linescan_profile = self.linescan_profile.copy()
                print("✓ Original profile data preserved for reference")
            else:
                self.original_linescan_profile = None

    def toggle_selection_for_fitting(self, use_selection=None):
        """
        Toggle option to enable/disable using selected data for fitting.
        
        Args:
            use_selection (bool, optional): If provided, sets the selection usage state.
                                          If None, toggles the current state.
        
        Returns:
            bool: Current state of selection usage for fitting
        """
        if use_selection is None:
            # Toggle current state
            current_state = getattr(self, 'use_selection_for_fitting', True)
            self.use_selection_for_fitting = not current_state
        else:
            # Set specific state
            self.use_selection_for_fitting = bool(use_selection)
        
        state_text = "enabled" if self.use_selection_for_fitting else "disabled"
        print(f"Selection usage for fitting: {state_text}")
        
        if self.selection_active:
            if self.use_selection_for_fitting:
                print("  → Next ERF fitting will use selected data interval")
            else:
                print("  → Next ERF fitting will use complete dataset (selection ignored)")
        else:
            print("  → No active selection - complete dataset will be used regardless")
        
        return self.use_selection_for_fitting

    def revert_to_full_dataset(self):
        """
        Revert to using the complete dataset for fitting operations.
        
        This method disables selection usage and ensures that subsequent
        fitting operations will use the full linescan data.
        
        Returns:
            bool: True if successfully reverted, False if no data available
        """
        if self.linescan_positions is None or self.linescan_profile is None:
            print("No dataset available to revert to")
            return False
        
        # Disable selection usage for fitting
        self.use_selection_for_fitting = False
        
        print("✓ Reverted to using complete dataset for fitting")
        print(f"  Full dataset range: {self.linescan_positions[0]:.2f} to {self.linescan_positions[-1]:.2f} nm")
        print(f"  Total data points: {len(self.linescan_positions)}")
        
        if self.selection_active:
            selected_range = f"{self.selection_start_pos:.2f} to {self.selection_end_pos:.2f} nm"
            print(f"  Selection remains active ({selected_range}) but will be ignored for fitting")
        
        return True

    def get_fitting_data_info(self):
        """
        Get information about which data will be used for the next fitting operation.
        
        Returns:
            dict: Information about the data that will be used for fitting
        """
        info = {
            'will_use_selection': False,
            'data_source': 'complete_dataset',
            'data_range': None,
            'data_points': 0,
            'selection_available': False,
            'selection_enabled': getattr(self, 'use_selection_for_fitting', True)
        }
        
        # Check if we have any data
        if self.linescan_positions is None or self.linescan_profile is None:
            info['data_source'] = 'no_data'
            return info
        
        # Check selection availability and usage
        selection_available = (self.selection_active and 
                             self.selected_positions is not None and 
                             self.selected_profile is not None)
        
        info['selection_available'] = selection_available
        
        if selection_available and getattr(self, 'use_selection_for_fitting', True):
            # Will use selected data
            info['will_use_selection'] = True
            info['data_source'] = 'selected_interval'
            info['data_range'] = [float(self.selected_positions[0]), float(self.selected_positions[-1])]
            info['data_points'] = len(self.selected_positions)
        else:
            # Will use complete dataset
            info['data_source'] = 'complete_dataset'
            info['data_range'] = [float(self.linescan_positions[0]), float(self.linescan_positions[-1])]
            info['data_points'] = len(self.linescan_positions)
        
        return info

    def export_results(self, event=None):
        """
        Export comprehensive results to files.
        """
        if not self.fitting_results:
            print("No results to export. Please perform ERF fitting first.")
            return
        
        # Default export directory: the folder of the loaded DM3 file, then the
        # loaded text-data folder, then the current working directory.
        default_dir = os.getcwd()
        if self.dm3_file_path:
            candidate = os.path.dirname(self.dm3_file_path)
            if os.path.isdir(candidate):
                default_dir = candidate
        elif self.text_data_path:
            candidate = os.path.dirname(self.text_data_path)
            if os.path.isdir(candidate):
                default_dir = candidate
        
        # Use a temporary hidden Tk root for the folder dialog + confirmation,
        # independent of matplotlib's active backend.
        temp_root = tk.Tk()
        temp_root.withdraw()
        try:
            export_dir = filedialog.askdirectory(
                title="Select Export Folder",
                initialdir=default_dir,
                parent=temp_root,
            )
            if not export_dir:
                print("⚠ Export cancelled by user.")
                return
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            exported_files = []
            
            # Export detailed results as JSON
            try:
                json_filename = os.path.join(export_dir, f"vcsel_erf_results_{timestamp}.json")
                with open(json_filename, 'w') as f:
                    json.dump(self.fitting_results, f, indent=2)
                print(f"✓ Results exported to {json_filename}")
                exported_files.append(json_filename)
            except Exception as json_error:
                print(f"⚠ Error exporting JSON results: {json_error}")
            
            # Export layer thicknesses as CSV
            try:
                csv_filename = os.path.join(export_dir, f"vcsel_layer_thicknesses_{timestamp}.csv")
                with open(csv_filename, 'w') as f:
                    f.write("Layer_Type,Layer_Number,Thickness_nm\n")
                    
                    # Export Material A layers with validation
                    if hasattr(self, 'layer_thicknesses') and 'Material_A' in self.layer_thicknesses:
                        for i, thickness in enumerate(self.layer_thicknesses['Material_A']):
                            if isinstance(thickness, (int, float)) and not np.isnan(thickness) and not np.isinf(thickness):
                                f.write(f"{self.material_name_a},{i+1},{thickness:.6f}\n")
                    
                    # Export Material B layers with validation
                    if hasattr(self, 'layer_thicknesses') and 'Material_B' in self.layer_thicknesses:
                        for i, thickness in enumerate(self.layer_thicknesses['Material_B']):
                            if isinstance(thickness, (int, float)) and not np.isnan(thickness) and not np.isinf(thickness):
                                f.write(f"{self.material_name_b},{i+1},{thickness:.6f}\n")
                print(f"✓ Layer thicknesses exported to {csv_filename}")
                exported_files.append(csv_filename)
            except Exception as csv_error:
                print(f"⚠ Error exporting layer thicknesses: {csv_error}")
            
            # Export fitted profile data
            if (self.fitted_parameters is not None and 
                self.linescan_positions is not None and 
                self.linescan_profile is not None):
                
                try:
                    profile_filename = os.path.join(export_dir, f"vcsel_fitted_profile_{timestamp}.txt")
                    
                    # Generate fitted curve
                    y_fit = self.build_erf_model_numpy(self.linescan_positions, self.fitted_parameters)
                    
                    # Validate array sizes before concatenation
                    if (len(self.linescan_positions) == len(self.linescan_profile) == len(y_fit)):
                        # Combine original and fitted data
                        export_data = np.column_stack((
                            self.linescan_positions,
                            self.linescan_profile,
                            y_fit,
                            self.linescan_profile - y_fit  # residuals
                        ))
                        
                        header = f"Position_nm\tOriginal_Intensity\tFitted_Intensity\tResiduals\n# Final fitting error: {self.final_loss:.2e}"
                        np.savetxt(profile_filename, export_data, delimiter='\t', header=header, comments='')
                        print(f"✓ Profile data exported to {profile_filename}")
                        exported_files.append(profile_filename)
                    else:
                        print(f"⚠ Skipping profile export: Array size mismatch "
                              f"(pos:{len(self.linescan_positions)}, "
                              f"profile:{len(self.linescan_profile)}, "
                              f"fit:{len(y_fit)})")
                        
                except Exception as profile_error:
                    print(f"⚠ Error exporting profile data: {profile_error}")
            
            # Export edge positions
            try:
                edge_filename = os.path.join(export_dir, f"vcsel_edge_positions_{timestamp}.txt")
                with open(edge_filename, 'w') as f:
                    f.write("Edge_Number\tPosition_nm\tParameter\n")
                    edge_param_indices = [4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67, 70, 73, 76, 79, 82, 85, 88, 91, 94, 97]
                    for i, pos in enumerate(self.edge_positions):
                        param_num = edge_param_indices[i] if i < len(edge_param_indices) else f"k{4 + i*3}"
                        f.write(f"{i+1}\t{pos:.6f}\t{param_num}\n")
                print(f"✓ Edge positions exported to {edge_filename}")
                exported_files.append(edge_filename)
            except Exception as edge_error:
                print(f"⚠ Error exporting edge positions: {edge_error}")
            
            # Confirmation dialog listing what was written and where
            if exported_files:
                file_list = "\n".join(f"• {os.path.basename(p)}" for p in exported_files)
                confirm_msg = (
                    f"✓ Export completed successfully!\n\n"
                    f"Folder:\n{export_dir}\n\n"
                    f"Files ({len(exported_files)}):\n{file_list}"
                )
                messagebox.showinfo("Export Complete", confirm_msg, parent=temp_root)
            else:
                messagebox.showwarning(
                    "Export",
                    f"No files were written to:\n{export_dir}\n\nSee the console for details.",
                    parent=temp_root,
                )
            print("✓ Export completed successfully!")
        finally:
            temp_root.destroy()


def main():
    """
    Main function to run the combined VCSEL analyzer.
    """
    if not HAS_SCIPY:
        print("✗ SciPy not available. Please install: pip install scipy")
        sys.exit(1)

    print("="*80)
    print("COMBINED ERF VCSEL CAVITY ANALYZER")
    print("="*80)
    print("Features:")
    print("- Load DM3 microscopy files and make intensity linescan")
    print("- Show scan profile")
    print("- Set number of ERF fitting parameters (k1, k2, ..., k97)")
    print("- High-precision fitting with error < 1e-10")
    print("- Show scan profile and fitting curve with edge points")
    print("- Show layer thickness contribution")
    print("- Comprehensive data export")
    print("="*80)
    
    # Create analyzer instance
    analyzer = CombinedVCSELAnalyzer()
    
    # Try to load DM3 file
    if analyzer.load_dm3_file():
        print("\n✓ DM3 file loaded successfully!")
        print("Starting interactive interface...")
        
        # Create interactive interface
        analyzer.create_interactive_interface()
    else:
        print("\n⚠ Could not load DM3 file.")
        print("You can still use the text data loading feature.")
        
        # Ask user what to do
        root = tk.Tk()
        root.withdraw()
        
        choice = messagebox.askyesno("No DM3 File", 
                                   "No DM3 file was loaded.\n\n" +
                                   "Would you like to load text data instead?\n" +
                                   "(Click 'No' to exit)")
        
        if choice:
            analyzer.load_text_data()
            if analyzer.linescan_profile is not None:
                print("✓ Text data loaded. You can now configure parameters and fit.")
        
        root.destroy()


if __name__ == "__main__":
    main()