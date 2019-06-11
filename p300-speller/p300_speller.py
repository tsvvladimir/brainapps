import os
import time
import pandas as pd
import numpy as np
import random
try:
    # for Python2
    import Tkinter as tk
    from Tkinter import *
except ImportError:
    # for Python3
    import tkinter as tk
    from tkinter import *
import argparse
import yaml
import pickle
import logging
from enum import Enum

from autocomplete import autocomplete

from brainflow import *

import classifier
from ui_helpers import StartScreen, SelectionRectangle

class SpecialChars (Enum):
    Delete = 'DEL'
    Space = 'SPACE'


class P300GUI (tk.Frame):
    """The main screen of the application that displays the character grid and spelling buffer"""
    def __init__ (self, master, settings, perform_calibration = False):
        tk.Frame.__init__ (self, master)

        # set some variables from settings
        self.settings = settings
        for key, value in self.settings.items ():
            setattr (self, key, value)
        self.num_cols = self.general['num_cols']
        self.num_rows = self.general['num_rows']
        self.last_row = [x.upper () for x in self.general['default_words']]
        # add delete and space buttons to last row
        self.last_row.append (SpecialChars.Delete.value)
        self.last_row.append (SpecialChars.Space.value)
        self.grid_width = self.window_settings['grid_width']
        self.grid_height = self.window_settings['grid_height']
        self.highlight_time = self.presentation['highlight_time']
        self.intermediate_time = self.presentation['intermediate_time']
        self.epoch_length = self.general['epoch_length']
        self.col_width = self.grid_width / self.num_cols
        self.row_height = self.grid_height / self.num_rows

        # df to store events
        self.event_data = pd.DataFrame (columns = ['event_start_time', 'orientation', 'highlighted', 'trial_row', 'trial_col', 'calibration'])

        # board params
        self.is_streaming = False
        if self.general['board'] == 'Cyton':
            self.board_id = CYTON.board_id
            self.board = BoardShim (self.board_id, self.general['port'])
            self.board.prepare_session ()
        else:
            raise ValueError ('unsupported board type')

        # training params
        self.is_training = False
        self.trial_row = -1
        self.trial_col = -1
        self.trial_count = 0
        self.perform_calibration = perform_calibration
        self.calibration_completed = False
        self.untrained_chars = list (range (self.num_cols * self.num_rows))
        self.training_sequence = list ()
        self.trial_in_progress = False
        self.char_highlighted = False

        # live params
        self.scaler = None
        self.pca = None
        self.classifier = None
        self.autocomplete = autocomplete.Autocomplete ()
        self.current_live_count = 0
        self.cols_weights = None
        self.rows_weights = None
        self.cols_predictions = None
        self.rows_predictions = None

        # general
        self.selection_rect = self.make_rectangle ()
        self.canvas = tk.Canvas (self)
        self.spelled_text = tk.StringVar ()
        self.spelled_text.set ('')
        self.text_buffer = tk.Entry (self, font = ('Arial', 24, 'bold'), cursor = 'arrow',
                                    insertbackground = '#ffffff', textvariable = self.spelled_text)
        self.sequence_count = 0
        self.char_select_rect = SelectionRectangle (
                                                    self.settings,
                                                    x = self.col_width * self.trial_col, y = self.row_height * self.trial_row,
                                                    width = self.col_width, length = self.row_height,
                                                    max_x = self.grid_width, max_y = self.grid_height,
                                                    color = self.presentation['char_select_color']
                                                    )
        self.create_widgets ()

    def set_training_mode (self, is_training):
        """Set training or live mode"""
        self.is_training = is_training
        if self.is_training:
            self.perform_calibration = True
        if self.perform_calibration:
            self.training_sequence += [x.upper () for x in self.calib_params['word']]
        if self.is_training:
            for i in range (self.training_params['num_trials']):
                rand_index = random.randint (0, len (self.untrained_chars) - 1)
                char_num = self.untrained_chars[rand_index]
                row = char_num // self.num_cols
                col = char_num % self.num_cols
                self.untrained_chars.remove (self.untrained_chars[rand_index])
                self.training_sequence.append (self.get_character (row, col))
        if self.training_sequence:
            logging.debug ('training sequence: %s' % ' '.join ([x for x in self.training_sequence]))

        if not self.is_training:
            with open (os.path.join (os.path.dirname (os.path.realpath (__file__)), 'data', self.settings['general']['scaler']), 'rb') as f:
                self.scaler = pickle.load (f)
            with open (os.path.join (os.path.dirname (os.path.realpath (__file__)), 'data',self.settings['general']['transformer']), 'rb') as f:
                self.pca = pickle.load (f)
            with open (os.path.join (os.path.dirname (os.path.realpath (__file__)), 'data', self.settings['general']['classifier']), 'rb') as f:
                self.classifier = pickle.load (f)

            self.current_live_count = 0
            self.cols_weights = numpy.zeros (self.num_cols).astype (numpy.float64)
            self.rows_weights = numpy.zeros (self.num_rows).astype (numpy.float64)
            self.cols_predictions = numpy.zeros (self.num_cols).astype (numpy.float64)
            self.rows_predictions = numpy.zeros (self.num_rows).astype (numpy.float64)

    def display_screen (self):
        """Adds this screen to the window"""
        self.place (relx = 0.5, rely = 0.5, anchor = CENTER)

    def remove_screen (self):
        """Removes this screen from the window"""
        self.place_forget ()

    def update (self):
        """Updates the gui based on the mode the application is in"""
        if not self.is_streaming:
            self.board.start_stream (7200 * self.board.fs_hz)
            self.is_streaming = True
        # Moves the selection rect off-screen
        self.selection_rect.move_to_col (-2)
        if self.training_sequence:
            self.training_update ()
        else:
            if self.is_training:
                self.write_and_exit ()
            else:
                if not self.calibration_completed:
                    self.spelled_text.set ('Your text:')
                    self.calibration_completed = True
                    if self.perform_calibration:
                        self.master.after (self.training_params['wait_timeout'], self.update)
                    else:
                        self.draw_characters ()
                        self.master.after (self.training_params['wait_timeout'], self.update)
                else:
                    self.live_update ()

    def training_update (self):
        """Updates the gui while in training mode"""
        # Highlight the character when we are currently not in the middle of a trial
        if not self.trial_in_progress:
            character = self.training_sequence[0]
            self.trial_row, self.trial_col = self.get_row_col (character)
            # Move the char highlight rect behind the character
            self.char_select_rect.move_to_col (self.trial_col, reset_top = False, rotate = False)
            self.char_select_rect.move_to_row (self.trial_row, reset_left = False, rotate = False)
            # highlight the character
            self.char_highlighted = True
            self.trial_in_progress = True
            self.draw ()
            self.spelled_text.set ('Look at: %s' % str (self.get_character (self.trial_row, self.trial_col)))
            # Wait
            self.master.after (self.settings['training_params']['wait_timeout'], self.update)
        # start blinking
        elif self.trial_in_progress:
            # Turn off the highlighting of the character
            if self.char_highlighted:
                self.char_highlighted = False
                self.draw ()
            # Proceed updating like normal
            if self.selection_rect.visible:
                # Update the position of the rectangle
                self.selection_rect.update ()
                self.record_event ()
                # Rectangle is set to visible, draw the canvas
                self.draw ()
                # Set it visibility for when this function is called again
                self.selection_rect.visible = False
                # Allow the rectangle to remain visible for a set time
                self.master.after (self.highlight_time, self.update)
            else:
                # Rectangle is set to invisible, update the canvas
                self.draw ()
                # Set visibility to visible for next update call
                self.selection_rect.visible = True

                if self.selection_rect.end_of_sequence ():
                    self.sequence_count = self.sequence_count + 1
                    if self.sequence_count >= self.training_params['seq_per_trial']:
                        self.trial_count = self.trial_count + 1
                        self.sequence_count = 0
                        self.trial_in_progress = False
                        self.training_sequence.pop (0)
                        if self.trial_count == len (self.calib_params['word']) and self.perform_calibration:
                            self.calibration_completed = True
                        self.master.after (self.epoch_length + self.intermediate_time, self.update)
                    else:
                        self.master.after (self.epoch_length + self.intermediate_time, self.update)
                else:
                    # Keep the rect invisible for a set amount of time
                    self.master.after (self.intermediate_time, self.update)


    def live_update (self):
        """Updates the position and visibility of the selection rectangle"""
        if self.selection_rect.visible:
            self.selection_rect.update ()
            self.record_event ()
            self.draw ()
            self.selection_rect.visible = False
            # Allow the rectangle to remain visible for a set time
            self.master.after (self.highlight_time, self.update)
        else:
            self.draw ()
            # Set visibility to visible for next update call
            self.selection_rect.visible = True
            if self.selection_rect.end_of_sequence ():
                self.sequence_count = self.sequence_count + 1
                if self.sequence_count >= self.live_params['seq_per_trial']:
                    self.master.after (self.epoch_length + self.intermediate_time, self.update)
                    time.sleep (self.epoch_length / 1000.0)
                    # get predicted character
                    predicted_row, predicted_col = self.get_predicted ()
                    # free old events
                    self.event_data = pd.DataFrame (columns = ['event_start_time', 'orientation', 'highlighted', 'trial_row', 'trial_col', 'calibration'])
                    if predicted_row is not None and predicted_col is not None:
                        self.add_text (predicted_row, predicted_col)
                        self.update_words ()
                        self.free_live_variables ()
                    else:
                        logging.info ('trying to expand data to improve confidence')
                    self.sequence_count = 0
                else:
                    self.master.after (self.settings['general']['epoch_length'] + self.intermediate_time, self.update)
            else:
                # Keep the rect invisible for a set amount of time
                self.master.after (self.intermediate_time, self.update)

    def free_live_variables (self):
        # free current weights
        self.current_live_count = 0
        self.cols_weights = numpy.zeros (self.num_cols).astype (numpy.float64)
        self.rows_weights = numpy.zeros (self.num_rows).astype (numpy.float64)
        self.cols_predictions = numpy.zeros (self.num_cols).astype (numpy.int64)
        self.rows_predictions = numpy.zeros (self.num_rows).astype (numpy.int64)

    def get_row_col (self, character):
        if character.isdigit ():
            cell_num = 26 + int (character)
        elif len (character) == 1:
            cell_num = ord (character.upper ()) - ord ('A')
        else:
            for i, word in enumerate (self.last_row):
                if word == character:
                    cell_num = self.num_cols * (self.num_rows - 1) + i
        row = cell_num // self.num_cols
        col = cell_num % self.num_cols
        return row, col

    def get_character (self, row, col):
        """Returns the character from the grid at the given row and column"""
        cell_num = (row * self.num_cols) + col
        if cell_num <= 25:
            return chr (ord ('A') + cell_num)
        elif row < (self.num_rows - 1):
            return str (cell_num - 26)
        elif row == (self.num_rows - 1):
            return self.last_row[col]
        else:
            raise ValueError ('wrong row\col')

    def draw (self):
        """Redraws the canvas"""
        self.canvas.delete ('all')
        if self.char_highlighted:
            self.selection_rect.x = -10000
            self.selection_rect.y = -10000
            self.char_select_rect.draw (self.canvas)
        else:
            self.selection_rect.draw (self.canvas)
        self.draw_characters ()

    def make_rectangle (self, orientation = 'vertical'):
        """Returns a new selection rectangle for this GUI"""
        if orientation == 'vertical':
            return SelectionRectangle (
                                        self.settings,
                                        x = 0, y = 0,
                                        width = self.col_width, length = self.grid_height,
                                        color = self.presentation['rect_color'],
                                        max_x = self.grid_width, max_y = self.grid_height
                                      )
        else:
            return SelectionRectangle (
                                        self.settings,
                                        x = 0, y = 0,
                                        width = self.grid_width, length = self.row_height,
                                        color = self.presentation['rect_color'],
                                        max_x = self.grid_width, max_y = self.grid_height
                                      )

    def draw_characters (self):
        """Draws"""
        max_word_len = max ([len (x) for x in self.last_row])
        for row in range (self.num_rows):
            for col in range (self.num_cols):
                element = self.get_character (row, col)
                # Determine if this character is printed white or black
                if self.selection_rect != None:
                    if ((self.selection_rect.is_vertical () and col == self.selection_rect.get_index ()
                         or not self.selection_rect.is_vertical () and row == self.selection_rect.get_index ())
                         and self.selection_rect.visible):

                        if row == (self.num_rows - 1):
                            font_size = int (self.col_width / (1.25 * max_word_len))
                        else:
                            font_size = int (self.col_width / 3.5)
                        canvas_id = self.canvas.create_text ((self.col_width * col) + (self.col_width / 2.5),
                                                    (self.row_height * row) + (self.row_height / 3),
                                                    font = ('Arial', font_size, 'bold'),
                                                    anchor = 'nw')
                        self.canvas.itemconfig (canvas_id, text = element, fill = self.settings['presentation']['highlight_char_color'])
                    else:
                        if row == (self.num_rows - 1):
                            font_size = int (self.col_width / (1.5 * max_word_len))
                        else:
                            font_size = int (self.col_width / 4)
                        canvas_id = self.canvas.create_text ((self.col_width * col) + (self.col_width / 2.5),
                                                    (self.row_height * row) + (self.row_height / 3),
                                                    font = ('Arial', font_size, 'bold'),
                                                    anchor = 'nw')
                        self.canvas.itemconfig (canvas_id, text = element, fill = self.settings['presentation']['default_char_color'])

    def add_space (self):
        """Adds a space '_' to the spelled text buffer"""
        self.spelled_text.set (self.spelled_text.get () + "_")
        self.text_buffer.icursor (len (self.spelled_text.get ()))
        self.last_row = [x.upper () for x in self.general['default_words']]
        self.last_row.append (SpecialChars.Delete.value)
        self.last_row.append (SpecialChars.Space.value)

    def delete_last (self):
        """Deletes the last character in the spelled text buffer"""
        if len (self.spelled_text.get ().split (':')[1]) > 0:
            self.spelled_text.set (self.spelled_text.get ()[:-1])
            self.text_buffer.icursor (len (self.spelled_text.get ()))

    def update_words (self):
        current_text = self.spelled_text.get ().split (':')[1]
        text_to_predict_next = current_text.replace ('_', ' ')
        predicted = self.autocomplete.split_predict (text_to_predict_next)
        logging.debug ('current text: %s predicted words: %s' % (text_to_predict_next, str (predicted)))
        for i, prediction in enumerate (predicted):
            if i < 4:
                self.last_row[i] = prediction[0].upper ()
            else:
                break

    def add_text (self, row, col):
        """Appends some given text to the sppelled text buffer"""
        if row == (self.num_rows - 1) and col == (self.num_cols - 2):
            self.delete_last ()
        elif row == (self.num_rows - 1) and col == (self.num_cols - 1):
            self.add_space ()
        elif row == (self.num_rows - 1):
            text = self.get_character (row, col)
            common, current_text = self.spelled_text.get ().split (':')
            new_text = common + ':' + current_text[0:current_text.rfind ('_') + 1] + text
            self.spelled_text.set (new_text)
            self.text_buffer.icursor (len (self.spelled_text.get ()))
            self.add_space ()
        else:
            text = self.get_character (row, col)
            self.spelled_text.set (self.spelled_text.get () + text)
            self.text_buffer.icursor (len (self.spelled_text.get ()))

    def create_widgets (self):
        """Populates the gui with all the necessary components"""
        self.master['bg'] = '#001c33'
        self['bg'] = '#001c33'
        # Displays the current text being typed
        self.text_buffer.grid (row = 0, pady = 20, sticky = tk.W + tk.E)
        self.text_buffer['fg'] = '#ffffff'
        self.text_buffer['bg'] = '#000000'
        # Canvas for drawing the grid of characters and the rectangle
        self.canvas['width'] = self.grid_width
        self.canvas['height'] = self.canvas['width']
        self.canvas['bg'] = self.settings['presentation']['grid_bg_color']
        self.canvas.grid (row = 2, sticky = tk.W + tk.E)
        # Frame to hold all buttons at the bttom of the gui
        self.bottom_button_pane = tk.Frame (self)
        self.bottom_button_pane.grid (pady = 10)
        # Button to delete the previous character
        self.back_space_button = tk.Button (self.bottom_button_pane, text = 'delete', command = self.delete_last, height = 1, width = 6)
        self.back_space_button.grid (row = 0,column = 0)
        # Button for adding a space character to the text_buffer
        self.space_button = tk.Button (self.bottom_button_pane, text = 'space', command = self.add_space, height = 1, width = 12)
        self.space_button.grid (row = 0,column = 1)
        # Button for exiting the application
        self.exit_button = tk.Button (self.bottom_button_pane, text = 'exit', command = self.write_and_exit, height = 1, width = 6)
        self.exit_button.grid (row = 0,column = 3)

    def record_event (self):
        """Sends epoch event codes and times to the main process"""
        index = self.selection_rect.get_index ()
        if self.selection_rect.is_vertical ():
            orientation = 'col'
        else:
            orientation = 'row'
        if self.perform_calibration and not self.calibration_completed:
            calibration = 1
        else:
            calibration = 0
        if self.is_training or (self.perform_calibration and not self.calibration_completed):
            self.event_data = self.event_data.append ({'event_start_time' : time.time (), 'orientation' : orientation, 'highlighted':index,
                                                        'trial_row': self.trial_row, 'trial_col' : self.trial_col, 'calibration': calibration}, ignore_index = True)
        else:
            self.event_data = self.event_data.append ({'event_start_time' : time.time (), 'orientation' : orientation, 'highlighted':index,
                                                        'trial_row': -1, 'trial_col' : -1, 'calibration': -1}, ignore_index = True)

    def write_and_exit (self):
        if self.is_training:
            time_created = int (time.time ())
            event_file = os.path.join (os.path.dirname (os.path.abspath (__file__)), 'data', 'events_%d.csv' % time_created)
            self.event_data.to_csv (event_file, index = False)

            data = self.board.get_board_data ()
            data_handler = DataHandler (self.board_id, numpy_data = data)
            data_handler.save_csv (os.path.join (os.path.dirname (os.path.abspath (__file__)), 'data', 'eeg_%d.csv' % time_created))
        self.master.quit ()

    def get_predicted (self):
        """perform prediction"""
        self.current_live_count = self.current_live_count + 1

        eeg_data = self.board.get_current_board_data (int (CYTON.fs_hz * (self.epoch_length * (self.live_params['seq_per_trial'] + 2)) / 1000.0 ))
        data_handler = DataHandler (self.board_id, numpy_data = eeg_data)
        eeg_data = data_handler.get_data ()
        eeg_data.index.name = 'index'
        self.event_data.index.name = 'index'

        data_x, _ = classifier.prepare_data (eeg_data, self.event_data, self.settings, False)
        if data_x.shape[0] != self.live_params['seq_per_trial'] * (self.num_cols + self.num_rows):
            logging.error ('Incorrect data shape:%d, exptected:%d' % (data_x.shape[0],
                            self.live_params['seq_per_trial'] * (self.num_cols + self.num_rows)))
        decisions = classifier.get_decison (data_x, self.scaler, self.pca, self.classifier)

        for i, decision in enumerate (decisions):
            event = self.event_data.iloc[i,:]
            if event['orientation'] == 'col':
                if decision > 0:
                    self.cols_predictions[event['highlighted']] = self.cols_predictions[event['highlighted']] + 1
                self.cols_weights[event['highlighted']] = self.cols_weights[event['highlighted']] + decision
            else:
                if decision > 0:
                    self.rows_predictions[event['highlighted']] = self.rows_predictions[event['highlighted']] + 1
                self.rows_weights[event['highlighted']] = self.rows_weights[event['highlighted']] + decision

        best_col_id = None
        best_row_id = None

        max_col_predictions = 0
        max_col_decision = 0
        max_row_predictions = 0
        max_row_decision = 0

        for i in range (self.num_cols):
            if self.cols_predictions[i] > max_col_predictions:
                best_col_id = i
                max_col_predictions = self.cols_predictions[i]
                max_col_decision = self.cols_weights[i]
            elif self.cols_predictions[i] == max_col_predictions and max_col_decision < self.cols_weights[i]:
                best_col_id = i
                max_col_predictions = self.cols_predictions[i]
                max_col_decision = self.cols_weights[i]

        for i in range (self.num_rows):
            if self.rows_predictions[i] > max_row_predictions:
                best_row_id = i
                max_row_predictions = self.rows_predictions[i]
                max_row_decision = self.rows_weights[i]
            elif self.rows_predictions[i] == max_row_predictions and max_row_decision < self.rows_weights[i]:
                best_row_id = i
                max_row_predictions = self.rows_predictions[i]
                max_row_decision = self.rows_weights[i]

        if self.current_live_count < self.live_params['max_repeat']:
            second_col_score = sorted (self.cols_predictions.tolist ())[-2]
            second_row_score = sorted (self.rows_predictions.tolist ())[-2]

            if best_col_id is not None:
                val = self.cols_predictions[best_col_id]
                if ((float (val - second_col_score)) / val < self.live_params['stop_thresh']):
                    best_col_id = None
                if val < 2:
                    best_col_id = None

            if best_row_id is not None:
                val = self.rows_predictions[best_row_id]
                if ((float (val - second_row_score)) / val < self.live_params['stop_thresh']):
                    best_row_id = None
                if val < 2:
                    best_row_id = None

        logging.info ('col weights %s' % ' '.join (['%.2f' % x for x in self.cols_weights]))
        logging.info ('row weights %s' % ' '.join (['%.2f' % x for x in self.rows_weights]))
        logging.info ('col predictions %s' % ' '.join ([str (x) for x in self.cols_predictions]))
        logging.info ('row predictions %s' % ' '.join ([str (x) for x in self.rows_predictions]))
        logging.info ('predicted col: %s predicted row:%s' % (str (best_col_id), str (best_row_id)))
        return best_row_id, best_col_id


def main ():
    parser = argparse.ArgumentParser ()
    parser.add_argument ('--settings', type = str, help  = 'settings file', default = 'ui_settings.yml')
    parser.add_argument ('--perform-calibration', action = 'store_true')
    parser.add_argument ('--debug', action = 'store_true')
    args = parser.parse_args ()
    if args.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logging.basicConfig (level = log_level)

    settings = yaml.load (open (args.settings))
    window_settings = settings['window_settings']

    root = tk.Tk ()
    root.title ('Brainflow P300 speller')
    root.protocol ('WM_DELETE_WINDOW', root.quit)
    root.geometry ('{}x{}'.format (window_settings['geometry_x'], window_settings['geometry_y']))
    root.resizable (width = False, height = False)
    speller_gui = P300GUI (root, settings, args.perform_calibration)
    start_screen = StartScreen (root, speller_gui)
    start_screen.display_screen ()
    root.mainloop ()


if __name__ == "__main__":
    main ()
