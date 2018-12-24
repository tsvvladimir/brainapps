import random
try:
    # for Python2
    import Tkinter as tk
    from Tkinter import *
except ImportError:
    # for Python3
    import tkinter as tk
    from tkinter import *


class StartScreen (tk.Frame):
    """Starting screen for the application"""
    def __init__ (self, master, next_screen):
        tk.Frame.__init__ (self, master)
        training_instructions = ('1) A character will be highlighted at the\n'
                        'beginning of each trial\n\n'
                        '2) Fixate on the character\n\n'
                        '3) Rows and columns will begin to flash\n\n'
                        '3) Continue to fixate on the character until\n'
                        'another character is highlighted\n')

        live_instructions = ('1) Fixate on the character you wish to select\n\n'
                    '2) A character will be predicted and types after\n'
                    'a set amount of rounds\n')

        calibration_instructions = ('At the begining of each session there is a calibations phase\n\n'
                    'During calibration phase you will spell a predefined word, so you should focus on\n'
                    'highlighted character\n')

        self.title_text = tk.Label (self, text = "Brainflow P300 Speller", font = ('Arial', 24))
        self.title_text.grid ()

        self.directions_label = tk.Label (self, text = 'Directions:', font = ('Arial', 18))
        self.directions_label.grid (sticky = tk.W)

        self.training_label = tk.Label(self, text = 'Training:', font = ('Arial', 16))
        self.training_label.grid (sticky = tk.W)

        self.training_text = tk.Label (self, text = training_instructions, font = ('Arial', 14), justify = LEFT)
        self.training_text.grid (sticky = tk.W)

        self.live_label = tk.Label (self, text = 'Live Spelling:', font = ('Arial', 16))
        self.live_label.grid (sticky = tk.W)

        self.calib_label = tk.Label (self, text = 'Calibration:', font = ('Arial', 16))
        self.calib_label.grid (sticky = tk.W)

        self.calib_text = tk.Label (self, text = calibration_instructions, font = ('Arial', 14), justify = LEFT)
        self.calib_text.grid (sticky = tk.W)

        self.live_text = tk.Label (self, text = live_instructions, font = ('Arial', 14), justify = LEFT)
        self.live_text.grid (sticky = tk.W)

        self.start_training_button = tk.Button (self, command = self.start_training, text = 'Train', font = ('Arial', 24, 'bold'), height = 4, width = 24)
        self.start_training_button.grid (pady = 3, sticky = tk.W + tk.E)

        self.start_live_button = tk.Button (self, command = self.start_live, text = 'Live', font = ('Arial', 24, 'bold'), height = 4, width = 24)
        self.start_live_button.grid (pady = 3, sticky = tk.W + tk.E)

        self.next_screen = next_screen

    def display_screen (self):
        """Adds this screen to the window"""
        self.place (relx = 0.5, rely = 0.5, anchor = CENTER)

    def remove_screen (self):
        """Removes this screen from the window"""
        self.place_forget ()

    def start_training (self):
        self.next_screen.set_training_mode (True)
        self.__start_speller ()

    def start_live (self):
        self.next_screen.set_training_mode (False)
        self.__start_speller ()

    def __start_speller (self):
        """Removes this frame and displays the grid of characters"""
        self.next_screen.display_screen ()
        self.next_screen.update ()
        self.remove_screen ()


class SelectionRectangle ():
    """Manages the rectangle that highlights the characters in the grid"""
    def __init__ (self, settings, x, y, length, width, max_x, max_y, color = '#ffffff'):
        self.settings = settings
        # x,y - top left position
        self.x = x
        self.y = y
        self.length = length
        self.width = width
        self.graphic_ref = None
        self.color = color
        self.max_x = max_x
        self.max_y = max_y
        self.remaining_rows = range (self.settings['general']['num_rows'])
        self.remaining_cols = range (self.settings['general']['num_cols'])
        self.visible = True

    def get_index (self):
        """Return the current row or column index of the rectangle"""
        if self.is_vertical ():
            return int (self.x / self.width)
        else:
            return int (self.y / self.length)

    def move_to_col (self, index, reset_top = True, rotate = True):
        """Moves and re-orients the rectangle to a column specified by an index"""
        # Reorient the rectangle to be vertical
        if not self.is_vertical() and rotate:
            self.rotate90 ()
        # Set the rectangle to the proper position
        self.x = index * self.width
        if reset_top:
            self.y = 0

    def move_to_row (self, index, reset_left = True, rotate = True):
        """Moves and re-orients the rectangle to a row specified by an index"""
        # Reorient the rectangle to be horizontal
        if self.is_vertical () and rotate:
            self.rotate90 ()
        # Set the rectangel to the proper position
        self.y = index * self.length
        if reset_left:
            self.x = 0

    def rotate90 (self):
        """Rotates the rectangle 90 degrees"""
        temp = self.width
        self.width = self.length
        self.length = temp

    def move_vertical (self, distance):
        """Moves the rectangle by some distance in the y-direction"""
        self.y += distance

    def move_horizontal (self, distance):
        """Moves the rectangle by some distance in the x-direction"""
        self.x += distance

    def is_vertical (self):
        """Returns true if the rectangle is oriented vertically"""
        return self.length > self.width

    def refill_available_rcs (self):
        """Refills the lists of available rows and columns with index values"""
        self.remaining_rows = range (self.settings['general']['num_rows'])
        self.remaining_cols = range (self.settings['general']['num_cols'])

    def select_rand_row (self):
        """Selects a row from the available_rows"""
        rand_index = random.randint (0, len (self.remaining_rows) - 1)
        row = self.remaining_rows[rand_index]
        return row

    def select_rand_col (self):
        """Selects a random column from the available_cols"""
        rand_index = random.randint (0, len (self.remaining_cols) - 1)
        col = self.remaining_cols[rand_index]
        return col

    def end_of_sequence (self):
        """Returns true if there are no more available moves for the rect"""
        return len (self.remaining_cols) == 0 and len (self.remaining_rows) == 0

    def update (self):
        """Moves the recangle by one row or column and creates epoch"""
        # Move the rectangle by randomly selecting a row or column
        if self.settings['presentation']['random_highlight']:
            # The remaining columns and row lists need to be refilled
            if self.end_of_sequence ():
                self.refill_available_rcs ()

            # Freely choose between available rows and columns
            if len (self.remaining_cols) > 0 and len (self.remaining_rows) > 0:
                # if previous one was a column I want the next one to be a column as well
                # I suppose it decreases double flash effect a little
                if self.is_vertical ():
                    min_thresh = 0.2
                else:
                    min_thresh = 0.8
                if random.random () > min_thresh:
                    next_col = self.select_rand_col ()
                    self.move_to_col (next_col)
                    self.remaining_cols.remove (next_col)
                else:
                    next_row = self.select_rand_row ()
                    self.move_to_row (next_row)
                    self.remaining_rows.remove (next_row)

            elif len (self.remaining_cols) == 0:
                next_row = self.select_rand_row ()
                self.move_to_row (next_row)
                self.remaining_rows.remove (next_row)

            elif len (self.remaining_rows) == 0:
                next_col = self.select_rand_col ()
                self.move_to_col (next_col)
                self.remaining_cols.remove (next_col)

        # Move linearly through all the rows and columns
        else:
            if self.is_vertical ():
                self.move_horizontal (self.width)
                if self.x + self.width > self.max_x:
                    self.x = 0
                    self.rotate90 ()
            else:
                self.move_vertical (self.length)
                if self.y + self.length > self.max_y:
                    self.y = 0
                    self.rotate90 ()

    def draw (self, canvas):
        """Draws the rectange to a Tkinter canvas"""
        if self.visible:
            if self.graphic_ref != None:
                canvas.delete (self.graphic_ref)
            self.graphic_ref = canvas.create_rectangle (
                                                        self.x, self.y,
                                                        self.x + self.width,
                                                        self.y + self.length,
                                                        fill = self.color
                                                        )
