import itertools
import random
import copy

class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        if len(self.cells) > 0 and self.count > 0:
            return self.cells

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        if len(self.cells) > 0 and self.count == 0:
            return self.cells

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            # update the sentence so that cell is no longer in the sentence, and update the count logic
            self.cells.remove(cell)
            self.count -= 1


    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells:
            self.cells.remove(cell)


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def find_neighbors(self, cell):
        """
        :param cell: tuple of (x,y) coordinates
        :return: returns an array of neighbors that are not already marked as safe or a mine
        """
        neighbors = []
        x,y = cell
        for i in range(x-1, x+2):
            for j in range(y-1, y+2):
                if 0 <= i < self.height and 0 <= j < self.width:
                    if (i,j) not in self.moves_made:
                        neighbors.append((i,j))

        return neighbors

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.
        """
        # 1) mark the cell as a move that has been made
        self.moves_made.add(cell)
        # 2) mark the cell as safe
        self.mark_safe(cell)
        # 3) add a new sentence to the AI's knowledge base
        # based on the value of `cell` and `count`
        #include cells whose state is still undetermined in the sentence.
        neighbors = self.find_neighbors(cell)
        sentence = Sentence(neighbors, count)
        self.knowledge.append(sentence)

        # 4) mark any additional cells as safe or as mines
        # if it can be concluded based on the AI's knowledge base

        # Simple Inferences
        for sentence in self.knowledge:
            # all mines
            if len(sentence.cells) == sentence.count:
                mine_cells = copy.deepcopy(sentence.cells)
                for cell in mine_cells:
                    self.mark_mine(cell)
                del sentence # clean up knowledge after marking all cells as mines

            # all safe
            elif sentence.count == 0:
                safe_cells = copy.deepcopy(sentence.cells)
                for cell in safe_cells:
                    self.mark_safe(cell)
                del sentence # clean up knowledge after marking all cells as safe
            else:
                continue

        # 5) add any new sentences to the AI's knowledge base
        # if they can be inferred from existing knowledge
        static_knowledge = copy.deepcopy(self.knowledge)
        for sentence1 in static_knowledge:
            if len(sentence1.cells) == 0:
                continue
            for sentence2 in static_knowledge:
                if len(sentence2.cells) == 0:
                    continue
                if sentence1 != sentence2 and sentence1.cells.issubset(sentence2.cells):
                    print('Found subset', 'S1', sentence1.cells, 'S2', sentence2.cells)
                    new_cells = sentence2.cells.difference(sentence1.cells)
                    new_count = sentence2.count - sentence1.count
                    for cell in new_cells:
                        if cell in self.moves_made:
                            new_cells.remove(cell)

                    for sentence in static_knowledge:
                        if new_cells.issubset(sentence.cells):
                            continue # already part of our sentence
                    self.knowledge.append(Sentence(new_cells, new_count))
                else:
                    continue

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        if len(self.safes) > 0:
            for move in self.safes:
                if move not in self.mines and move not in self.moves_made:
                    return move

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        for i in range(self.height):
            for j in range(self.width):
                move = (i, j)
                if move not in self.mines and move not in self.moves_made:
                    return move
