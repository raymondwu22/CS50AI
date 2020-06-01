from os import system
import time

import crossword as cw
from generate import CrosswordCreator


def get_var(crossword, direction, i, j):
    """
    Get variable from crossword.

    Returns the variable with the given i, j and dircetion.
    Returns None if not found.
    """
    for var in crossword.variables:
        if var.i == i and var.j == j and var.direction == direction:
            return var
    return


def print_result(result, false_prefix, true_string):
    """Print boolean result."""
    if not result:
        print(f"{false_prefix} ", end='')
    print(true_string)
    print()


def main():
    """
    Test Functionality Implementations.

    Tests the Implemented Functions against the Spacifications.
    """
    # Create Generic Test Crosswords
    crossword_0 = cw.Crossword('data/structure0.txt', 'data/words0.txt')
    crossword_1 = cw.Crossword('data/structure1.txt', 'data/words1.txt')
    crossword_2 = cw.Crossword('data/structure2.txt', 'data/words2.txt')
    ##crossword_test = cw.Crossword('data/structure_test.txt', 'data/words_test.txt')
    # Test Implemented Functions on demand
    tests = {
        'N': 'enforce_node_consistency()',
        'R': 'revise()',
        '3': 'ac3()',
        'A': 'assignment_complete()',
        'C': 'consistent()',
        'O': 'order_domain_values()',
        'U': 'select_unassigned_variable()',
        'B': 'backtrack()'
    }
    system('cls')
    while True:
        print('Testing the CrosswordCreator:')
        for key, description in tests.items():
            print(f'{key}: {description}')
        action = (input(f'Your choice (<Enter> to stop): ') + ' ')[0].upper()
        if action == ' ':
            break
        if action not in tests.keys():
            print(f'Illegal Choice: {action}')
        else:
            # Start the chosen Test
            system('cls')
            print(f'Test {tests[action]}')
            print()
            if action == 'N':
                # 'N': 'enforce_node_consistency'
                creator = CrosswordCreator(crossword_0)
                # Prepare Testcase
                for var, domain in creator.domains.items():
                    print(var, len(domain))
                print()
                # Apply enforce_node_consistency()
                input(f'Press <Enter> to test {tests[action]} ...')
                creator.enforce_node_consistency()
                print()
                for var, domain in creator.domains.items():
                    print(var, domain)
                print()
            elif action == 'R':
                # 'R': 'revise'
                creator = CrosswordCreator(crossword_0)
                # Prepare Testcase
                across_4_1 = get_var(crossword_0, cw.Variable.ACROSS, 4, 1)
                down_0_1 = get_var(crossword_0, cw.Variable.DOWN, 0, 1)
                down_1_4 = get_var(crossword_0, cw.Variable.DOWN, 1, 4)
                creator.enforce_node_consistency()
                testset = (across_4_1, down_0_1, down_1_4)
                creator.domains[across_4_1].add('NOUN')
                creator.domains[down_0_1] = {'SEVEN'}
                creator.domains[down_1_4] = {'NOUN'}
                for var in testset:
                    print(var, creator.domains[var])
                print()
                # revise(across_4_1, down_0_1) (to check overlapping gharacter)
                input(f'Press <Enter> to make [{across_4_1}] comply with [{down_0_1}] ...')
                print_result(creator.revise(across_4_1, down_0_1), 'NO', 'Words removed from domain')
                # if not creator.revise(across_4_1, down_0_1):
                #    print('No ', end='')
                # print('Words removed from domain!')
                # print()
                for var in testset:
                    print(var, creator.domains[var])
                print()
                # revise(across_4_1, down_0_1) AGAIN (to check return False)
                input(f'Press <Enter> to make [{across_4_1}] comply with [{down_0_1}] ...')
                print_result(creator.revise(across_4_1, down_0_1), 'NO', 'Words removed from domain')
                for var in testset:
                    print(var, creator.domains[var])
                print()
                # revise(across_4_1, down_1_4) (to check all words must be different)
                input(f'Press <Enter> to make [{across_4_1}] comply with [{down_1_4}] ...')
                print_result(creator.revise(across_4_1, down_1_4), 'NO', 'Words removed from domain')
                for var in testset:
                    print(var, creator.domains[var])
            elif action == '3':
                # '3': 'ac3'
                creator = CrosswordCreator(crossword_0)
                creator.enforce_node_consistency()
                # Prepare Testcase
                down_0_1 = get_var(crossword_0, cw.Variable.DOWN, 0, 1)
                for var, domain in creator.domains.items():
                    print(var, domain)
                print()
                # Apply ac3()
                input(f'Press <Enter> to test {tests[action]} ...')
                creator.ac3()
                print()
                for var, domain in creator.domains.items():
                    print(var, domain)
                print()
            elif action == 'A':
                # 'A': 'assignment_complete'
                creator = CrosswordCreator(crossword_0)
                # Prepare first testcase
                across_0_1 = get_var(crossword_0, cw.Variable.ACROSS, 0, 1)
                across_4_1 = get_var(crossword_0, cw.Variable.ACROSS, 4, 1)
                down_0_1 = get_var(crossword_0, cw.Variable.DOWN, 0, 1)
                down_1_4 = get_var(crossword_0, cw.Variable.DOWN, 1, 4)
                assignment = dict()
                assignment[across_0_1] = 'SIX'
                assignment[across_4_1] = 'NINE'
                assignment[down_1_4] = 'FIVE'
                print('---')
                creator.print(assignment)
                print('---')
                # Apply assignment_complete(assignment)
                input(f'Press <Enter> to test {tests[action]} ...')
                print_result(creator.assignment_complete(assignment), 'Not yet', 'Complete')
                # Prepare second testcase
                assignment[down_0_1] = 'SEVEN'
                print('---')
                creator.print(assignment)
                print('---')
                # Apply assignment_complete(assignment)
                input(f'Press <Enter> to test {tests[action]} ...')
                print_result(creator.assignment_complete(assignment), 'Not yet', 'Complete')
            elif action == 'C':
                # 'C': 'consistent'
                creator = CrosswordCreator(crossword_0)
                # Prepare first testcase
                across_0_1 = get_var(crossword_0, cw.Variable.ACROSS, 0, 1)
                across_4_1 = get_var(crossword_0, cw.Variable.ACROSS, 4, 1)
                down_0_1 = get_var(crossword_0, cw.Variable.DOWN, 0, 1)
                down_1_4 = get_var(crossword_0, cw.Variable.DOWN, 1, 4)
                assignment = dict()
                assignment[across_4_1] = 'SIX'
                print('---')
                creator.print(assignment)
                print('---')
                # Check Unary Constraint: Correct Length of word
                input(f'Press <Enter> to test {tests[action]} ...')
                print()
                print_result(creator.consistent(assignment), 'NOT', 'Consistent')
                # Prepare second testcase
                assignment[across_4_1] = 'NINE'
                assignment[down_1_4] = 'NINE'
                creator.print(assignment)
                print('---')
                # Check Binary Constraint: All words must be different
                input(f'Press <Enter> to test {tests[action]} ...')
                print()
                print_result(creator.consistent(assignment), 'NOT', 'Consistent')
                # Prepare second testcase
                assignment[down_0_1] = 'SEVEN'
                assignment[down_1_4] = 'FIVE'
                creator.print(assignment)
                print('---')
                # Check Binary Constraint: All words must be different
                input(f'Press <Enter> to test {tests[action]} ...')
                print()
                print_result(creator.consistent(assignment), 'NOT', 'Consistent')
            elif action == 'O':
                # 'O': 'order_domain_values'
                creator = CrosswordCreator(crossword_0)
                # Prepare Testcase
                across_0_1 = get_var(crossword_0, cw.Variable.ACROSS, 0, 1)
                across_4_1 = get_var(crossword_0, cw.Variable.ACROSS, 4, 1)
                down_0_1 = get_var(crossword_0, cw.Variable.DOWN, 0, 1)
                down_1_4 = get_var(crossword_0, cw.Variable.DOWN, 1, 4)
                creator.enforce_node_consistency()
                assignment = dict()
                # assignment[down_0_1] = 'SEVEN'
                # assignment[across_0_1] = 'SIX'
                assignment[down_1_4] = 'FIVE'
                var = across_0_1
                # var = across_4_1
                # var = down_0_1
                # var = down_1_4
                creator.print(assignment)
                for x_var in creator.domains.keys() - assignment.keys():
                    print(x_var, creator.domains[x_var])
                # Check ...
                input(f'Press <Enter> to test {tests[action]} for {var}...')
                print()
                print(var, creator.order_domain_values(var, assignment))

                ### 'O': 'order_domain_values'
                ##creator = CrosswordCreator(crossword_1)
                ### Prepare Testcase
                ##across_2_1 = get_var(crossword_1, cw.Variable.ACROSS, 2, 1)
                ##across_4_4 = get_var(crossword_1, cw.Variable.ACROSS, 4, 4)
                ##across_6_5 = get_var(crossword_1, cw.Variable.ACROSS, 6, 5)
                ##down_2_1 = get_var(crossword_1, cw.Variable.DOWN, 2, 1)
                ##down_1_7 = get_var(crossword_1, cw.Variable.DOWN, 1, 7)
                ##down_1_12 = get_var(crossword_1, cw.Variable.DOWN, 1, 12)
                ##creator.enforce_node_consistency()
                ##assignment = dict()
                ###assignment[across_2_1] = 'INTELLIGENCE'
                ###assignment[down_1_4] = 'FIVE'
                ##var = across_2_1
                ##var = across_4_4
                ###var = across_6_5
                ###var = down_2_1
                ###var = down_1_7
                ###var = down_1_12
                ##creator.print(assignment)
                ##for x_var in creator.domains.keys() - assignment.keys():
                ##    print(x_var, creator.domains[x_var])
                ### Check ...
                ##input(f'Press <Enter> to test {tests[action]} for {var}...')
                ##print()
                ##print(var, creator.order_domain_values(var, assignment))
            elif action == 'U':
                # 'U': 'select_unassigned_variable'
                creator = CrosswordCreator(crossword_0)
                # Prepare First Testcase
                across_0_1 = get_var(crossword_0, cw.Variable.ACROSS, 0, 1)
                across_4_1 = get_var(crossword_0, cw.Variable.ACROSS, 4, 1)
                down_0_1 = get_var(crossword_0, cw.Variable.DOWN, 0, 1)
                down_1_4 = get_var(crossword_0, cw.Variable.DOWN, 1, 4)
                creator.enforce_node_consistency()
                assignment = dict()
                assignment[down_1_4] = 'FIVE'
                creator.print(assignment)
                for x_var in creator.domains.keys() - assignment.keys():
                    print(x_var, creator.domains[x_var])
                print('---')
                for var in creator.domains.keys() - assignment.keys():
                    print(var, creator.get_domain_values(var, assignment), creator.degree(var))
                print('---')
                # Check ...
                input(f'Press <Enter> to test {tests[action]} ...')
                print()
                var = creator.select_unassigned_variable(assignment)
                print(var, creator.get_domain_values(var, assignment))
                # Prepare Second Testcase
                print('=========')
                assignment = dict()
                assignment[across_4_1] = 'NINE'
                creator.print(assignment)
                for x_var in creator.domains.keys() - assignment.keys():
                    print(x_var, creator.domains[x_var])
                print('---')
                for var in creator.domains.keys() - assignment.keys():
                    print(var, creator.get_domain_values(var, assignment), creator.degree(var))
                print('---')
                # Check ...
                input(f'Press <Enter> to test {tests[action]} ...')
                print()
                var = creator.select_unassigned_variable(assignment)
                print(var, creator.get_domain_values(var, assignment))
            elif action == 'B':
                # 'B': 'backtrack'
                while True:
                    # Choose puzzle structure
                    system('cls')
                    print('0: Puzzle Structure 0')
                    print('1: Puzzle Structure 1')
                    print('2: Puzzle Structure 2')
                    print()
                    puzzle = (input('Choose Puzzle Structure (0-2) (<Enter> to exit): ') + ' ')[0]
                    if puzzle == ' ':
                        break
                    elif puzzle == '0':
                        creator = CrosswordCreator(crossword_0)
                    elif puzzle == '1':
                        creator = CrosswordCreator(crossword_1)
                    elif puzzle == '2':
                        creator = CrosswordCreator(crossword_2)
                    else:
                        input('Illegal Choice! Press <Enter> to continue')
                        continue
                    # Prepare Testcase
                    creator.enforce_node_consistency()
                    assignment = dict()
                    # Show Testcase
                    creator.print(assignment)
                    for var, domain in creator.domains.items():
                        print(var, domain)
                    input(f'Press <Enter> to test {tests[action]} ...')
                    print()
                    start = time.perf_counter()
                    creator.enforce_node_consistency()
                    creator.ac3()
                    assignment = creator.backtrack(assignment)
                    end = time.perf_counter()
                    # Print result
                    if assignment is None:
                        print("No solution.")
                    else:
                        creator.print(assignment)
                    print()
                    print(f'Backtracking took {end - start:.4f} seconds')
                    input('Press <Enter> to continue: ')
        # End the chosen Test
        print()
        input('Press Enter to return to menu:')
        system('cls')


if __name__ == "__main__":
    main()