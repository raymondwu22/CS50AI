# Search
- Examples
    - the 15 puzzle 
    - Trying to solve a maze
    - Driving directions (google maps)


#### Search Problems
- Terminologies
    - **agent** - entity that perceives its environment and acts upon that environment
        - e.g. a car in the driving directions
        - e.g. the AI or person solving a 15 puzzle
    - **state** - a config of an agent in its environment
    - **initial state** - state in which the agent begins
    - **actions** - choices that can be made in a state
        - function **actions(s)** returns the set of actions that can be executed in state *s*
            - e.g. 15 puzzle - can slide tile right, left, up, down
    - **transition model** - a description of what state *results* from performing any applicable action in any state
        - function result(s, a) returns the state resulting from performing an action *a* in state *s*
    - **state space** - the set of all states reachable from the initial state by any sequence of actions
    - **goal test** - way to determine whether a given state is a goal state
    - **path cost** - numerical cost associated with a given path
    - **solution** - a sequence of actions that leads from the initial state to the goal state
    - **optimal solution** - a solution that has the *lowest* path cost among all solutions

#### Search Problems Have:
- initial state
- actions
- transition model
- goal test
- path cost function

#### Node - Data Structure ####
a data structure that keeps track of
- a state
- a parent
- an action
- a path cost

#### Approach ####
- Start with a `frontier` that contains the initial state
- repeat:
    - if the frontier is empty, then no solution.
    - remove a note from the frontier
    - if the note is the goal state, return the solution
    - expand node, add resulting nodes to the frontier
- `PROBLEM`: issue occurs when there is a `cycle` in the path
    - can lead to an endless loop
- `OPTIMIZATION`: 
    - keep track of what we have already visited

#### Revised Approach
- Start with a `frontier` that contains the initial state
- Started with an empty `explored` set
- repeat:
    - if the frontier is empty, then no solution.
    - remove a note from the frontier
    - if the note is the goal state, return the solution
    **- add the node to the explored state**
    - expand node, add resulting nodes to the frontier if they aren't already in the frontier or explored set
 
`stack` - last-in first-out data type
- depth-first search: search algorithm that always expands the **deepest** node in the frontier 
    - as long as out maze is finite, it will eventually explore the entire maze and find a solution
    - although DFS will ultimately find a solution, it may not be the optimal solution
    
`queue` - first-in first-out data type
- breadth-first search: search algorithm that always expands the **shallowest** node in the frontier

#### Types of Search algorithms/strategies ####
- uninformed search: search strategy that uses no problem-specific knowledge
    - e.g. DFS and BFS do not care about the structure of the problem. Both algorithms look at the actions available and take those options
- informed search: search strategies that uses problem-specific knowledge to find solutions more efficiently
    - `greedy best-first search`: search algorithms that expands the node that is closest to the goal, as _estimated_ by a heuristic function `h(n)`
        - heuristic function? Manhattan distance
            - how many squares vertically or horizontally are we away from our goal?
    - `A* search`: search algorithm that expands node with lowest value of g(n) + h(n)
        - `g(n)` = cost to reach node
        - `h(n)` = estimated cost to goal
        - optimal if
            - `h(n)` is admissible (never overestimates the true cost), and
            - `h(n)` is consistent (for ever node n and successor `n'` with step cost `c, h(n)≤h(n') + c`)
      
#### Adversarial Search
- e.g. games such as Tic Tac Toe
- `Minimax` works very well for deterministic 2P games
    - Translates all possible outcomes for a game to values like -1, 0, 1
    - MAX (X) aims to maximize score. 
        - tries to Win(1) or Tie(0)
    - MIN (O) aims to minimize score.
        - aims to Win(-1) or Tie(0)
 - `Game`
    - S0: initial state
    - Player(s): returns which player to move in state s
    - Actions(s): returns legal moves in state s
    - Result(s, a): returns state after action `a` taken in state `s
    - Terminal(s): checks if state s is a terminal state    
    - Utility(s): final numberical value for terminal state `s`
- Initial State
    - empty game board (2D array)
    - Player(s)
        - Assume X is P1
    - Actions(s)
        - Takes game state and output a set of possible actions
    - Result(s,a)
    - Terminal(s)
        - return a boolean 
    - Utility(s)
        - return a value for the terminal state
- pseudocode:
    - given a state s:
        - MAX picks action `a` in ACTIONS(s) that produces highest value of MIN-VALUE(RESULT(s,a))
        - MIN picks action `a` in ACTIONS(s) that produces smallest value of MAX-VALUE(RESULT(s,a))
    - Implementation:
    
         
         function MAX-VALUE(state):
             if TERMINAL(state):
                return UTILITY(state)
             v = -infinity
             for action in ACTIONS(state):
                v = MAX(v, MIN-VALUE(RESULT(state,action)))
             return v
         
         function MIN-VALUE(state):
             if TERMINAL(state):
                return UTILITY(state)
             v = infinity
             for action in ACTIONS(state):
                v = MIN(v, MAX-VALUE(RESULT(state,action)))
             return v
    
 #### Optimizations
- Alpha-beta pruning
    - keeps track the best/worst terminal scores 
    - Improves efficiency by removing nodes through this optimization
    - Stops searching for scores if we know that branch already has a score better/worse than the current max or min value 
- Better approach
    - Depth-Limited minimax
        - after a certain number of moves, stop considering additional moves that come after that (e.g. 10 - 12 moves deep)
        - problem: how do we score these game states without any resolution?
            - add: evaluation function
                - function that estimates the expected utility of the game from a given state