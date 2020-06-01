## Optimization
- *optimization* - choosing the best option from a set of options
    - e.g. game playing AI that chooses the best move out of a possible set of moves
- First of the algorithms we'll look at is `local search`
- *local search* - search algorithms that maintain a single node and searches by moving to a neighboring
node
    - maintains a single node/state. Will run this algorithm by maintaining that single node and moving
    ourselves to one of the neighboring nodes throughout the search process.
    - applicable in the context where we do not care about the path at all, and all we care about is the solution.
    - e.g. two types of buildings (houses and hospitals), goal is to find a way to place two hospitals on a map with 
    the objective of minimizing the distance between the houses and the hospitals
        - calculate distance with the `heuristic`, the Manhattan distance (how many rows and columns need to move in a 
        grid layout to get to a hospital) 
    - we can think about the problem more generally as a `state-space landscape`.
        - a *state-space landscape* diagram includes vertical bars that represent a particular state the world could be
        in. 
            - e.g. each vertical bar represents a config of the hospitals
            - the height of the vertical bar represents some function of the state or value of the state (height = cost
            of that particular configuration)
            - with state-space landscapes, we have one of two goals:
                1. maximize the value of the function, e.g. find a *global maximum*
                    - call the function we are trying to optimize the `objective function`, some function that measures 
                    for any given state, how good is that state such tat we can take any state, pass it into the 
                    objective and get a value for how good that state is.
                2. minimize the value of the function, e.g. find a *global minimum*
                    - call the function that we're calculating a `cost function`. Each state can have a monetary, time 
                    or distance cost as some examples.
        - recall in local search we are just maintaining a single state, called the *current state* inside of a node in
        a data structure that keeps track of where we are currently. 
            - From the current state we can move to one of its neighbor states.
### Hill climbing
- *Hill Climbing* is the simplest local search algorithm
    - for example, if we are trying to maximize the value of state (global maximum) the algorithm will only consider 
    the neighbors of the state, and pick the higher of the two. 
        -Repeat until the algorithm reaches a point where both of its neighbors have a value less than the current state
    - The algorithms works in the exact opposite way for a global minimum
    - Pseudocode:
        - function HILL-CLIMB(`problem`):
            `current` = initial state of `problem`
            repeat:
                `neighbor` = highest valued neighbor of `current`
                if `neighbor` not better than `current`:
                    return `current`
                `current` = `neighbor`
    - Limitation of hill climbing algorithm - it may not always give the optimal solution and may get stuck at the 
    `local maxima`
        - local maxima are states that are higher or less than its neighbors, but we have not found the global optimum
        - `flat local maximum` - states that all have the same exact value and thus none of the neighbors are better and 
        will get stuck at these points.
        - `shoulder` - point in the graph that has potential to go up or down, but still unable to make upward or 
        downward progress.
    - Hill Climbing *Variants*
            |      variant      |                   definition                  |
            |-------------------|-----------------------------------------------|
            |  steepest-ascent  | choose the highest-valued neighbor            |
            |    stochastic     | choose randomly from higher-valued neighbors  |
            |   first-choice    | choose the first higher-valued neighbor       |
            |   random-restart  | conduct hill climbing multiple times          |
            | local beam search | chooses the k highest-valued neighbors        |
        - each of the variants still have the same risk of ending up at a local minimum or maximum
            - reduce risk by repeating the process multiple times 
    - Searching through entire state spaces for global optimums becomes very expensive and time intensive especially 
    as the state space gets bigger. Thus using local search algorithms can often be quite good especially if we don't 
    care about the best possible outcome and "good enough" is okay.
    - The real problem with many of these different types of hill climbing variants is that they never make a move that 
    makes out situation worse.
        - These algorithms will always compare with the neighbors and consider if it can do better than its current 
        state and move to one of those neighbors.
        - Ultimately we will need to be able to move to situations that may be locally worse, but allow the algorithms 
        to find a global max or min.

### Simulated Annealing
- simulating a process of a "high-temperature system" where things are moving randomly, but over time the temp decreases
and we settle at an ultimate solution.
- idea revolves around a state-space landscape beginning at an initial state.
    - traditional hill climbing algorithms will always pick the neighbor that provides the optimal solution.
    - but if we want the chance to find the global maximum, we cant always just make good moves. We have to also allow
    the algorithm to make 'bad moves' in the short term, but can allow us to ultimately find the global maximum.
    - of course once we hit a 'global maximum', we do not want to move to a state that is worse than our current state.
        - where the metaphor for annealing comes in. We want to start making more random moves and over time make fewer
        of these random moves.
- Early on, higher "temperature": more likely to accept neighbors that are worse than current state
- Later on, lower "temperature": less likely to accept neighbors that are worse than current state
- Pseudocode:
    - function SIMULATED-ANNEALING(`problem`, `max`):
        `current` = initial state of `problem`
        for t = 1 to `max`:
            T = TEMPERATURE(t)
            `neighbor` = random neighbor of `current`
            ΔE = how much better `neighbor` is than `current`
            if ΔE > 0:
                current = neighbor
            with probability e**ΔE/T set `current` = `neighbor`
        return `current`
        - with the probability e**ΔE/T, we are able to not just randomly choose worse options, but be able to more 
        selectively select items that are only a bit worse (avoid extremes)
- These hill-climbing simulated annealing algorithms have a lot of applications - anytime you can formulate the problem 
to explore a certain config and ask if the neighbors are better than this current config, and have some way of measuring
that
    - e.g. facility location problems, traveling salesman problem
- very famous example `traveling salesman problem` - start with cities and want to find a route that takes the salesman
 through each city and have the salesman end up where they started.
    - Goal: minimize the total distance travelled (cost for the path)
    - e.g. delivery companies
    - turns out this is a very computationally expensive problem (NP Complete), ultimately will need to find an approx.
    that is good enough even if it is not the globally optimum solution
    - try to formulate this traveling salesman as a local search problem. Picking some state, some config and some route
    between the nodes (cities) and measure the cost of that state (distance) and try to minimize the cost.
        - what does it mean to have a neighbor of this state?
            - what happens if we pick two of these edges between nodes and switch the edges

### Linear Programming
- Comes up in the context where we're trying to optimize for some mathematical function.
- Also comes up when we might have real numbered values (not just discrete fixed values).
- Family of types of problems where we might have a situation that looks like:
    - Minimize a cost function c1x1 + c2x2 + ... + cnxn
    - With constraints of form a1x1 + a2x2 + ... + anxn ≤ b or of form a1x1 + a2x2 + ... + anxn = b
    - With bounds for each variable li ≤ xi ≤ ui
- if a problem can be formed in the terms above, then there are a number of algorithms that already exist for these 
problems
    - Two machines X1 and X2. X1 costs $50/hr to run, X2 costs $80/hr to run. Goal is to min cost.
    - X1 requires 5 units of labor per hour. X2 requires 2 units of labor per hour. Total of 20 units of labor to spend.
    - X2 produces 10 units of output per hour. X2 produces 12 units of output per hour. Company needs 90 units of output.
    - Represented as:
        - *Cost Function*: 50x1 + 80x2
        - *Constraint*: 5x1 + 2x2 ≤ 20
        - *Constraint*: 10x1 + 12x2 ≥ 90 = (-10x1) + (-12x2) ≤ -90
            - multiply by -1 to flip sign to ≤
- Some popular algorithms that fit these constraints are`Simplex` and `Interior-Point`.

### Constraint Satisfaction
- Basic idea: we have some number of variables that need to take on some values. We need to figure out what values each 
of the variables should take on. But the variables are subject to constraints that limit what values those variables can
actually take on.
- Real world example:
    - exam scheduling. Exams only happen on select days (e.g. MTW) and we do not want students to take two exams on the 
    same day
        - represent each course as a node in the graph. Create an edge between two nodes if there is a constraint 
        between those two nodes.
    - Family of types of problems where we might have a situation that looks like:
        - Set of variables {X1, X2, ..., Xn}
        - Set of domains for each variable {D1, D2, ..., Dn}
        - Set of constraints C
    - Popular game: *Sudoku*
        - Variables - empty squares in the puzzle. e.g. {(0,2),(1,1),(1,2), ...}
        - Set of domains - {1,2,3,4,5,6,7,8,9} for each variable
        - Constraints - Cells cant be equal to each other. e.g. {(0,2)=/=(1,1)=/=(1,2)=/=...}
- Constraints come in multiple forms:
    - *hard constraints* - constraints that must be satisfied in a correct solution
        - e.g. in Sudoku cannot repeat values in the same row or col.
    - *soft constraints* - constraints that express some notion of which solutions are preferred over others
        - e.g. For exam scheduling, some students prefer one exam is in the morning over another.
- Can classify constraints into different categories
    - *unary constraints* - constraint involving only one variable
        - e.g. {A =/= Monday}
    - *binary constraint* - constraint involving two variables
        - e.g. {A =/= B}

### Node Consistency
- Using knowledge of unary and binary constraints, we can try to make the problem 'node consistent'.
- *node consistency* - when all the values in a variable's domain satisfy the variable's unary constraints
- e.g. two classes A and B. Each of which have an exam in the following domains: {Mon, Tue, Wed}
    - constraints: {A=/=Mon, B=/=Tues, B=/=Mon, A=/=B}
    - node consistency: try to ensure that all of the values for a variable's domain satisfy the constraints above.
        - initially A does not satisfy the constraints (domain includes Mon)
            - can remove Mon from A's domain = {Tue, Wed}
        - initially B does not satisfy the constraints (domain includes Tues, Mon)
            - can remove Tues from B's domain = {Wed}
        - now we considered all of the unary constraints and have made them *node consistent*
        - we have ignored {A=/=B}, but this is a binary constraint since it involves 2 variables.

### Arc Consistency
- *arc consistency* - when all the values in a variable's domain satisfy the variable's binary constraints
    - e.g. if we are looking to make A from the earlier example arc-consistent, we are not longer focused on the 
    unary constraints, but rather the binary constraints {A=/=B}
    - arc is another word for the *edge* that connects two of the nodes in the graph
    - defined more precisely: to make X arc-consistent with respect to Y, remove elements from X's domain
    until every choice for X has a possible choice for Y.
        - e.g. from the previous example, we can made the example arc consistent by:
            - A's domain = {Tue}
            - B's domain = {Wed}
- Pseudocode: Makes X arc consistent, with respect to Y (remove anything in X's domain that does not make for a 
possible option for Y)
    - function REVISE(`csp`,`X`,`Y`):
        `revised` = false
        for x in `X`.domain:
            if no y in ``Y`.domain` satisfies constraint for (`X`,`Y`):
            delete x from `X.domain`
            `revised` = `True`
        return `revised` 
- Generally, when we want to enforce arc consistency, we'll often want to enforce it not just for a single arc, but for
the entire constraint satisfaction problem (csp).
    - function AC-3(`csp`):
        `queue` = all arcs in `csp`
        while `queue` non-empty:
            (X,Y) = DEQUEUE(`queue`)
            if REVISE(`csp`,`X`,`Y`):
                if size of `X.domain` == 0:
                    return `False`
                for each Z in `X.neighbors` - {Y}:
                    ENQUEUE(`queue`, (Z,X))
        return `True`
- arc consistency only considers consistency between a binary constraint (e.g. between two nodes) and we are not 
considering the rest of the nodes just yet.
    - so just using AC3 will not always solve the problem
    - we still need to search to solve the problem, e.g. by using classical traditional search algorithms.
        - recall *Search Problem* consists of:
            - initial state - empty assignment (no variables)
            - actions - add a {variable=value} to assignment
            - transition model - shows how adding an assignment changes the assignment
            - goal test - check if all variables assigned and constraints all satisfied
            - path cost function - all paths have same cost (not relevant)
    - cannot implement naive BFS or DFS searches, would be too inefficient
    - we can take advantage of the structure of the csp itself to improve effiencies
        - one key idea is to order the variables, since it does not matter what order we assign variables in. 
            - does not change the fundamental nature of the assignment.

### Backtracking Search
- The main idea of *backtracking search* is to make assignments from variables to values, and if we ever get stuck or
arrive at a place where there is no way we can make forward progress, we will *backtrack* and try another path instead. 
- Pseudocode:
    - function BACKTRACK(`assignment`, `csp`):
        if `assignment` complete: return `assignment`
        `var` = SELECT-UNASSIGNED-VAR(`assignment`, `csp`)
        for `value` in DOMAIN-VALUES(`var`, `assignment`, `csp`):
            if `value` consistent with `assignment`:
                add {`var`=`value`} to `assignment`
                result = BACKTRACK(`assignment`, `csp`)
                if `result` =/= `failure`: return `result`
            remove {`var`=`value`} from `assignment`
        return `failure`
- Going back to the idea of *Inference* to be more intelligent and make the solution more efficient
    - How can get avoid going down a path that is a dead-end with the knowledge that we have initially?
        - Yes, we can look at the structure of the graph to infer some knowledge
        - without having to do any additional searches, if we can just enforce arc consistency, we can figure out what 
        the assignments of all the variables should be without needing to backtrack. 
            - done by `interleaving` the search process and the inference step by trying to enforce arc consistency.
- *maintaining arc-consistency* - algorithm for enforcing arc-consistency every time we make a new assignment.
    - may be done by using the *AC-3* algorithm at the very beginning of the problem before we even begin searching in 
    order to limit the domain of the variables to make it easier to search.
        - eliminates possible values from domains whereever possible
    - when we make a new assignment to `X`, calls `AC-3`, starting with a queue of all arcs (`Y`,`X`), where Y is a 
    neighbor of X
    - Revised Pseudocode:
        - function BACKTRACK(`assignment`, `csp`):
            if `assignment` complete: return `assignment`
            `var` = SELECT-UNASSIGNED-VAR(`assignment`, `csp`)
            for `value` in DOMAIN-VALUES(`var`, `assignment`, `csp`):
                if `value` consistent with `assignment`:
                    add {`var`=`value`} to `assignment`
                    `inferences` = INFERENCE(`assignment`, `csp`)
                    if `inferences` =/= `failure`: add `inferences` to `assignment`
                    result = BACKTRACK(`assignment`, `csp`)
                    if `result` =/= `failure`: return `result`
                remove {`var`=`value`} and `inferences` from `assignment`
            return `failure`
- there are other heuristics we can use to improve the efficiency of our search process, by improving the functions
within the algorithm
    - `SELECT-UNASSIGNED-VAR` selects a variable in the `csp` that has not already been assigned.
        - following certain heuristics to make the process more efficient
        - *minimum remaining values (MRV) heuristic*: select the variable that has the smallest domain
            - variables with smaller domains are easier to prune to reach a solution
        - *degree heuristic*: select the variable that has the highest degree
            - useful if all nodes have the same MRV. 
            - a *degree* is defined as the number of nodes that are constrained to that node.
                - by choosing a variable with the highest degree, it immediately constrains the rest of the variables 
                more and more likely to eliminate large sections of the state space
    - `DOMAIN-VALUES` takes a variable and returns a sequence of all the values within the domain.
        - we've naively selected values in order from the domain, but it can be more efficient to choose values that 
        are more likely to be a solution first
            - we can look at the number of constraints that have already been added
        - *least-constraining values heuristic*: return variables in order by number of choices that are ruled out for 
        neighboring variables
            - try least-constraining values first
            - by not ruling out many options, we leave open the possibility of finding a solution without needing to 
            backtrack.
- ultimate takeaway is that we can formulate a problem is multiple different ways:
    1. Local Search
    2. Linear Programming
    3. Constraint Satisfaction
    - if we can formulate problems into these known problems then we can use these techniques to solve a wide-variety 
    of problems all in the world of optimization inside of AI.