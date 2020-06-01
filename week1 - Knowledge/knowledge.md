# Knowledge

- **Knowledge-based agents**: agents that reason by operating internal representations of knowledge

## Logic
- **sentence**: an assertion about the world in a knowledge representation language
- Propositional Logic
    - is based on a logic of propositions, or statements about the world
    - Proposition Symbols: P Q R
        - represents a fact or sentence about the world
        - e.g. P represents the world is raining
        - Q represents the fact that Harry visited Harold today.
        - How do we connect these individual facts? Introduce additional symbols called _logical connectives_
- **Logical connectives**
    - NOT - "¬"
    - AND - "∧"
    - OR - "∨" 
    - IMPLICATION - "→" 
    - BICONDITIONAL - "↔" 

## Truth Tables
    - Not (¬)
|  P  | ¬P  |
|-----|-----|
|false|true |
|true |false|
    - And (∧)
|  P  | Q  |  P∧Q |
|-----|-----|-----|
|false|false|false|
|false|true |false|
|true |false|false|
|true |true |true |
    - Or (∨)
|  P  | Q  |  P∨Q |
|-----|-----|-----|
|false|false|false|
|false|true |true |
|true |true |true |
|true |false|true |
    - IMPLICATION (→) 
|  P  | Q  |  P→Q |
|-----|-----|-----|
|false|false|true |
|false|true |true |
|true |false|false|
|true |true |true |
    - BICONDITIONAL (↔) 
|  P  | Q  |  P→Q |
|-----|-----|-----|
|false|false|true |
|false|true |false|
|true |false|false|
|true |true |true |

- **model**: assignment of a truth value to every propositional symbol
    - e.g. P: It is raining. 
           Q: It is a Tuesday.
        - a model assigns values to these: {P=true, Q=false}
    - if there are `n` variables, then there are 2^n possible models
- **knowledge base**: a set of sentences known by a knowledge-based agent
    - representation of knowledge for our AI. Information that our AI knows about the world.
    - AI uses information from our knowledge base to draw conclusions about the rest of the world
- **Entailment**
    - `a` ⊨ `B`
        - In every model n which sentence `a` is true, sentence `B` is also true.
- **Inference**: the process of deriving new sentences from old ones
    - e.g.
        - P: it is a Tuesday.
        - Q: It is raining.
        - R: Harry will go for a run.
    - knowledge base (KB): (P∧ ¬Q) → R
        - {P= true, ¬Q= true} 
            - Inference: R=true
- **Inference Algorithm**: some process we can use to figure out whether or not we can draw some conclusion
    - Goal: to answer a central question about entailment
        - Does KB ⊨ `a` ?
 
## Model Checking algorithm
- remember models are assignments of all of our propositional symbols inside of our language to a truth value
- think of it as a possible world
- model checking algorithm enumerates all of the options
- To determine if KB ⊨ `a`:
    - Enumerate all possible models
    - if in every model where `KB` is true, `a` is true, then `KB` entails `a`
- P: it is a Tuesday.   Q: It is raining.   R: Harry will go for a run.
KB: (P∧ ¬Q) → R       P       ¬Q
Query:  R
|  P  | Q   |  R  |  KB |
|-----|-----|-----|-----|
|false|false|false|  F  |
|false|false|true |  F  |
|false|true |false|  F  |
|false|true |true |  F  |
|true |false|false|  F  |
|true |false|true |**T**|
|true |true |false|  F  |
|true |true |true |  F  |

## Knowledge Engineering
- Take a general purpose problem and distill the problem down to knowledge that is representable by a computer
- Take a computer that is able to do something, e.g. model checking or inference problem, and actually solve the problem
- A few examples of how we can apply logical symbols, and logical formulas to be able to encode the idea of knowledge engineering
    - e.g. Game of Clue
        - Person, place, and weapon
        - we can provide the information, and the AI will doing the inference and figuring out what conclusions it is able to draw
        - e.g.
            - (mustard v plum v scarlet)
            - (ballroom v kitchen v library)
            - (knife v revolver v wrench)
            -               ¬ plum
            - ¬ mustard v ¬library v ¬revolver
    - Logic puzzles
        - Gilderoy, Minerva, Pomona and Horace each belong to a different one of the four houses: Gryffindor, Hufflepull, Ravenclaw,Slytherin House.
        - Gilderoy belongs to Gryffindor or Ravenclas
        - Pomona does not belong to Flytherin
        - Minerva belongs to Gryffindor
        - Propositional Symbols:
            - need 16 symbols, one for each person and house
                e.g. GidleroyGryffindor, PomonaHufflepuff, etc.
            - (PomonaSlytherin -> ¬PomonaHufflepuff)
                - know this for all four people and all four houses
    - Mastermind
        - R G B Y, and guess how many are in the right position
    - Model checking is not an efficient algorithm
        - to be able to model check, we need to take all of our variables and enumerate all of the possibilities they can be in.
        - e.g. for n variables, we need to enumerate 2**n possibilities in order to perform model checking

## Inference Rules
- rules we can apply, that can take knowledge that already exists and translates it to new forms of knowledge
- structured with a horizontal line
    - above the line - a premise, some known truth
    - below the line - the conclusion we arrive at after we apply the logic from the inference rules
- will ultimately be translated into propositional logic
1. `Modus Ponens`
    If it is raining, then Harry is inside
            It is raining.
    _____________________________________
            Harry is inside.
    - formally:
          a -> B
            a
        _________
            B
    - Note this is a completely different approach than model checking, which looks at all possible worlds (permutations), and  see whats true in each of the worlds
    - We do not look at any specific world, we just deal with the knowledge that we know and what conclusions we can draw from that
 2. `And elimination` 
      Harry is friends with Ron and Hermione   
      _____________________________________
      Harry is friends with Hermione
      - formally: a ∧ B -> a
3. `Double Negation Elimination`
        It is not true that Harry did not pass the test.
        _____________________________________
        Harry passed the test.
        - formally: ¬(¬a) -> a
4. `Implication Elimination` - translates if/then statements to OR statements
    If it is raining, then Harry is inside
    _____________________________________
     It is not raining or Harry is inside.
     - formally: 
            a -> B
     ___________________
            ¬a v B
5. `Biconditional Elimination`
    It is raining if and only if Harry is inside.
    ___________________________________________
    If it is raining, then Harry is inside, 
    and if Harry is inside, then it is raining
    - formally:
            a <-> B
     _____________________
       (a -> B) ∧ (B -> a)
 6.`De Morgan's Law` - take an AND and translate to an OR by moving the NOT
    It is not true that both Harry and Ron passed the test
    _____________________________________________________
    Harry did not pass the test or Ron did not pass the test
        - formally: 
               ¬(a ∧ B)
        _____________________
               ¬a v ¬B
7. `De Morgan's Law` 
    It is not true that Harry or Ron passed the test
    _____________________________________________________
    Harry did not pass the test and Ron did not pass the test
    - formally: 
               ¬(a v B)
        _____________________
               ¬a ∧ ¬B
8. `Distributive Property}`
    - formally:
            (a ∧ (B v y))
        _____________________
          (a ∧ B) v (a ∧ y)
    - formally:
            (a v (B ∧ y))
        _____________________
          (a v B) ∧ (a v y)

### How can we use these inference rules to actually draw conclusions, and prove some entailment?
- Recall `Search Problems`
    - initial state
    - actions
    - transition model
    - goal test
    - path cost function 
- now that we have these inference rules that take some set of inferences and propositional logic and get us some new set of sentences and propositional logic
    - we can treat those sentences or sets of sentences as states inside of a search problem
    - e.g. `Theorem Proving`
        - initial state: starting knowledge base
        - actions: inference rules that can be applied
        - transition model: new knowledge base after inference
        - goal test: check statement we're trying to prove
        - path cost function: number of steps in proof

### Resolution
- Based on another inference rule, that will let us prove anything that can be proven with a knowledge base
    - e.g. (Ron is in the Great Hall) v (Hermione is in the library)
                    Ron is not in the Great Hall
            _____________________________________________________
                    Hermione is in the library
    - `Complimentary literals`
        - `Unit resolution rule`:
              P v Q
               ¬P
            ---------
                Q
        - the Q does not have to be a single statement, but it could be multiple, all chained together in a clause
            - e.g. 
             P v Q1 v Q2 v ...v Qn
                      ¬P
            ------------------------
                Q1 v Q2 v ...v Qn
        -  e.g. (Ron is in the Great Hall) v (Hermione is in the library)
                (Ron is not in the Great Hall) v (Harry is sleeping)
                _____________________________________________________
                (Hermione is in the library) v (Harry is sleeping)
        - formally:
                    P v Q
                   ¬P v R
                -------------
                    Q v R
            - similar to the last case, Q or R does not have to be a single statement
                    P v Q1 v Q2 v...v Qn
                   ¬P v R1 v R2 v...v Rm
                ----------------------------
                    Q1 v Q2 v...v Qn v R1 v R2 v...v Rm
- definitions:
    - `clause` = a disjunction (connected with OR) of literals (propositional symbols)
        - e.g. P v Q v R
        - note that clauses are connected with `OR`
    - `conjunctive normal form` = logical sentence that is a conjunction of clauses
        - e.g. (A v B v C) ∧ (D v ¬E) ∧ (F v G)
        - conjuction = connected with `AND`
        - this is a standard form that we can translate a logical sentence into that makes it easy to work with and manipulate 
        - can take any sentence in logic and turn it into a `conjunctive normal form` by applying inference rules and transformations to it
     - What is the process of taking a logical formula and converting it into a `CNF`?
        - we need to take all of the symbols that are not part of `CNF` (biconditionals, implications, etc) and turn them into `CNF
    - Conversion to `CNF`
        1. Eliminate biconditionals (<->)
            - turn (a <-> B) into (a -> B) ∧ (B -> a)
        2. Eliminate implications(->)
            - turn (a -> B) into ¬a v B
        3. Move ¬ inwards using De Morgan's Law
            - turn ¬(a ∧ B) into ¬a v ¬B
        4. Use distributive law to distribute v wherever possible
    - Example:
        - (P v Q) -> R              `eliminate implication`
        - ¬(P v Q) v R              `De Morgan's Law`
        - (¬P ∧ ¬Q) v R             `Distributive Law`
        - (¬P v R) ∧ (¬Q v R)
 
### Inference by Resolution
- The process by which clauses are resolved. Using the resolution rule to draw a conclusion
- based on the same idea
    - e.g.
            P v Q
           ¬P v R
          --------
           (Q v R)
    - e.g.
            P v Q v S
           ¬P v R v S
          --------------
           (Q v S v R v S) => factoring (remove dups) =>  (Q v R v S)
    - e.g.
            P
           ¬P  
         --------
            () == False
- To determine if KB ⊨ `a`:
    - Check if (KB ∧ ¬a) is a contradiction? (Proof by contradiction)
        - If so, then KB ⊨ `a`
        - Otherwise, no entailment.
    - `Resolution Algorithm`:
        1. Convert (KB ∧ ¬a) to `CNF`.
        2. Keep checking to see if we can use resolution to produce a new clause.
            2a. If ever we produce the empty clause (equivalent to False), we have a contradiction and KB ⊨ `a`
            2b. Otherwise, if we can't add new clauses, no entailment.
    - ex:
        - Does (A v B) ∧ (¬B v C) ∧ (¬C) entail A?
            - assume ¬A
            - (A v B) ∧ (¬B v C) ∧ (¬C) ∧ (¬A)
            - 4 clauses: (A v B) (¬B v C) (¬C) (¬A)
            - (¬B v C) (¬C) have `complimentary literals`
                - ¬B must be true
            - (A v B) (¬B) are new `complimentary literals`
                - A must be true
            - (¬A) (A) are new `complimentary literals`
                - () is False. There is a contradiction therefore the KN entails A!
- There are many difference algorithms that can be used for inference, we've only covered a few
- Remember, this is all based off of `propositional logic` where we have individual symbols that are connected by AND, OR, NOT, IMPLIES and BICONDITIONALS
    - there are limitations in propositional logic
        - e.g. Logic puzzle and the Hogwarts example
- Another type of logic is `First-order Logic`, which makes it easier to express certain types of ideas
    - going back to propositional logic, we had symbols for each combinations of expressions, which can only be T or F
        - e.g. Propositional Symbols for:
            - MinervaGryffindor
            - MinervaHufflepuff
            - MinervaRavenclaw
            - MinervaSlytherin
        - Constant Symbols      Predicate Symbols
            Minerva                 Person
            Poona                   House
            Horace                  BelongsTo
            Gilderoy
            Gryffindor
            Hufflepuff
            Ravenclaw
            Slytherin 
    - Examples of First-order Logic:
        - Person(Minerva) = Minerva is a person.
        - House(Gryffindor) = Gryffindor is a house.
        - ¬House(Minerva) = Minerva is not a house.
        - BelongsTo(Minerva, Gryffindor) = Minerva belongs to Gryffindor
    - In `First-order Logic`, we only need one symbol for each person, and house. The predicates are used to express the connection between people and houses
        - Very expressive, while minimizing the number of symbols that need to be created (8, 4 people, 4 houses)
    - Additional Features to express more complex ideas `Quantifiers`:
        - `Universal Quantification (∀)` - Express an idea for all values of a variable
            - for all values of x, some statement will hold true
            - e.g. ∀x. BelongsTo(x, Gryffindor) -> ¬(BelongsTo(x, Hufflepuff))
                - For all objects x, if x belongs to Gryffindor, then x does not belong to Hufflepuff
        - `Existential Quantification (∃)` - Express an idea will be true for some values of a variable (at least one)
            - e.g. ∃x. House(x) ∧ BelongsTo(Minerva, x)
                - There exists an object x such that x is a house and Minerva belongs to x
                - Minerva belongs to a house
        - e.g. ∀x. Person(x) -> (∃y. House(y) ∧ BelongsTo(x, y))
            - for all objects of x, if x is a Person, then there exists an object y that is a house, and x belongs to y.
            - Every person belongs to a house.