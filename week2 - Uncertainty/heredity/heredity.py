import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]

def prob_gene(num_genes):
    probability = 1

    if num_genes == 1:
        probability *= .5
    elif  num_genes == 2:
        probability *= PROBS["mutation"]
    else:
        probability *= (1 - PROBS["mutation"])
    return probability


def prob_no_gene(num_genes):
    probability = 1

    if num_genes == 1:
        probability *= .5
    elif num_genes == 2:
        probability *= (1 - PROBS["mutation"])
    else:
        probability *= PROBS["mutation"]
    return probability

# For anyone with parents in the data set, each parent will pass one of their two genes on to their child randomly,
# and there is a PROBS["mutation"] chance that it mutates
def conditional_probability(mother, father, one_gene, two_genes, num_child_genes):
    gene_mom = 1
    gene_dad = 1
    num_mom_genes = -1
    num_dad_genes = -1

    if mother in one_gene:
        num_mom_genes = 1
    elif mother in two_genes:
        num_mom_genes = 2
    else:
        num_mom_genes = 0

    if father in one_gene:
        num_dad_genes = 1
    elif father in two_genes:
        num_dad_genes = 2
    else:
        num_dad_genes = 0

    if num_child_genes == 0:
        gene_mom = prob_no_gene(num_mom_genes)
        gene_dad = prob_no_gene(num_dad_genes)
    elif num_child_genes == 1:
        # mother provides gene
        gene_mom = prob_gene(num_mom_genes) * prob_no_gene(num_dad_genes)
        # father provides gene
        gene_dad = prob_gene(num_dad_genes) * prob_no_gene(num_mom_genes)
        return gene_dad + gene_mom
    if num_child_genes == 2:
        gene_mom = prob_gene(num_mom_genes)
        gene_dad = prob_gene(num_dad_genes)

    return gene_dad * gene_mom


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    results = []
    d = {}
    joint = 1
    for person in people:
        # For anyone with no parents listed in the data set, use the probability distribution
        # PROBS["gene"] to determine the probability that they have a particular number of the gene.
        if not people[person]['mother'] and not people[person]['father']:
            # For any person not in one_gene or two_genes, calculate the probability that
            # they have no copies of the gene; and for anyone not in have_trait,
            # we would like to calculate the probability that they do not have the trait.
            if person not in one_gene and person not in two_genes:
                if person in have_trait:
                    prob = round(PROBS["gene"][0] * PROBS["trait"][0][True],4)
                else:
                    prob = round(PROBS["gene"][0] * PROBS["trait"][0][False],4)
                results.append(prob)
                d['zero'] = prob
            # one_gene is a set of all people for whom we want to compute the probability that they have one copy of the gene.
            elif person in one_gene:
                if person in have_trait:
                    prob = round(PROBS["gene"][1] * PROBS["trait"][1][True],4)
                else:
                    prob = round(PROBS["gene"][1] * PROBS["trait"][1][False], 4)
                results.append(prob)
                d['one'] = prob
            # two_genes is a set of all people for whom we want to compute the probability that they have two copies of the gene.
            elif person in two_genes:
                if person in have_trait:
                    prob = round(PROBS["gene"][2] * PROBS["trait"][2][True],4)
                else:
                    prob = round(PROBS["gene"][2] * PROBS["trait"][2][False], 4)
                results.append(prob)
                d['two'] = prob
        else:
            # Calculate conditional probability for anyone with parents in the data set
            if person not in one_gene and person not in two_genes:
                conditional_prob = conditional_probability(people[person]["mother"],
                                                           people[person]["father"],
                                                           one_gene, two_genes, 0)
                if person in have_trait:
                    conditional_prob *= PROBS["trait"][0][True]
                else:
                    conditional_prob *= PROBS["trait"][0][False]

            elif person in one_gene:
                conditional_prob = conditional_probability(people[person]["mother"],
                                                           people[person]["father"],
                                                           one_gene, two_genes, 1)
                if person in have_trait:
                    conditional_prob *= PROBS["trait"][1][True]
                else:
                    conditional_prob *= PROBS["trait"][1][False]
            elif person in two_genes:
                conditional_prob = conditional_probability(people[person]["mother"],
                                                           people[person]["father"],
                                                           one_gene, two_genes, 2)
                if person in have_trait:
                    conditional_prob *= PROBS["trait"][2][True]
                else:
                    conditional_prob *= PROBS["trait"][2][False]

            results.append(conditional_prob)
            d['child'] = conditional_prob

    for result in results:
        joint *= result
    # print(d, joint)
    return joint


# people_data = load_data('./data/family0.csv')
# one_gene = {"Harry"}
# two_genes = {"James"}
# trait = {"James"}
# print(joint_probability(people_data, one_gene, two_genes, trait))


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        if person in one_gene:
            probabilities[person]['gene'][1] += p
            if person in have_trait:
                probabilities[person]['trait'][True] += p
            else:
                probabilities[person]['trait'][False] += p
        elif person in two_genes:
            probabilities[person]['gene'][2] += p
            if person in have_trait:
                probabilities[person]['trait'][True] += p
            else:
                probabilities[person]['trait'][False] += p
        else:
            probabilities[person]['gene'][0] += p
            if person in have_trait:
                probabilities[person]['trait'][True] += p
            else:
                probabilities[person]['trait'][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        # print(person, probabilities[person])
        temp_true = probabilities[person]['trait'][True]
        temp_false = probabilities[person]['trait'][False]
        gene_zero = probabilities[person]['gene'][0]
        gene_one = probabilities[person]['gene'][1]
        gene_two = probabilities[person]['gene'][2]
        probabilities[person]['trait'][True] = temp_true / (temp_true + temp_false)
        probabilities[person]['trait'][False] = temp_false / (temp_true + temp_false)
        probabilities[person]['gene'][0] = gene_zero / (gene_zero + gene_one + gene_two)
        probabilities[person]['gene'][1] = gene_one / (gene_zero + gene_one + gene_two)
        probabilities[person]['gene'][2] = gene_two / (gene_zero + gene_one + gene_two)

if __name__ == "__main__":
    main()
