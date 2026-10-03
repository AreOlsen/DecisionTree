# Decision Trees and ID3

## Decision Trees

A Decision Tree is a supervised machine learning algorithm used for **classification** and **regression**.

The model represents decisions as a tree:

- **Root node**: The first feature used to split the data.
- **Internal nodes**: Decisions based on feature values.
- **Branches**: Possible outcomes of a decision.
- **Leaf nodes**: Final predictions.

For classification, a tree might look conceptually like:

```text
             Outlook?
            /        \
        Sunny       Rain
          |           |
       Humidity?     Wind?
       /      \      /    \
    High    Normal  Yes    No
     |         |     |      |
   No Play   Play   No     Play
```

The goal is to repeatedly split the dataset so that the resulting groups become as homogeneous as possible with respect to the target variable.

## Entropy

ID3 uses **entropy** to measure the impurity or uncertainty of a dataset.

For a set of examples `S`:

```text
H(S) = -Σ p(c) log₂ p(c)
```

where `p(c)` is the proportion of examples belonging to class `c`.

For binary classification:

- Entropy = `0` means all examples belong to the same class.
- Higher entropy means the classes are more mixed.
- Maximum entropy occurs when the classes are equally distributed.

Example:

```text
10 Yes, 0 No  -> Entropy = 0

5 Yes, 5 No   -> Entropy = 1
```

## Information Gain

ID3 selects features using **information gain**.

Information gain measures how much uncertainty is reduced after splitting the dataset using a particular feature.

```text
IG(S, A) = H(S) - Σ (|Sv| / |S|) H(Sv)
```

where:

- `S` is the original dataset.
- `A` is the feature being evaluated.
- `Sv` is the subset produced by a particular value of `A`.

The feature with the **highest information gain** is selected for the next split.

## ID3 Algorithm

ID3 builds the tree recursively.

### Step 1: Calculate Impurity

Calculate the impurity of the current dataset based on the target variable.

### Step 2: Calculate Information Gain

For every available feature, calculate how much information would be gained by splitting on that feature.

### Step 3: Choose the Best Feature

Select the feature with the highest information gain.

### Step 4: Split the Dataset

Create a branch for each possible value of the selected feature.

### Step 5: Repeat

Apply the same process recursively to each resulting subset.

### Step 6: Create Leaf Nodes

A leaf is created when, for example:

- All examples in the subset belong to the same class.
- There are no remaining features.
- A stopping condition has been reached.

## Example

Suppose we want to predict whether someone will play tennis based on:

```text
Outlook
Temperature
Humidity
Wind
```

ID3 evaluates each feature and calculates its information gain.

If `Outlook` has the highest information gain, it becomes the root:

```text
Outlook
├── Sunny
├── Overcast
└── Rain
```

ID3 then evaluates the remaining features separately within each branch.

## Input and Output Requirements

This implementation expects **numerical values only** for both the input features and the output/target values.

- Input features (`X`) must contain numerical values.
- Output/target values (`y`) must also be numerical.
- Categorical values such as strings or text are not supported directly.
- Categorical data must be converted into numerical representations before being passed to the implementation.

For example:

```text
Input:
[1.5, 20.0, 3.2]

Output:
0
```

rather than:

```text
Input:
["red", "large", "yes"]

Output:
"approved"
```

The implementation does not perform categorical encoding automatically.

## Overfitting

A decision tree can continue splitting until it memorizes the training data.

This produces a tree with very low training error but potentially poor performance on unseen data.

Common ways to control this include:

- Limiting tree depth.
- Requiring a minimum number of samples per split.
- Pruning the tree.
