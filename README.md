# *** PyGAwLLfs: Genetic Algorithm with Linkage Learning in Python ***

This project implements a **Genetic Algorithm with Linkage Learning (GAwLL)** for **Feature Selection**. During its process, the algorithm estimates variable importance and variable interaction.

---

## **Project Structure**

* **`PyGAwLLfs/`**
    * **`src/`** - All source code.
        * **`core/`** - GAwLL core logic (e.g., GAwLL class, eVIG, importance, linkage learning mutation).
        * **`data/`** - Datasets (.dat files).
        * **`models/`** - ML models (Decision Tree, KNN, MLP, RandomForest).
        * **`results/`** - Saved execution outputs (PNG, CSV, TXT).
        * **`util/`** - Utilities (dataset handling, statistics, value saving).
        * **`visualization/`** - Plotting for analysis (interaction graphs, importances).
        * **`main.py`** - Project entry point.
    * **`requirements.txt`** - Python dependencies.

---
## **Data Format**

This project expects datasets in .dat format and must contain some informations about the problem. 
* **MODEL**: name of the dataset. 
* **TYPE**: type of problem (1 for classification and 2 for regression).
* **N_ATTRIBUTES**: total number of features.
* **N_EXAMPLES**: total number of data instances.
* **N_CLASSES**: number of unique classes in the target variable (only for classification problems).
* **DATASET**: this keyword signals the start of the data section.

**Data rows**: Each row represents an instance. Features are space-separated floating-point numbers, and the target variable is the last value in the row.

**Example (`boson.dat`):**

* **`MODEL:`** BOSON
* **`TYPE:`** 1
*  **`N_ATTRIBUTES:`** 28
* **`N_EXAMPLES:`** 52035
* **`N_CLASSES:`** 2
* **`DATASET:`** 
0.09409524016143742 0.5679407716028971 0.6030046997760847 0.2272047042336912 0.4103571856511061 0.13799260150096007 0.4060363444059322 0.04393373374730941 1.0 0.0770715016874201 0.46339839505646385 0.8645376774482116 1.0 0.02866574415584571 0.2689229596616497 0.7099737489713134 0.0 0.004542887262786788 0.27192398782781174 0.4996672837213508 0.0 0.010722967773972105 0.06055278722293059 0.1961549317791694 0.09241450027036263 0.059804938787505896 0.1129895419655916 0.09976952838731448 1
#... more data rows follow

---

## **Algorithm Configuration**

**GAwLL** parameters are primarily defined in `src/main.py`:

* ***`max_generations`***: maximum generations.
* ***`chrom_size`***: number of features (read from `.dat` header).
* ***`mutation_probability`***: gene mutation likelihood (based on `chrom_size`).
* ***`fitness_function`***: evaluates subset quality via ML model performance; penalizes large subsets.
* **ML Model Parameters**: specific ML model settings (e.g., `max_depth` for Random Forest).

Additionally, the **`linkage_learning`** parameter (found in the `GAwLL` class instantiation within `src/main.py`) controls whether the algorithm uses linkage learning (`True`) or a standard genetic algorithm (`False`).

More internal GAwLL variables (e.g., ***`POPULATION_SIZE`***, ***`CROSSOVER_RATE`***, ***`TAU_RESET_GENERATION`***) are in `src/core/gawll.py`.

**When running the algorithm**, you must provide a **random seed** to the `instance.run()` method (e.g., `instance.run(seed=42)`). This ensures reproducibility of results.

To modify parameters, edit `src/main.py` or `src/core/gawll.py` directly.

## **Setup and Run**

### **Prerequisites:**
* **Python 3.13 or higher** is required.

### **Steps:**

1.  **Navigate to Project Root:**
    Open your terminal and go to the `PyGAwLLfs/` directory.
    ```bash
    cd /path/to/PyGAwLLfs/
    ```

2.  **Install Dependencies:**
    Install required Python libraries using `pip`.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare Data:**
    Place your `.dat` dataset files (e.g., `boson.dat`) into the **`data/`** folder at the project's root.

4.  **Execute the Main Script:**
    Run the algorithm. Outputs will be saved in the `results/` folder.
    ```bash
    python -m src.main
    ```

---

## **Results**

The `results/` folder will contain the following outputs:

* **Interaction Graphs (PNG):** Visualizations of the **estimated variable interactions**, including a full graph and a reduced form highlighting stronger relationships.
* **Importance Histogram (PNG):** A plot showing the **estimated importance** for each variable.
* **GAwLL Importance Values (CSV):** Raw numerical data of the **estimated feature importances**.
* **GAwLL Interaction Values (CSV):** A symmetric matrix where each element $a_{ij}$ represents the **estimated interaction value** between variables `i` and `j`.
* **Best Individual (TXT):** A log file detailing the **best feature subset (chromosome)** found by the algorithm and its corresponding **best fitness** score.

---			 
